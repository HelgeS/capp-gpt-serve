"""Benchmarking script for the CAPP GPT API service."""

import asyncio
import aiohttp
import time
import statistics
import argparse
import sys
from dataclasses import dataclass, field
from typing import List, Dict
from datetime import datetime
import random


@dataclass
class BenchmarkResult:
    """Result of a single API request."""

    success: bool
    response_time: float  # in milliseconds
    status_code: int
    response_size: int = 0
    error_message: str = ""


@dataclass
class BenchmarkStats:
    """Statistics from a benchmark run."""

    total_requests: int
    successful_requests: int
    failed_requests: int
    total_time: float  # in seconds
    response_times: List[float] = field(default_factory=list)  # in milliseconds

    @property
    def success_rate(self) -> float:
        """Calculate success rate as percentage."""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100

    @property
    def requests_per_second(self) -> float:
        """Calculate requests per second."""
        if self.total_time == 0:
            return 0.0
        return self.total_requests / self.total_time

    @property
    def avg_response_time(self) -> float:
        """Average response time in milliseconds."""
        if not self.response_times:
            return 0.0
        return statistics.mean(self.response_times)

    @property
    def min_response_time(self) -> float:
        """Minimum response time in milliseconds."""
        if not self.response_times:
            return 0.0
        return min(self.response_times)

    @property
    def max_response_time(self) -> float:
        """Maximum response time in milliseconds."""
        if not self.response_times:
            return 0.0
        return max(self.response_times)

    def percentile(self, p: float) -> float:
        """Calculate percentile of response times."""
        if not self.response_times:
            return 0.0
        sorted_times = sorted(self.response_times)
        k = (len(sorted_times) - 1) * p / 100
        f = int(k)
        c = int(k) + 1
        if f == c:
            return sorted_times[f]
        return sorted_times[f] * (c - k) + sorted_times[c] * (k - f)


class APIBenchmark:
    """Benchmark runner for CAPP GPT API."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)

        # Sample request variations for testing
        self.sample_requests = [
            {
                "part_characteristics": {
                    "geometry": "geometry_prismatic",
                    "holes": "holes_normal",
                    "external_threads": "external_threads_yes",
                    "surface_finish": "surface_finish_normal",
                    "tolerance": "tolerance_medium",
                    "batch_size": "batch_size_small",
                },
                "max_processes": 5,
                "temperature": 0.8,
                "include_confidence": True,
            },
            {
                "part_characteristics": {
                    "geometry": "geometry_pure_axisymmetric",
                    "holes": "holes_none",
                    "external_threads": "external_threads_no",
                    "surface_finish": "surface_finish_very_good",
                    "tolerance": "tolerance_tight",
                    "batch_size": "batch_size_large",
                },
                "max_processes": 10,
                "temperature": 1.0,
                "include_confidence": True,
            },
            {
                "part_characteristics": {
                    "geometry": "geometry_unconventional",
                    "holes": "holes_large",
                    "external_threads": "external_threads_yes",
                    "surface_finish": "surface_finish_rough",
                    "tolerance": "tolerance_rough",
                    "batch_size": "batch_size_medium",
                },
                "max_processes": 15,
                "temperature": 0.5,
                "include_confidence": False,
            },
        ]

    async def make_request(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        method: str = "GET",
        data: Dict = None,
    ) -> BenchmarkResult:
        """Make a single API request and measure performance."""
        start_time = time.time()

        try:
            url = f"{self.base_url}{endpoint}"

            if method.upper() == "POST":
                async with session.post(url, json=data) as response:
                    content = await response.read()
                    response_time = (time.time() - start_time) * 1000  # Convert to ms

                    return BenchmarkResult(
                        success=response.status < 400,
                        response_time=response_time,
                        status_code=response.status,
                        response_size=len(content),
                        error_message=(
                            "" if response.status < 400 else f"HTTP {response.status}"
                        ),
                    )
            else:
                async with session.get(url) as response:
                    content = await response.read()
                    response_time = (time.time() - start_time) * 1000  # Convert to ms

                    return BenchmarkResult(
                        success=response.status < 400,
                        response_time=response_time,
                        status_code=response.status,
                        response_size=len(content),
                        error_message=(
                            "" if response.status < 400 else f"HTTP {response.status}"
                        ),
                    )

        except Exception as e:
            response_time = (time.time() - start_time) * 1000  # Convert to ms
            return BenchmarkResult(
                success=False,
                response_time=response_time,
                status_code=0,
                response_size=0,
                error_message=str(e),
            )

    async def warmup(self, session: aiohttp.ClientSession, requests: int = 5):
        """Perform warmup requests to prepare the service."""
        print(f"🔥 Warming up with {requests} requests...")

        for _ in range(requests):
            await self.make_request(session, "/health")
            await asyncio.sleep(0.1)

        # One prediction request for model warmup
        await self.make_request(session, "/predict", "POST", self.sample_requests[0])
        print("✅ Warmup completed")

    async def benchmark_sequential(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        requests: int,
        method: str = "GET",
        data: Dict = None,
    ) -> BenchmarkStats:
        """Run sequential benchmark (one request at a time)."""
        print(f"🚀 Running sequential benchmark: {requests} requests to {endpoint}")

        results = []
        start_time = time.time()

        for i in range(requests):
            if i > 0 and i % 10 == 0:
                print(f"  Progress: {i}/{requests} ({(i/requests)*100:.1f}%)")

            # For prediction endpoint, use random sample data
            request_data = data
            if endpoint == "/predict" and not data:
                request_data = random.choice(self.sample_requests)

            result = await self.make_request(session, endpoint, method, request_data)
            results.append(result)

            # Small delay to avoid overwhelming the server
            await asyncio.sleep(0.01)

        total_time = time.time() - start_time

        return self._calculate_stats(results, total_time)

    async def benchmark_concurrent(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        requests: int,
        concurrency: int,
        method: str = "GET",
        data: Dict = None,
    ) -> BenchmarkStats:
        """Run concurrent benchmark with specified concurrency level."""
        print(
            f"🚀 Running concurrent benchmark: {requests} requests, {concurrency} concurrent"
        )

        # Create semaphore to limit concurrency
        semaphore = asyncio.Semaphore(concurrency)

        async def limited_request():
            async with semaphore:
                request_data = data
                if endpoint == "/predict" and not data:
                    request_data = random.choice(self.sample_requests)
                return await self.make_request(session, endpoint, method, request_data)

        start_time = time.time()

        # Create all tasks
        tasks = [limited_request() for _ in range(requests)]

        # Execute with progress reporting
        results = []
        completed = 0

        for coro in asyncio.as_completed(tasks):
            result = await coro
            results.append(result)
            completed += 1

            if completed % max(1, requests // 10) == 0:
                print(
                    f"  Progress: {completed}/{requests} ({(completed/requests)*100:.1f}%)"
                )

        total_time = time.time() - start_time

        return self._calculate_stats(results, total_time)

    async def benchmark_sustained(
        self,
        session: aiohttp.ClientSession,
        endpoint: str,
        duration_seconds: int,
        concurrency: int,
        method: str = "GET",
        data: Dict = None,
    ) -> BenchmarkStats:
        """Run sustained load test for specified duration."""
        print(
            f"🚀 Running sustained benchmark: {duration_seconds}s duration, {concurrency} concurrent"
        )

        semaphore = asyncio.Semaphore(concurrency)
        results = []
        start_time = time.time()
        end_time = start_time + duration_seconds

        async def sustained_request():
            async with semaphore:
                request_data = data
                if endpoint == "/predict" and not data:
                    request_data = random.choice(self.sample_requests)
                return await self.make_request(session, endpoint, method, request_data)

        # Keep making requests until time is up
        request_count = 0
        while time.time() < end_time:
            batch_size = min(concurrency * 2, 50)  # Process in batches
            batch_end_time = min(time.time() + 1.0, end_time)  # 1 second batches

            tasks = []
            while time.time() < batch_end_time and len(tasks) < batch_size:
                tasks.append(sustained_request())

            if tasks:
                batch_results = await asyncio.gather(*tasks)
                results.extend(batch_results)
                request_count += len(batch_results)

                # Progress update every ~5 seconds
                if request_count % (concurrency * 5) == 0:
                    elapsed = time.time() - start_time
                    rps = request_count / elapsed
                    print(
                        f"  Progress: {elapsed:.1f}s elapsed, {request_count} requests, {rps:.1f} RPS"
                    )

        total_time = time.time() - start_time

        return self._calculate_stats(results, total_time)

    def _calculate_stats(
        self, results: List[BenchmarkResult], total_time: float
    ) -> BenchmarkStats:
        """Calculate statistics from benchmark results."""
        successful_results = [r for r in results if r.success]
        failed_results = [r for r in results if not r.success]

        stats = BenchmarkStats(
            total_requests=len(results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            total_time=total_time,
            response_times=[r.response_time for r in successful_results],
        )

        return stats

    def print_stats(self, stats: BenchmarkStats, title: str):
        """Print detailed statistics."""
        print(f"\n📊 {title}")
        print("=" * 60)
        print(f"Total requests:     {stats.total_requests:,}")
        print(
            f"Successful:         {stats.successful_requests:,} ({stats.success_rate:.1f}%)"
        )
        print(f"Failed:             {stats.failed_requests:,}")
        print(f"Total time:         {stats.total_time:.2f}s")
        print(f"Requests/second:    {stats.requests_per_second:.2f}")

        if stats.response_times:
            print("\nResponse Times (ms):")
            print(f"  Min:              {stats.min_response_time:.2f}")
            print(f"  Max:              {stats.max_response_time:.2f}")
            print(f"  Average:          {stats.avg_response_time:.2f}")
            print(f"  P50 (median):     {stats.percentile(50):.2f}")
            print(f"  P90:              {stats.percentile(90):.2f}")
            print(f"  P95:              {stats.percentile(95):.2f}")
            print(f"  P99:              {stats.percentile(99):.2f}")

        self._print_histogram(stats.response_times)

    def _print_histogram(self, response_times: List[float], bins: int = 10):
        """Print a simple text histogram of response times."""
        if not response_times:
            return

        print("\nResponse Time Distribution:")

        min_time = min(response_times)
        max_time = max(response_times)
        bin_width = (max_time - min_time) / bins

        if bin_width == 0:
            print(f"  All responses: {min_time:.2f}ms")
            return

        # Create bins
        bin_counts = [0] * bins
        for time_ms in response_times:
            bin_index = min(int((time_ms - min_time) / bin_width), bins - 1)
            bin_counts[bin_index] += 1

        max_count = max(bin_counts)
        scale = 50 / max_count if max_count > 0 else 1

        for i, count in enumerate(bin_counts):
            bin_start = min_time + i * bin_width
            bin_end = min_time + (i + 1) * bin_width
            bar_length = int(count * scale)
            bar = "█" * bar_length
            print(f"  {bin_start:6.1f}-{bin_end:6.1f}ms: {bar} ({count})")


async def main():
    """Main benchmark runner."""
    parser = argparse.ArgumentParser(description="Benchmark CAPP GPT API")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="API base URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--requests",
        "-r",
        type=int,
        default=100,
        help="Number of requests for sequential/concurrent tests (default: 100)",
    )
    parser.add_argument(
        "--concurrency",
        "-c",
        type=int,
        default=10,
        help="Concurrency level for concurrent tests (default: 10)",
    )
    parser.add_argument(
        "--duration",
        "-d",
        type=int,
        default=30,
        help="Duration in seconds for sustained test (default: 30)",
    )
    parser.add_argument(
        "--endpoint",
        choices=["health", "tokens", "predict", "all"],
        default="all",
        help="Which endpoint to benchmark (default: all)",
    )
    parser.add_argument(
        "--test-type",
        choices=["sequential", "concurrent", "sustained", "all"],
        default="all",
        help="Which test type to run (default: all)",
    )
    parser.add_argument("--no-warmup", action="store_true", help="Skip warmup requests")
    parser.add_argument(
        "--timeout",
        type=int,
        default=30,
        help="Request timeout in seconds (default: 30)",
    )

    args = parser.parse_args()

    benchmark = APIBenchmark(args.url, args.timeout)

    print("🎯 CAPP GPT API Benchmark")
    print(f"Target: {args.url}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("-" * 60)

    # Test connectivity first
    async with aiohttp.ClientSession(timeout=benchmark.timeout) as session:
        print("🔍 Testing connectivity...")
        result = await benchmark.make_request(session, "/health")
        if not result.success:
            print(f"❌ Failed to connect to API: {result.error_message}")
            sys.exit(1)
        print("✅ API is reachable")

        # Warmup
        if not args.no_warmup:
            await benchmark.warmup(session)

        # Define endpoints to test
        endpoints = []
        if args.endpoint == "all":
            endpoints = [
                # ("/health", "GET", None),
                # ("/tokens", "GET", None),
                ("/predict", "POST", None)
            ]
        else:
            endpoint_map = {
                "health": ("/health", "GET", None),
                "tokens": ("/tokens", "GET", None),
                "predict": ("/predict", "POST", None),
            }
            endpoints = [endpoint_map[args.endpoint]]

        # Run benchmarks
        for endpoint, method, data in endpoints:
            endpoint_name = endpoint.replace("/", "")

            if args.test_type in ["sequential", "all"]:
                stats = await benchmark.benchmark_sequential(
                    session, endpoint, args.requests, method, data
                )
                benchmark.print_stats(
                    stats, f"Sequential Test - {endpoint_name.title()} Endpoint"
                )

            if args.test_type in ["concurrent", "all"]:
                stats = await benchmark.benchmark_concurrent(
                    session, endpoint, args.requests, args.concurrency, method, data
                )
                benchmark.print_stats(
                    stats, f"Concurrent Test - {endpoint_name.title()} Endpoint"
                )

            if args.test_type in ["sustained", "all"]:
                stats = await benchmark.benchmark_sustained(
                    session, endpoint, args.duration, args.concurrency, method, data
                )
                benchmark.print_stats(
                    stats, f"Sustained Test - {endpoint_name.title()} Endpoint"
                )

    print("\n🎉 Benchmark completed!")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️  Benchmark interrupted by user")
        sys.exit(1)
