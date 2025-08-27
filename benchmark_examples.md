# API Benchmarking Examples

This document shows common usage patterns for the `benchmark_api.py` script.

## Prerequisites

1. Start the CAPP GPT API server:
   ```bash
   uv run python -m uvicorn capp_gpt_serve.main:app --host 0.0.0.0 --port 8000
   ```

2. Install additional dependencies for benchmarking:
   ```bash
   pip install aiohttp  # If not already installed
   ```

## Basic Usage

### Quick health check benchmark
```bash
python benchmark_api.py --endpoint health --requests 50
```

### Benchmark all endpoints with default settings
```bash
python benchmark_api.py
```

### Focus on the main prediction endpoint
```bash
python benchmark_api.py --endpoint predict --requests 100 --concurrency 5
```

## Performance Testing Scenarios

### 1. Sequential Performance Test
Test individual request latency:
```bash
python benchmark_api.py --test-type sequential --requests 100 --endpoint predict
```

### 2. Concurrent Load Test
Test throughput under concurrent load:
```bash
python benchmark_api.py --test-type concurrent --requests 200 --concurrency 20 --endpoint predict
```

### 3. Sustained Load Test
Test performance over time:
```bash
python benchmark_api.py --test-type sustained --duration 60 --concurrency 10 --endpoint predict
```

### 4. Comprehensive Benchmark
Run all test types on all endpoints:
```bash
python benchmark_api.py --requests 100 --concurrency 15 --duration 30
```

## Advanced Usage

### High Concurrency Test
```bash
python benchmark_api.py --concurrency 50 --requests 500 --endpoint predict --timeout 60
```

### Quick Smoke Test
```bash
python benchmark_api.py --requests 10 --no-warmup --endpoint health
```

### Custom Server Location
```bash
python benchmark_api.py --url http://production-server:8000 --requests 100
```

## Interpreting Results

The benchmark script provides detailed metrics:

### Response Time Metrics
- **Min/Max**: Fastest and slowest response times
- **Average**: Mean response time
- **Percentiles**: 
  - P50 (median): 50% of requests were faster than this
  - P90: 90% of requests were faster than this
  - P95: 95% of requests were faster than this
  - P99: 99% of requests were faster than this

### Throughput Metrics
- **Requests/second**: Average throughput during the test
- **Success rate**: Percentage of successful requests

### Distribution Histogram
Shows how response times are distributed across different ranges.

## Example Output

```
🎯 CAPP GPT API Benchmark
Target: http://localhost:8000
Timestamp: 2024-01-15 14:30:45
------------------------------------------------------------
🔍 Testing connectivity...
✅ API is reachable
🔥 Warming up with 5 requests...
✅ Warmup completed

📊 Concurrent Test - Predict Endpoint
============================================================
Total requests:     100
Successful:         98 (98.0%)
Failed:             2
Total time:         12.45s
Requests/second:    8.03

Response Times (ms):
  Min:              245.67
  Max:              1,234.56
  Average:          456.78
  P50 (median):     423.45
  P90:              678.90
  P95:              789.12
  P99:              1,123.45

Response Time Distribution:
   245.7- 344.6ms: ████████████████████████████████ (32)
   344.6- 443.4ms: ████████████████████████████████████████ (40)
   443.4- 542.3ms: ████████████████████ (20)
   542.3- 641.1ms: ████████ (8)
   641.1- 740.0ms: ████ (4)
   ...
```

## Performance Baselines

### Typical Response Times (on local machine)
- **Health endpoint**: 1-5ms
- **Tokens endpoint**: 5-15ms  
- **Predict endpoint**: 200-2000ms (depends on model complexity)

### Recommended Concurrency Levels
- **Development**: 1-5 concurrent requests
- **Testing**: 10-20 concurrent requests
- **Load Testing**: 20-50 concurrent requests

## Troubleshooting

### Connection Errors
- Ensure the API server is running
- Check the URL and port
- Verify firewall settings

### Timeout Errors
- Increase `--timeout` value for slower responses
- Reduce concurrency level
- Check server resources (CPU, memory)

### High Error Rates
- Reduce concurrency level
- Check server logs for errors
- Verify request payload format