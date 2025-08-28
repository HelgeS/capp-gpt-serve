"""Test script for the CAPP GPT API."""

import requests
import json
import time

API_BASE = "http://localhost:8000"


def test_health_endpoint():
    """Test the health check endpoint."""
    print("Testing health endpoint...")
    response = requests.get(f"{API_BASE}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()


def test_tokens_endpoint():
    """Test the tokens endpoint."""
    print("Testing tokens endpoint...")
    response = requests.get(f"{API_BASE}/tokens")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print("Available token categories:")
        for category, tokens in data.items():
            print(f"  {category}: {len(tokens)} tokens")
            if tokens:
                print(f"    Examples: {tokens[:3]}")
    print()


def test_prediction_endpoint():
    """Test the prediction endpoint."""
    print("Testing prediction endpoint...")

    # Example request
    request_data = {
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
    }

    print(f"Request: {json.dumps(request_data, indent=2)}")

    start_time = time.time()
    response = requests.post(f"{API_BASE}/predict", json=request_data)
    request_time = (time.time() - start_time) * 1000

    print(f"Status: {response.status_code}")
    print(f"Request time: {request_time:.2f}ms")

    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
    else:
        print(f"Error: {response.text}")
    print()


def main():
    """Run all tests."""
    print("CAPP GPT API Test Suite")
    print("=" * 30)

    try:
        test_health_endpoint()
        test_tokens_endpoint()
        test_prediction_endpoint()
        print("All tests completed!")
    except requests.exceptions.ConnectionError:
        print(
            "Error: Could not connect to API. Make sure the server is running on http://localhost:8000"
        )
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
