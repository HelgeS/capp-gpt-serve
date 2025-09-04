"""Validation test script for CAPP GPT API.

This script tests all part-process chain pairs from parts_and_process_chains.json
against the API service to validate correctness of predictions.
"""

import requests
import json
import time
from typing import Dict, List, Tuple, Any


API_BASE = "http://localhost:8000"


def load_test_data(file_path: str) -> List[Dict[str, Any]]:
    """Load test data from JSON file."""
    with open(file_path, "r") as f:
        return json.load(f)


def convert_part_features_to_api_format(
    part_features: Dict[str, str],
) -> Dict[str, str]:
    """Convert part features to API format with proper prefixes."""
    # Map feature keys to API format with prefixes
    api_format = {}
    for key, value in part_features.items():
        if key == "geometry":
            api_format[key] = f"geometry_{value}"
        elif key == "holes":
            api_format[key] = f"holes_{value}"
        elif key == "external_threads":
            api_format[key] = f"external_threads_{value}"
        elif key == "surface_finish":
            api_format[key] = f"surface_finish_{value}"
        elif key == "tolerance":
            api_format[key] = f"tolerance_{value}"
        elif key == "batch_size":
            api_format[key] = f"batch_size_{value}"
        else:
            api_format[key] = value

    return api_format


def compare_process_chains(expected: List[str], actual: List[str]) -> bool:
    """Compare two process chains for exact match (same items in same order)."""
    return expected == actual


def make_prediction_request(
    part_characteristics: Dict[str, str], max_processes: int = 10
) -> Tuple[bool, List[List[str]], str]:
    """Make a prediction request to the API."""
    request_data = {
        "part_characteristics": part_characteristics,
        "max_processes": max_processes,
        "temperature": 0.1,  # Low temperature for more deterministic results
        "include_confidence": True,
    }

    try:
        response = requests.post(f"{API_BASE}/predict", json=request_data, timeout=30)

        if response.status_code == 200:
            result = response.json()
            # Extract process chains from the response
            if "process_chains" in result:
                return True, result["process_chains"], ""
            else:
                print(result)
                return False, [], "No process_chains in response"
        else:
            return False, [], f"HTTP {response.status_code}: {response.text}"

    except requests.exceptions.RequestException as e:
        return False, [], f"Request error: {str(e)}"


def test_single_entry(entry: Dict[str, Any], entry_index: int) -> Dict[str, Any]:
    """Test a single entry from the test data."""
    part_features = entry["part_features"]
    expected_chains = entry["process_chains"]

    print(f"Testing entry {entry_index + 1}: {part_features}")

    # Convert to API format
    api_features = convert_part_features_to_api_format(part_features)

    # Make API request
    success, predicted_chains, error_msg = make_prediction_request(
        api_features, max_processes=len(expected_chains)
    )

    if not success:
        print(f"  ❌ API request failed: {error_msg}")
        return {
            "entry_index": entry_index,
            "part_features": part_features,
            "api_success": False,
            "error": error_msg,
            "expected_chains": expected_chains,
            "predicted_chains": [],
            "individual_chain_results": [],
            "all_chains_correct": False,
            "correct_chain_count": 0,
            "total_chain_count": len(expected_chains),
        }

    # Compare each predicted chain with expected chains
    individual_results = []
    correct_count = 0

    print(
        f"  Expected {len(expected_chains)} chains, got {len(predicted_chains)} chains"
    )

    # Check each expected chain against all predicted chains
    for i, expected_chain in enumerate(expected_chains):
        found_match = False
        for j, predicted_chain in enumerate(predicted_chains):
            if compare_process_chains(expected_chain, predicted_chain):
                found_match = True
                print(
                    f"  ✅ Expected chain {i + 1} matches predicted chain {j + 1}: {expected_chain}"
                )
                break

        if not found_match:
            print(f"  ❌ Expected chain {i + 1} not found: {expected_chain}")

        individual_results.append(
            {"expected_chain": expected_chain, "found_match": found_match}
        )

        if found_match:
            correct_count += 1

    # If there were failures, show all predicted chains for debugging
    if correct_count < len(expected_chains):
        print(f"  📋 All predicted chains ({len(predicted_chains)}):")
        for i, chain in enumerate(predicted_chains):
            print(f"    {i + 1}: {chain}")
        print(f"  📋 All expected chains ({len(expected_chains)}):")
        for i, chain in enumerate(expected_chains):
            print(f"    {i + 1}: {chain}")

    # Check if all expected chains were found
    all_correct = correct_count == len(expected_chains)

    if all_correct:
        print(f"  ✅ All {len(expected_chains)} chains correct!")
    else:
        print(f"  ❌ Only {correct_count}/{len(expected_chains)} chains correct")

    print()

    return {
        "entry_index": entry_index,
        "part_features": part_features,
        "api_success": True,
        "error": None,
        "expected_chains": expected_chains,
        "predicted_chains": predicted_chains,
        "individual_chain_results": individual_results,
        "all_chains_correct": all_correct,
        "correct_chain_count": correct_count,
        "total_chain_count": len(expected_chains),
    }


def run_validation_tests(
    json_file_path: str = "data/parts_and_process_chains.json",
) -> Dict[str, Any]:
    """Run validation tests on all entries in the JSON file."""
    print("CAPP GPT API Validation Test Suite")
    print("=" * 50)
    print()

    # Load test data
    try:
        test_data = load_test_data(json_file_path)
        print(f"Loaded {len(test_data)} test entries from {json_file_path}")
    except Exception as e:
        print(f"Error loading test data: {e}")
        return {"error": str(e)}

    # Test API connectivity
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.status_code != 200:
            print(f"Warning: Health check failed with status {response.status_code}")
        else:
            print("✅ API connectivity confirmed")
    except Exception as e:
        print(f"❌ Cannot connect to API at {API_BASE}: {e}")
        return {"error": f"API connection failed: {e}"}

    print()

    # Run tests
    start_time = time.time()
    results = []

    for i, entry in enumerate(test_data):
        result = test_single_entry(entry, i)
        results.append(result)

        # Add a small delay to avoid overwhelming the API
        time.sleep(0.1)
        # break

    total_time = time.time() - start_time

    # Calculate statistics
    total_entries = len(results)
    successful_requests = sum(1 for r in results if r["api_success"])
    failed_requests = total_entries - successful_requests

    entries_with_all_correct = sum(1 for r in results if r["all_chains_correct"])

    total_individual_chains = sum(r["total_chain_count"] for r in results)
    correct_individual_chains = sum(r["correct_chain_count"] for r in results)

    # Print summary
    print("=" * 50)
    print("VALIDATION RESULTS SUMMARY")
    print("=" * 50)
    print(f"Total test entries: {total_entries}")
    print(f"Successful API requests: {successful_requests}")
    print(f"Failed API requests: {failed_requests}")
    print(f"Total execution time: {total_time:.2f} seconds")
    print()
    print("CORRECTNESS STATISTICS:")
    print(
        f"Entries with ALL process chains correct: {entries_with_all_correct}/{total_entries} ({100 * entries_with_all_correct / total_entries:.1f}%)"
    )
    print(
        f"Individual process chains correct: {correct_individual_chains}/{total_individual_chains} ({100 * correct_individual_chains / total_individual_chains:.1f}%)"
    )
    print()

    # Show failed entries
    if failed_requests > 0:
        print("FAILED REQUESTS:")
        for result in results:
            if not result["api_success"]:
                print(f"  Entry {result['entry_index'] + 1}: {result['error']}")
        print()

    # Show entries with incorrect predictions
    incorrect_entries = [
        r for r in results if r["api_success"] and not r["all_chains_correct"]
    ]
    if incorrect_entries:
        print(f"ENTRIES WITH INCORRECT PREDICTIONS ({len(incorrect_entries)}):")
        for result in incorrect_entries:
            print(
                f"  Entry {result['entry_index'] + 1}: {result['correct_chain_count']}/{result['total_chain_count']} correct"
            )
            print(f"    Features: {result['part_features']}")
        print()

    return {
        "total_entries": total_entries,
        "successful_requests": successful_requests,
        "failed_requests": failed_requests,
        "entries_all_correct": entries_with_all_correct,
        "total_individual_chains": total_individual_chains,
        "correct_individual_chains": correct_individual_chains,
        "execution_time": total_time,
        "detailed_results": results,
    }


def main():
    """Main function to run the validation tests."""
    try:
        results = run_validation_tests()

        if "error" in results:
            print(f"Test suite failed: {results['error']}")
            return 1

        # Save detailed results to file
        output_file = "validation_results.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Detailed results saved to {output_file}")

        return 0

    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        return 1
    except Exception as e:
        print(f"Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
