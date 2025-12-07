import pytest
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import os

# Global dictionary to collect test results
TEST_RESULTS = defaultdict(lambda: defaultdict(dict))


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """Hook to capture test results after each test run."""
    outcome = yield
    report = outcome.get_result()

    # Only process if this is a test_extract_calories test in the call phase
    if report.when == "call" and item.name.startswith("test_extract_calories"):
        # Extract test_id and provider from test parameters
        if hasattr(item, "callspec"):
            params = item.callspec.params
            test_id = params.get("id")
            provider = params.get("provider")

            if test_id and provider:
                provider_name = provider["model"] + (' on ' + provider["api_base"] if provider.get("api_base") else '')

                # Check if repeated summary is available
                if hasattr(item, "_repeated_summary"):
                    passed, total = item._repeated_summary
                    TEST_RESULTS[test_id][provider_name] = {"passed": passed, "total": total}
                    print(f"\n[DEBUG] Captured: {test_id} / {provider_name} = {passed}/{total}")


def pytest_sessionfinish(session, exitstatus):
    """Generate markdown table at the end of test session."""
    print(f"\n[DEBUG] Session finishing. TEST_RESULTS has {len(TEST_RESULTS)} test cases")

    # Create results directory and timestamped filename
    project_path = os.getenv('PROJECT_PATH', '/nutrition_information_extraction')
    results_dir = Path(project_path) / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.now()
    datestamp = now.strftime("%Y%m%d-%H%M%S")
    output_path = results_dir / f"{datestamp}.md"

    # Get all unique providers and test cases
    providers = sorted(set(provider for test_results in TEST_RESULTS.values()
                          for provider in test_results.keys()))
    test_ids = sorted(TEST_RESULTS.keys())

    print(f"[DEBUG] Providers: {providers}")
    print(f"[DEBUG] Test IDs: {test_ids}")

    # Build markdown table
    lines = []
    lines.append(f"# Test Results ({now.strftime('%Y-%m-%d %H:%M:%S')})\n")

    # Header row
    header = "| Test Case | " + " | ".join(providers) + " |"
    lines.append(header)

    # Separator row
    separator = "|" + "---|" * (len(providers) + 1)
    lines.append(separator)

    # Data rows
    for test_id in test_ids:
        row = f"| {test_id} |"
        for provider in providers:
            result = TEST_RESULTS[test_id].get(provider, {})
            passed = result.get('passed', 0)
            total = result.get('total', 0)
            cell = f" {passed}/{total}" if total > 0 else " - "
            row += cell + " |"
        lines.append(row)

    # Write to file
    if test_ids:
        output_path.write_text("\n".join(lines) + "\n")
        print(f"\n✓ Results written to {output_path}")
    else:
        print(f"\n. Nothing to write.")
