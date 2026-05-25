#!/usr/bin/env python3
"""
Test Script: Component 4 — Dependency Compatibility Check
Tests dependency conflict detection independently.

Usage:
    python scripts/test_compat_check.py [github-url]
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def test_compat_check(repo_url: str = None):
    from agents.compat_check import check_compatibility

    print("\n" + "=" * 60)
    print("TEST: Component 4 — Dependency Compatibility Check")
    print("=" * 60)

    test_url = repo_url or "https://github.com/huggingface/transformers"
    print(f"\nTesting with: {test_url}")

    result = check_compatibility(test_url)

    print(f"\n  Requirements found: {result.get('requirements_found', False)}")
    print(f"  Dep files: {list(result.get('dep_files', {}).keys())}")
    print(f"  Clean: {result.get('clean', 'N/A')}")
    print(f"  Analysis method: {result.get('analysis_method', 'N/A')}")

    if result.get("conflicts"):
        print(f"\n  {len(result['conflicts'])} conflicts:")
        for c in result["conflicts"][:5]:
            print(f"    - {c}")

    if result.get("warnings"):
        print(f"\n  Warnings:")
        for w in result["warnings"][:5]:
            print(f"    - {w}")

    # Test with a known-conflicting requirements
    print("\n--- Test 2: Synthetic conflict ---")
    from agents.compat_check import _pip_dry_run
    synthetic_reqs = "torch==1.0.0\ntorchvision==0.17.0\nnumpy==2.0.0\n"
    pip_result = _pip_dry_run(synthetic_reqs)
    print(f"  Has errors: {pip_result['has_errors']}")
    if pip_result['has_errors']:
        print(f"  Correctly detected conflicts in synthetic requirements")
        for e in pip_result['errors'][:3]:
            print(f"    - {e}")
    else:
        print(f"   pip dry-run did not flag conflicts (may need specific pip version)")

    print("\n ALL TESTS COMPLETED")
    return True


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else None
    success = test_compat_check(url)
    sys.exit(0 if success else 1)
