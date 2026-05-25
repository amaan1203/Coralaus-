#!/usr/bin/env python3
"""
Test Script: Component 3 — Repo Health Check
Tests GitHub health scoring independently (Coral or API fallback).

Usage:
    python scripts/test_repo_health.py [github-url]
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def test_repo_health(repo_url: str = None):
    from agents.repo_health import check_repo_health, compute_health_score
    from agents.coral_utils import get_coral_client

    print("\n" + "=" * 60)
    print("TEST: Component 3 — Repo Health Check")
    print("=" * 60)

    coral = get_coral_client()
    print(f"Coral available: {coral.available}")

    # Test repos with known characteristics
    test_repos = [
        ("https://github.com/huggingface/transformers", "Active, well-maintained"),
        ("https://github.com/tensorflow/tensor2tensor", "Archived/stale"),
    ]

    if repo_url:
        test_repos.insert(0, (repo_url, "User-provided"))

    for url, description in test_repos:
        print(f"\n--- Testing: {url} ({description}) ---")
        result = check_repo_health(url)

        score = result.get("health_score", -1)
        emoji = result.get("health_emoji", "⚪")
        label = result.get("health_label", "Unknown")

        print(f"  Score: {emoji} {score}/100 ({label})")

        signals = result.get("signals", {})
        print(f"  Last commit: {signals.get('last_commit_days_ago', '?')} days ago")
        print(f"  Open issues: {signals.get('open_issues', '?')}")
        print(f"  Stars: {signals.get('stars', '?')}")
        print(f"  Forks: {signals.get('forks', '?')}")
        print(f"  Archived: {signals.get('archived', '?')}")
        print(f"  Coral queries used: {result.get('coral_queries_used', 0)}")
        if result.get("fallback"):
            print(f"  ⚠️  Used GitHub API fallback (Coral not available)")

    # Test health scoring function directly
    print("\n--- Unit test: compute_health_score ---")
    assert compute_health_score(30, 10, 1000, False) == 100, "Recent active repo should score 100"
    assert compute_health_score(400, 10, 100, False) == 75, "1yr stale repo should score 75"
    assert compute_health_score(800, 110, 5, False) == 20, "Old repo with issues should score low"
    assert compute_health_score(100, 10, 100, True) == 0, "Archived repo should score 0"
    print("  ✅ All scoring unit tests passed")

    print(f"\n📊 Total Coral queries: {coral.get_query_count()}")
    print("\n✅ ALL TESTS COMPLETED")
    return True


if __name__ == "__main__":
    url = sys.argv[1] if len(sys.argv) > 1 else None
    success = test_repo_health(url)
    sys.exit(0 if success else 1)
