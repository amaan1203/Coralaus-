#!/usr/bin/env python3
"""
Test Script: Component 2 — PapersWithCode Search
Tests paper search and implementation discovery independently.

Usage:
    python scripts/test_pwc_search.py
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def test_pwc_search():
    from agents.pwc_search import search_implementation
    from agents.pwc_mcp_client import get_s2_client

    print("\n" + "=" * 60)
    print("TEST: Component 2 — PapersWithCode Search")
    print("=" * 60)

    pwc = get_s2_client()
    if not pwc.available:
        print("  PapersWithCode API not available — check your network")
        return False

    # Test 1: Search a well-known paper by arxiv_id
    print("\n--- Test 1: Search by arXiv ID ---")
    paper1 = {
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
    }
    result1 = search_implementation(paper1)
    print(f"  Found: {result1['found']}")
    if result1['found']:
        print(f"  ✅ Repo: {result1['repo_url']}")
        print(f"  ✅ Stars: {result1.get('stars', 'N/A')}")
        print(f"  ✅ Official: {result1.get('is_official', 'N/A')}")
        print(f"  ✅ Method: {result1.get('search_method', 'N/A')}")
    else:
        print("   No implementation found (API may be rate limited)")

    # Test 2: Search by title
    print("\n--- Test 2: Search by title ---")
    paper2 = {
        "title": "BERT: Pre-training of Deep Bidirectional Transformers",
        "arxiv_id": "",
    }
    result2 = search_implementation(paper2)
    print(f"  Found: {result2['found']}")
    if result2['found']:
        print(f"  ✅ Repo: {result2['repo_url']}")
        print(f"  ✅ Method: {result2.get('search_method', 'N/A')}")

    # Test 3: Search a likely non-existent paper
    print("\n--- Test 3: Non-existent paper ---")
    paper3 = {
        "title": "A Completely Fake Paper Title That Does Not Exist XYZ123",
        "arxiv_id": "",
    }
    result3 = search_implementation(paper3)
    print(f"  Found: {result3['found']}")
    if not result3['found']:
        print("  ✅ Correctly returned no results")
    else:
        print(f"  ⚠️  Unexpected result: {result3['repo_url']}")

    # Stats
    print(f"\n📊 Total PWC API calls: {pwc.get_call_count()}")

    print("\n✅ ALL TESTS COMPLETED")
    return True


if __name__ == "__main__":
    success = test_pwc_search()
    sys.exit(0 if success else 1)
