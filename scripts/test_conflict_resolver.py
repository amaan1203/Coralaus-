#!/usr/bin/env python3
"""
Test Script: Component 5 — Conflict Resolution & Dockerfile Generation
Tests Groq-based conflict resolution independently.

Usage:
    python scripts/test_conflict_resolver.py
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def test_conflict_resolver():
    from agents.conflict_resolver import resolve_conflicts

    print("\n" + "=" * 60)
    print("TEST: Component 5 — Conflict Resolution & Dockerfile Generation")
    print("=" * 60)

    # Test 1: With conflicts
    print("\n--- Test 1: Requirements with conflicts ---")
    test_reqs = """torch==1.9.0
torchvision==0.14.0
numpy>=1.20,<1.24
scipy==1.7.0
transformers==4.30.0
tokenizers==0.13.0
"""
    test_conflicts = [
        "torch 1.9.0 requires torchvision 0.10.x but torchvision 0.14.0 requires torch>=1.13",
        "numpy>=1.20,<1.24 conflicts with scipy 1.7.0 which needs numpy<1.23",
    ]

    result = resolve_conflicts(test_reqs, test_conflicts, "Test ML Paper")

    print(f"  Dockerfile generated: {bool(result.get('dockerfile'))}")
    print(f"  Dockerfile length: {len(result.get('dockerfile', ''))} chars")
    print(f"  LLM model: {result.get('llm_model', 'None (fallback)')}")
    print(f"  Resolution notes: {len(result.get('resolution_notes', []))}")

    if result.get("dockerfile"):
        print(f"\n  --- Dockerfile preview (first 500 chars) ---")
        print(f"  {result['dockerfile'][:500]}")

    for note in result.get("resolution_notes", [])[:3]:
        print(f"  📝 {note}")

    # Test 2: Without conflicts (should still generate Dockerfile)
    print("\n--- Test 2: Clean requirements ---")
    clean_reqs = "requests>=2.28\nnumpy>=1.24\n"
    result2 = resolve_conflicts(clean_reqs, [], "Clean Paper")

    print(f"  Dockerfile generated: {bool(result2.get('dockerfile'))}")
    assert result2.get("dockerfile"), " Should generate Dockerfile even without conflicts"
    print("   Dockerfile generated for clean requirements")

    print("\n ALL TESTS COMPLETED")
    return True


if __name__ == "__main__":
    success = test_conflict_resolver()
    sys.exit(0 if success else 1)
