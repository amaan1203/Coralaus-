#!/usr/bin/env python3
"""
Test Script: Component 6 — No-Implementation Generator
Tests from-scratch implementation generation independently.

Usage:
    python scripts/test_no_impl_generator.py
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


def test_no_impl_generator():
    from agents.no_impl_generator import generate_from_scratch, _extract_keyterms, _basic_keyterm_extraction

    print("\n" + "=" * 60)
    print("TEST: Component 6 — No-Implementation Generator")
    print("=" * 60)

    # Test 1: Basic keyterm extraction (no LLM)
    print("\n--- Test 1: Basic keyterm extraction (no LLM) ---")
    abstract = """We propose a new simple network architecture, the Transformer,
    based solely on attention mechanisms, dispensing with recurrence and convolutions
    entirely. Experiments on two machine translation tasks show these models to be
    superior in quality while being more parallelizable and requiring significantly
    less time to train."""

    keywords = _basic_keyterm_extraction(abstract)
    print(f"  Keywords: {keywords}")
    assert len(keywords) > 0, " Should extract at least some keywords"
    print(f"   Extracted {len(keywords)} keywords")

    # Test 2: LLM keyterm extraction (if Groq available)
    print("\n--- Test 2: LLM keyterm extraction ---")
    if os.environ.get("GROQ_API_KEY"):
        keywords_llm = _extract_keyterms(abstract)
        print(f"  Keywords (Groq): {keywords_llm}")
        print(f"  Groq returned {len(keywords_llm)} keywords")
    else:
        print("   GROQ_API_KEY not set — skipping LLM keyterm test")

    # Test 3: Full generation (may take a minute)
    print("\n--- Test 3: Full from-scratch generation ---")
    # Try to load current_paper.json if available
    test_paper = None
    paths_to_check = ["output/current_paper.json", "../output/current_paper.json"]
    for p in paths_to_check:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    test_paper = json.load(f)
                print(f"  Loaded paper for test: '{test_paper.get('title', 'Unknown')}'")
                break
            except Exception as e:
                print(f"   Failed to load {p}: {e}")

    if not test_paper:
        print("  Using default hardcoded paper: Attention Is All You Need")
        test_paper = {
            "title": "Attention Is All You Need",
            "abstract": abstract,
            "sections": [
                {"heading": "Introduction", "text": "Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation."},
                {"heading": "Model Architecture", "text": "The Transformer follows an encoder-decoder structure using stacked self-attention and point-wise, fully connected layers for both the encoder and decoder."},
            ],
            "keywords": ["transformer", "attention", "self-attention", "machine translation"],
        }

    result = generate_from_scratch(test_paper)

    print(f"  Keyterms: {result.get('keyterms', [])}")
    print(f"  Related repos: {result.get('related_repos', [])}")
    print(f"  Implementation: {len(result.get('implementation_py', ''))} chars")
    print(f"  Dockerfile: {len(result.get('dockerfile', ''))} chars")
    print(f"  Coral queries: {result.get('coral_queries_used', 0)}")
    print(f"  LLM calls: {result.get('llm_calls', {})}")

    assert result.get("implementation_py"), " Should generate some implementation"
    assert result.get("dockerfile"), " Should generate a Dockerfile"
    print("   Implementation and Dockerfile generated")

    # Preview
    print(f"\n  --- Implementation preview (first 300 chars) ---")
    print(f"  {result['implementation_py'][:300]}")

    print("\n ALL TESTS COMPLETED")
    return True


if __name__ == "__main__":
    success = test_no_impl_generator()
    sys.exit(0 if success else 1)
