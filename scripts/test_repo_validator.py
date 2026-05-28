#!/usr/bin/env python3
"""
Test Script: Component 2.5 — Repository Validator
Tests repository validation, embedding semantic checks, fuzzy keyword matching,
dependency checks, and multi-candidate ranking.

Usage:
    python scripts/test_repo_validator.py
"""

import sys
import os
import json
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def test_repo_validator():
    from agents.repo_validator import validate_repo, validate_and_rank_candidates
    from agents.coral_utils import get_coral_client

    print("\n" + "=" * 60)
    print("TEST: Component 2.5 — Repository Validator")
    print("=" * 60)

    coral = get_coral_client()
    print(f"Coral available: {coral.available}")

    # 1. Mock Paper Specifications
    paper_json = {
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks in an encoder-decoder configuration. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable.",
        "arxiv_id": "1706.03762",
        "keywords": ["Transformer", "attention", "encoder-decoder", "self-attention", "machine translation"],
    }

    # 2. Mock Candidates representing different match levels
    # Candidate A: Official TF implementation (High Match)
    # Candidate B: Stale or unrelated (Low Match / Mismatch)
    # Candidate C: An unrelated popular PyTorch repository (Mismatch)
    candidates = [
        {
            "repo_url": "https://github.com/tensorflow/tensor2tensor",
            "stars": 12000,
            "is_official": True,
            "search_method": "arxiv_id"
        },
        {
            "repo_url": "https://github.com/huggingface/transformers",
            "stars": 120000,
            "is_official": False,
            "search_method": "title"
        },
        {
            "repo_url": "https://github.com/django/django",
            "stars": 75000,
            "is_official": False,
            "search_method": "keywords"
        }
    ]

    print("\n--- Running Multi-Candidate Validation & Ranking ---")
    best, ranked_results = validate_and_rank_candidates(paper_json, candidates)

    print("\nRanking Results:")
    for idx, r in enumerate(ranked_results):
        print(f"\n Rank #{idx+1}: {r['repo_url']}")
        print(f"  - Score: {r['confidence_score']:.2%} ({r['classification']})")
        print(f"  - Validator breakdown:")
        print(f"    * Semantic Similarity: {r['validator_scores']['semantic']:.2f}")
        print(f"    * Concept Matching:    {r['validator_scores']['concept']:.2f}")
        print(f"    * Dependency Matching: {r['validator_scores']['dependency']:.2f}")
        if r.get("error"):
            print(f"  - Error: {r['error']}")

    print("\n" + "-" * 50)
    print(f"   Selected Primary Repo: {best['repo_url']}")
    print(f"   Confidence Score: {best['confidence_score']:.2%} ({best['classification']})")
    print("-" * 50)

    # Simple sanity checks - avoid strict score thresholds which fail if GitHub API rate-limits us
    assert len(ranked_results) == len(candidates), "Should return validation results for all candidates"
    assert ranked_results[-1]["repo_url"] == "https://github.com/django/django", "Django should be ranked last (mismatch)!"

    print("\nALL VALIDATOR TEST CASES PASSED SUCCESSFULLY")
    return True


if __name__ == "__main__":
    success = test_repo_validator()
    sys.exit(0 if success else 1)
