#!/usr/bin/env python
"""Utility: Given a paper PDF path, run ingestion -> implementation search -> validation.

Usage:
    python scripts/get_best_repo_for_paper.py /path/to/paper.pdf

Prints the best repository JSON to stdout and exits with code 0 on success.
"""
import argparse
import json
import logging
import sys
import os
import csv

# Ensure project root is on sys.path so `agents.*` imports work when executed from scripts/
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(override=True)

from agents.ingest import parse_paper
from agents.pwc_search import search_implementation
from agents.repo_validator import validate_and_rank_candidates


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_pipeline(paper_path: str) -> dict:
    # 1. Parse the paper into structured JSON
    logger.info(f"Parsing paper: {paper_path}")
    paper_json = parse_paper(paper_path)

    # 2. Search for candidate implementations
    logger.info("Searching for implementations via PapersWithCode / Semantic Scholar / GitHub...")
    search_res = search_implementation(paper_json)
    candidates = search_res.get("candidates", [])

    if not candidates:
        logger.info("No implementation candidates found.")
        return {"found": False, "reason": "no_candidates", "paper": paper_json}

    # 3. Validate and rank candidates
    logger.info(f"Validating {len(candidates)} candidates...")
    best, all_results = validate_and_rank_candidates(paper_json, candidates, include_pruned=True)

    return {"found": True, "best": best, "all": all_results, "paper_meta": {"title": paper_json.get("title"), "arxiv_id": paper_json.get("arxiv_id")}}


def main(argv=None):
    p = argparse.ArgumentParser(description="Find best implementation repository for a paper PDF")
    p.add_argument("paper", help="Path to the paper PDF file")
    p.add_argument("--out", help="Optional output JSON path to save results")
    args = p.parse_args(argv)

    try:
        res = run_pipeline(args.paper)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                json.dump(res, f, indent=2, ensure_ascii=False)
        # Also write a CSV with per-candidate details next to the JSON output (or alongside script if no --out)
        csv_path = None
        if args.out:
            csv_path = os.path.splitext(args.out)[0] + ".csv"
        else:
            base = os.path.splitext(os.path.basename(args.paper))[0]
            csv_path = os.path.join(_PROJECT_ROOT, f"{base}_candidates.csv")

        all_candidates = res.get("all", []) if res.get("found") else []
        if all_candidates:
            fieldnames = [
                "repo_url", "search_method", "stars", "is_official", "pre_boost_score",
                "confidence_score", "classification", "pruned", "prune_reason", "semantic", "concept", "dependency", "codebase",
                "boost_weak_officiality", "boost_acronym", "boost_stars", "boost_implementation_of_sentence", "boost_official_override", "boost_applied_total"
            ]
            try:
                with open(csv_path, "w", newline='', encoding="utf-8") as cf:
                    writer = csv.DictWriter(cf, fieldnames=fieldnames)
                    writer.writeheader()
                    for c in all_candidates:
                        vd = c.get("validator_scores", {})
                        boosts = c.get("boosts", {}) or {}
                        row = {
                            "repo_url": c.get("repo_url"),
                            "search_method": c.get("search_method"),
                            "stars": c.get("stars"),
                            "is_official": c.get("is_official"),
                            "pre_boost_score": c.get("pre_boost_score"),
                            "confidence_score": c.get("confidence_score"),
                            "classification": c.get("classification"),
                            "pruned": c.get("pruned", False),
                            "prune_reason": c.get("prune_reason", ""),
                            "semantic": vd.get("semantic"),
                            "concept": vd.get("concept"),
                            "dependency": vd.get("dependency"),
                            "codebase": vd.get("codebase"),
                            "boost_weak_officiality": boosts.get("weak_officiality"),
                            "boost_acronym": boosts.get("acronym"),
                            "boost_stars": boosts.get("stars"),
                            "boost_implementation_of_sentence": boosts.get("implementation_of_sentence", 0.0),
                            "boost_official_override": boosts.get("official_override"),
                            "boost_applied_total": boosts.get("applied_total"),
                        }
                        writer.writerow(row)
                logger.info(f"Candidate CSV written to: {csv_path}")
            except Exception:
                logger.exception(f"Failed to write CSV to {csv_path}")
        # Print best repo as compact JSON to stdout
        print(json.dumps(res.get("best") if res.get("found") else res, ensure_ascii=False, indent=2))
        return 0
    except Exception as e:
        logger.exception("Pipeline failed")
        print(json.dumps({"error": str(e)}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
