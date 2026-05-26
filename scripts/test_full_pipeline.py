#!/usr/bin/env python3
"""
Test Script: Full Pipeline End-to-End
Runs the complete coralus pipeline on a sample paper.

Usage:
    python scripts/test_full_pipeline.py [path-to-pdf]

If no PDF is provided, downloads 'Attention Is All You Need' from arXiv.
"""

import sys
import os
import json
import logging
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv(override=True)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def run_full_pipeline(pdf_path: str = None):
    from agents.ingest import parse_paper
    from agents.pwc_search import search_implementation
    from agents.repo_health import check_repo_health
    from agents.compat_check import check_compatibility
    from agents.conflict_resolver import resolve_conflicts
    from agents.no_impl_generator import generate_from_scratch
    from agents.output_builder import build_output, save_output, format_summary
    from agents.coral_utils import get_coral_client
    from agents.pwc_mcp_client import get_s2_client

    print("\n" + "=" * 70)
    print("  coralus — Full Pipeline Test")
    print("=" * 70)

    start_time = time.time()
    coral = get_coral_client()
    pwc = get_s2_client()
    total_llm_calls = {"gemini_flash": 0, "groq_llama": 0}

    # --- Download sample if needed ---
    if not pdf_path:
        import requests
        pdf_path = os.path.join("sample_papers", "attention_is_all_you_need.pdf")
        os.makedirs("sample_papers", exist_ok=True)
        if not os.path.exists(pdf_path):
            print("\n Downloading sample paper...")
            resp = requests.get("https://arxiv.org/pdf/1706.03762v7", timeout=30)
            if resp.status_code == 200:
                with open(pdf_path, "wb") as f:
                    f.write(resp.content)
                print(f"  Downloaded to {pdf_path}")
            else:
                print(f" Download failed: {resp.status_code}")
                return False

    # --- Component 1: PDF Ingestion ---
    print("\n Component 1: PDF Ingestion")
    print("-" * 40)
    try:
        paper_json = parse_paper(pdf_path, os.path.join("output", "current_paper.json"))
        print(f"   Title: {paper_json.get('title', 'Unknown')}")
        print(f"   Sections: {len(paper_json.get('sections', []))}")
        print(f"   Parsed by: {paper_json.get('parsed_by', 'unknown')} (NO LLM)")
    except Exception as e:
        print(f"   Failed: {e}")
        return False

    # --- Component 2: PapersWithCode Search ---
    print("\n Component 2: PapersWithCode Search")
    print("-" * 40)
    search_result = search_implementation(paper_json)
    print(f"  Implementation found: {search_result['found']}")
    if search_result['found']:
        print(f"   Repo: {search_result['repo_url']}")
        print(f"   Method: {search_result.get('search_method')}")

    health_result = None
    compat_result = None
    resolver_result = None
    generator_result = None

    if search_result['found']:
        # --- Component 3: Repo Health ---
        print("\n Component 3: Repo Health Check")
        print("-" * 40)
        health_result = check_repo_health(search_result['repo_url'])
        score = health_result.get('health_score', -1)
        print(f"   Health: {health_result['health_emoji']} {score}/100 ({health_result['health_label']})")
        print(f"  Coral queries: {health_result.get('coral_queries_used', 0)}")

        # --- Component 4: Dependency Check ---
        print("\n Component 4: Dependency Compatibility")
        print("-" * 40)
        compat_result = check_compatibility(
            search_result['repo_url'],
            health_result.get('owner'),
            health_result.get('repo')
        )
        print(f"   Clean: {compat_result.get('clean')}")
        print(f"  Conflicts: {len(compat_result.get('conflicts', []))}")

        # --- Component 5: Conflict Resolution ---
        if not compat_result.get('clean'):
            print("\n Component 5: Conflict Resolution")
            print("-" * 40)
            req_content = ""
            for name, content in compat_result.get('dep_files', {}).items():
                if 'requirement' in name.lower():
                    req_content = content
                    break
            resolver_result = resolve_conflicts(
                req_content, compat_result.get('conflicts', []), paper_json.get('title', ''),
                entrypoint=compat_result.get('entrypoint'),
            )
            total_llm_calls['groq_llama'] += 1
            print(f"  Dockerfile generated: {bool(resolver_result.get('dockerfile'))}")
        else:
            print("\n Component 5: Skipped (no conflicts)")
    else:
        # --- Component 6: Generate from Scratch ---
        print("\n Component 6: Generate from Scratch")
        print("-" * 40)
        generator_result = generate_from_scratch(paper_json)
        total_llm_calls.update(generator_result.get('llm_calls', {}))
        print(f"   Keywords: {generator_result.get('keyterms', [])[:5]}")
        print(f"   Implementation: {len(generator_result.get('implementation_py', ''))} chars")

    # --- Component 7: Output Assembly ---
    print("\n Component 7: Output Assembly")
    print("-" * 40)
    final_output = build_output(
        paper_json=paper_json,
        search_result=search_result,
        health_result=health_result,
        compat_result=compat_result,
        resolver_result=resolver_result,
        generator_result=generator_result,
        coral_queries_total=coral.get_query_count(),
        llm_calls=total_llm_calls,
    )
    saved = save_output(final_output)
    print(f"   Saved {len(saved)} artifacts")
    for name, path in saved.items():
        print(f"    📁 {name}: {path}")

    # --- Summary ---
    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)
    print(format_summary(final_output))
    print(f"\n  Total time: {elapsed:.1f}s")
    print(f" Coral queries: {coral.get_query_count()}")
    print(f" PWC API calls: {pwc.get_call_count()}")

    print("\n FULL PIPELINE TEST COMPLETED SUCCESSFULLY")
    return True


if __name__ == "__main__":
    pdf_path = sys.argv[1] if len(sys.argv) > 1 else None
    success = run_full_pipeline(pdf_path)
    sys.exit(0 if success else 1)
