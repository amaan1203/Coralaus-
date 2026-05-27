"""
Batch Dockerfile Validation Script

Runs the full Coralaus pipeline on every PDF in sample_papers/, generates
Dockerfiles, validates each one, then attempts a real `docker build` on the
best candidate.

Usage:
    python scripts/validate_all_sample_papers.py [--build] [--paper FILENAME]

Flags:
    --build           Also run a real docker build on the best Dockerfile
    --paper FILENAME  Run only on a specific sample paper (e.g. LORA.pdf)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

# Ensure project root is on sys.path
PROJECT_ROOT = str(Path(__file__).parent.parent)
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(override=True)

from agents.ingest import parse_paper
from agents.pwc_search import search_implementation
from agents.repo_health import check_repo_health
from agents.compat_check import check_compatibility
from agents.conflict_resolver import resolve_conflicts
from agents.no_impl_generator import generate_from_scratch
from agents.dockerfile_validator import (
    validate_dockerfile,
    build_dockerfile_test,
    format_validation_summary,
)
from agents.repo_validator import validate_repo_relevance, format_relevance_badge

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

SAMPLE_PAPERS_DIR = os.path.join(PROJECT_ROOT, "sample_papers")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "output", "batch_validation")


def run_pipeline_for_paper(pdf_path: str) -> dict:
    """
    Run the full pipeline for a single paper and return the validation report.
    """
    paper_name = os.path.basename(pdf_path)
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing: {paper_name}")
    logger.info(f"{'='*60}")

    report = {
        "paper_file": paper_name,
        "paper_title": "",
        "pipeline_success": False,
        "implementation_found": False,
        "repo_url": None,
        "repo_relevance": None,
        "dockerfile_generated": False,
        "dockerfile_content": "",
        "requirements_content": "",
        "dockerfile_validation": None,
        "errors": [],
        "timing_s": 0,
    }

    start = time.monotonic()

    try:
        # --- Component 1: Parse PDF ---
        logger.info(f"[1/5] Parsing PDF: {paper_name}")
        output_json_path = os.path.join(OUTPUT_DIR, f"{paper_name}.paper.json")
        paper_json = parse_paper(pdf_path, output_json_path)
        report["paper_title"] = paper_json.get("title", "Unknown")
        logger.info(f"  Title: {report['paper_title']}")

        # --- Component 2: Search Implementation ---
        logger.info(f"[2/5] Searching for implementation...")
        search_result = search_implementation(paper_json)
        report["implementation_found"] = search_result.get("found", False)
        report["repo_url"] = search_result.get("repo_url")

        # --- Component 2b: Repo Relevance Check ---
        if report["implementation_found"] and report["repo_url"]:
            logger.info(f"[2b] Validating repo relevance: {report['repo_url']}")
            relevance = validate_repo_relevance(report["repo_url"], paper_json)
            report["repo_relevance"] = relevance
            logger.info(f"  {format_relevance_badge(relevance)}")

        dockerfile = ""
        requirements = ""

        if report["implementation_found"] and report["repo_url"]:
            # --- Component 3+4: Health & Compat ---
            logger.info(f"[3/5] Checking repo health: {report['repo_url']}")
            health_result = check_repo_health(report["repo_url"])

            logger.info(f"[4/5] Checking dependencies...")
            compat_result = check_compatibility(
                report["repo_url"],
                health_result.get("owner"),
                health_result.get("repo"),
            )

            # --- Component 5: Resolve & Generate Dockerfile ---
            req_content = ""
            for name, content in compat_result.get("dep_files", {}).items():
                if "requirement" in name.lower():
                    req_content = content
                    break

            logger.info(f"[5/5] Generating Dockerfile...")
            if req_content or not compat_result.get("clean", True):
                resolver_result = resolve_conflicts(
                    req_content,
                    compat_result.get("conflicts", []),
                    paper_json.get("title", ""),
                    entrypoint=compat_result.get("entrypoint"),
                )
                dockerfile = resolver_result.get("dockerfile", "")
                requirements = resolver_result.get("resolved_requirements", req_content)
            elif req_content:
                # Clean repo, still generate Dockerfile
                resolver_result = resolve_conflicts(
                    req_content, [], paper_json.get("title", ""),
                    entrypoint=compat_result.get("entrypoint"),
                )
                dockerfile = resolver_result.get("dockerfile", "")
                requirements = resolver_result.get("resolved_requirements", req_content)

        else:
            # No implementation found → generate from scratch
            logger.info(f"[3-5] Generating from scratch (no implementation found)...")
            gen_result = generate_from_scratch(paper_json)
            dockerfile = gen_result.get("dockerfile", "")
            requirements = ""

        report["dockerfile_content"] = dockerfile
        report["requirements_content"] = requirements
        report["dockerfile_generated"] = bool(dockerfile)

        # --- Validate the Dockerfile ---
        if dockerfile:
            logger.info(f"Validating generated Dockerfile...")
            validation = validate_dockerfile(dockerfile, requirements)
            report["dockerfile_validation"] = validation
            logger.info(f"  {format_validation_summary(validation)}")

            # Save Dockerfile to disk
            df_path = os.path.join(OUTPUT_DIR, f"{paper_name}.Dockerfile")
            with open(df_path, "w", encoding="utf-8") as f:
                f.write(dockerfile)
        else:
            logger.warning(f"  No Dockerfile was generated for {paper_name}")

        report["pipeline_success"] = True

    except Exception as e:
        report["errors"].append(str(e))
        logger.error(f"Pipeline failed for {paper_name}: {e}", exc_info=True)

    report["timing_s"] = round(time.monotonic() - start, 1)
    return report


def run_batch(pdf_filter: str = None, do_build: bool = False):
    """
    Run the pipeline for all sample papers and generate a batch report.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect PDFs
    pdf_files = sorted([
        f for f in os.listdir(SAMPLE_PAPERS_DIR)
        if f.lower().endswith(".pdf")
    ])

    if pdf_filter:
        pdf_files = [f for f in pdf_files if pdf_filter.lower() in f.lower()]
        if not pdf_files:
            logger.error(f"No sample paper matching '{pdf_filter}' found")
            return

    logger.info(f"Found {len(pdf_files)} sample papers to process")

    reports = []
    for pdf_file in pdf_files:
        pdf_path = os.path.join(SAMPLE_PAPERS_DIR, pdf_file)
        report = run_pipeline_for_paper(pdf_path)
        reports.append(report)

    # --- Summary ---
    print("\n" + "=" * 70)
    print("BATCH VALIDATION SUMMARY")
    print("=" * 70)

    total = len(reports)
    successful = sum(1 for r in reports if r["pipeline_success"])
    with_dockerfile = sum(1 for r in reports if r["dockerfile_generated"])
    valid_dockerfiles = sum(
        1 for r in reports
        if (r.get("dockerfile_validation") or {}).get("valid", False)
    )

    print(f"\nPapers processed:     {total}")
    print(f"Pipeline succeeded:   {successful}/{total}")
    print(f"Dockerfiles generated:{with_dockerfile}/{total}")
    print(f"Valid Dockerfiles:    {valid_dockerfiles}/{with_dockerfile}")

    # Per-paper results
    print(f"\n{'Paper':<35} {'Dockerfile':<15} {'Validation':<20} {'Relevance':<25} {'Time'}")
    print("-" * 110)

    for r in reports:
        name = r["paper_file"][:33]
        df_status = "✅ generated" if r["dockerfile_generated"] else "❌ none"
        val = r.get("dockerfile_validation", {})
        val_status = format_validation_summary(val) if val else "—"
        rel = r.get("repo_relevance", {})
        rel_status = format_relevance_badge(rel) if rel else "—"
        timing = f"{r['timing_s']}s"
        print(f"{name:<35} {df_status:<15} {val_status:<20} {rel_status:<25} {timing}")

    # Relevance check results
    print(f"\n--- Repo Relevance Results ---")
    for r in reports:
        rel = r.get("repo_relevance")
        if rel:
            print(f"  {r['paper_file'][:30]}: {format_relevance_badge(rel)}")
            if rel.get("evidence"):
                for ev in rel["evidence"]:
                    print(f"    ✓ {ev}")
            if rel.get("warnings"):
                for w in rel["warnings"]:
                    print(f"    ⚠ {w}")

    # Dockerfile validation details
    print(f"\n--- Dockerfile Validation Details ---")
    for r in reports:
        val = r.get("dockerfile_validation")
        if val:
            print(f"\n  {r['paper_file']}:")
            print(f"    Base image: {val.get('base_image', '?')}")
            print(f"    Fresh: {val.get('base_image_fresh', '?')}")
            if val.get("issues"):
                for issue in val["issues"]:
                    print(f"    ❌ {issue}")
            if val.get("warnings"):
                for warning in val["warnings"]:
                    print(f"    ⚠  {warning}")

    # --- Docker Build Test ---
    if do_build and with_dockerfile > 0:
        print(f"\n{'='*70}")
        print("DOCKER BUILD TEST")
        print(f"{'='*70}")

        # Pick the best candidate: fewest issues + warnings
        buildable = [
            r for r in reports
            if r["dockerfile_generated"] and r.get("dockerfile_validation")
        ]
        buildable.sort(key=lambda r: (
            len(r["dockerfile_validation"].get("issues", [])),
            len(r["dockerfile_validation"].get("warnings", [])),
        ))

        best = buildable[0]
        print(f"\nBest candidate: {best['paper_file']}")
        print(f"Title: {best['paper_title']}")
        print(f"Validation: {format_validation_summary(best['dockerfile_validation'])}")
        print(f"\nRunning docker build...")

        build_result = build_dockerfile_test(
            best["dockerfile_content"],
            best["requirements_content"],
            tag="coralaus-batch-test",
        )

        if not build_result.get("available"):
            print(f"⚠️  Docker not available: {build_result.get('error', 'unknown')}")
        elif build_result["success"]:
            print(f"✅ Docker build SUCCEEDED in {build_result['duration_s']}s")
        else:
            print(f"❌ Docker build FAILED: {build_result.get('error', 'unknown')}")

        if build_result.get("log"):
            print(f"\n--- Build Log (last 20 lines) ---")
            log_lines = build_result["log"].strip().splitlines()
            for line in log_lines[-20:]:
                print(f"  {line}")

        best["build_result"] = {
            k: v for k, v in build_result.items() if k != "log"
        }

    # Save full report
    report_path = os.path.join(OUTPUT_DIR, "batch_report.json")
    # Strip large content fields for the JSON report
    slim_reports = []
    for r in reports:
        slim = dict(r)
        slim["dockerfile_content"] = slim["dockerfile_content"][:200] + "..." if slim["dockerfile_content"] else ""
        slim["requirements_content"] = slim["requirements_content"][:200] + "..." if slim["requirements_content"] else ""
        slim_reports.append(slim)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total_papers": total,
            "successful": successful,
            "dockerfiles_generated": with_dockerfile,
            "valid_dockerfiles": valid_dockerfiles,
            "reports": slim_reports,
        }, f, indent=2, ensure_ascii=False)

    print(f"\n📄 Full report saved to: {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Dockerfile validation for sample papers")
    parser.add_argument("--build", action="store_true", help="Run docker build test on best candidate")
    parser.add_argument("--paper", type=str, default=None, help="Run only for a specific paper file")
    args = parser.parse_args()

    run_batch(pdf_filter=args.paper, do_build=args.build)
