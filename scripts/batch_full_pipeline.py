#!/usr/bin/env python3
"""
Batch Repository Validation Execution Script
Runs Ingestion (Component 1), Search (Component 2), and validation ranking (Component 2.5)
for all sample papers in `sample_papers/` and outputs a summary CSV.
"""

import os
import sys
import time
import logging

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(override=True)

from agents.ingest import parse_paper
from agents.pwc_search import search_implementation
from agents.repo_validator import validate_and_rank_candidates

# Configure root logger to warning to reduce screen clutter
logging.getLogger().setLevel(logging.WARNING)

def main():
    sample_papers_dir = os.path.join(PROJECT_ROOT, "sample_papers")
    if not os.path.exists(sample_papers_dir):
        print(f"Error: {sample_papers_dir} directory not found.")
        sys.exit(1)
        
    pdf_files = sorted([f for f in os.listdir(sample_papers_dir) if f.lower().endswith(".pdf")])
    if not pdf_files:
        print("No PDF sample papers found.")
        sys.exit(0)
        
    print(f"Found {len(pdf_files)} sample papers to process.")
    results = []
    
    for pdf in pdf_files:
        pdf_path = os.path.join(sample_papers_dir, pdf)
        print(f"\n" + "=" * 80)
        print(f" PROCESSING: {pdf}")
        print("=" * 80)
        
        start = time.monotonic()
        success = False
        repo_url = "No implementation found"
        confidence_score = 0.0
        classification = "MISMATCH"
        paper_title = pdf
        
        try:
            # 1. Ingestion
            print("1. Parsing PDF...")
            paper_json = parse_paper(pdf_path, os.path.join(PROJECT_ROOT, "output", "current_paper.json"))
            paper_title = paper_json.get("title", pdf)
            
            # 2. Search
            print("2. Searching candidates...")
            search_result = search_implementation(paper_json)
            
            # 3. Validate and Rank
            if search_result.get("found") and search_result.get("candidates"):
                print(f"3. Validating {len(search_result['candidates'])} candidates...")
                best_match, ranked_results = validate_and_rank_candidates(paper_json, search_result["candidates"], include_pruned=True)
                repo_url = best_match.get("repo_url", "No implementation found")
                confidence_score = best_match.get("confidence_score", 0.0)
                classification = best_match.get("classification", "MISMATCH")
                
                # Write individual candidates CSV
                pdf_base = os.path.splitext(pdf)[0]
                csv_path = os.path.join(PROJECT_ROOT, f"{pdf_base}_candidates.csv")
                import csv
                fieldnames = [
                    "repo_url", "search_method", "stars", "is_official", "pre_boost_score",
                    "confidence_score", "classification", "pruned", "prune_reason", "semantic", "concept", "dependency", "codebase",
                    "boost_weak_officiality", "boost_acronym", "boost_stars", "boost_implementation_of_sentence", "boost_official_override", "boost_applied_total"
                ]
                try:
                    with open(csv_path, "w", newline='', encoding="utf-8") as cf:
                        writer = csv.DictWriter(cf, fieldnames=fieldnames)
                        writer.writeheader()
                        for c in ranked_results:
                            vd = c.get("validator_scores", {}) or {}
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
                                "boost_weak_officiality": boosts.get("weak_officiality", 0.0),
                                "boost_acronym": boosts.get("acronym", 0.0),
                                "boost_stars": boosts.get("stars", 0.0),
                                "boost_implementation_of_sentence": boosts.get("implementation_of_sentence", 0.0),
                                "boost_official_override": boosts.get("official_override", False),
                                "boost_applied_total": boosts.get("applied_total", 0.0),
                            }
                            writer.writerow(row)
                    print(f"Candidates CSV written to: {csv_path}")
                except Exception as ce:
                    print(f"[Warning] Failed to write candidates CSV for {pdf}: {ce}")
            else:
                print("3. No candidates found to validate.")
                
            success = True
        except Exception as e:
            print(f"[Error] Pipeline failed on {pdf}: {e}")
            
        elapsed = time.monotonic() - start
        
        results.append({
            "pdf_name": pdf,
            "paper_title": paper_title,
            "repo_url": repo_url,
            "confidence_score": f"{confidence_score:.2%}",
            "classification": classification,
            "elapsed_s": round(elapsed, 2),
            "success": success
        })
        
    print("\n" + "=" * 80)
    print("BATCH REPO VALIDATION SUMMARY")
    print("=" * 80)
    print(f"{'Paper File (PDF)':<30} | {'Final GitHub Repo URL / Status':<40} | {'Score':<8} | {'Class':<12}")
    print("-" * 100)
    for r in results:
        print(f"{r['pdf_name'][:30]:<30} | {r['repo_url'][:40]:<40} | {r['confidence_score']:<8} | {r['classification']:<12}")
    print("=" * 80)

    # Save results to a CSV file in the output folder
    import csv
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, "batch_results.csv")
    
    try:
        with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["pdf_name", "paper_title", "repo_url", "confidence_score", "classification", "elapsed_s", "success"])
            writer.writeheader()
            for r in results:
                writer.writerow(r)
        print(f"\n[Summary] Batch validation results successfully saved to CSV: {csv_path}")
    except Exception as e:
        print(f"[Warning] Failed to write results to CSV: {e}")

if __name__ == "__main__":
    main()
