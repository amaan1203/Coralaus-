#!/usr/bin/env python3
"""
Test Script: Component 1 — PDF Ingestion
Tests GROBID-based PDF parsing independently.

Usage:
    python scripts/test_ingest.py [path-to-pdf]

If no PDF is provided, downloads a sample paper from arXiv.
"""

import sys
import os
import json
import logging
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def download_sample_paper() -> str:
    """Download 'Attention Is All You Need' from arXiv as a test PDF."""
    import requests

    url = "https://arxiv.org/pdf/1706.03762v7"
    logger.info(f"Downloading sample paper from {url}...")

    pdf_path = os.path.join("sample_papers", "attention_is_all_you_need.pdf")
    os.makedirs("sample_papers", exist_ok=True)

    if os.path.exists(pdf_path):
        logger.info(f"Sample paper already exists at {pdf_path}")
        return pdf_path

    resp = requests.get(url, timeout=30)
    if resp.status_code == 200:
        with open(pdf_path, "wb") as f:
            f.write(resp.content)
        logger.info(f"Downloaded to {pdf_path} ({len(resp.content) / 1024:.1f} KB)")
        return pdf_path
    else:
        logger.error(f"Download failed: {resp.status_code}")
        return None


def test_ingest(pdf_path: str):
    """Test PDF parsing."""
    from agents.ingest import parse_paper

    print("\n" + "=" * 60)
    print("TEST: Component 1 — PDF Ingestion")
    print("=" * 60)

    output_path = os.path.join("output", "test_current_paper.json")

    try:
        paper = parse_paper(pdf_path, output_path)

        # Validate output
        assert paper.get("title"), "❌ FAIL: title is empty"
        print(f"✅ Title: {paper['title']}")

        assert paper.get("abstract"), "❌ FAIL: abstract is empty"
        print(f"✅ Abstract: {paper['abstract'][:100]}...")

        assert paper.get("authors"), "❌ FAIL: no authors found"
        print(f"✅ Authors: {len(paper['authors'])} found")
        for a in paper["authors"][:3]:
            print(f"   - {a.get('first', '')} {a.get('last', '')}")

        print(f"✅ Year: {paper.get('year', 'N/A')}")
        print(f"✅ DOI: {paper.get('doi', 'N/A')}")
        print(f"✅ arXiv ID: {paper.get('arxiv_id', 'N/A')}")
        print(f"✅ Sections: {len(paper.get('sections', []))}")
        print(f"✅ References: {len(paper.get('references', []))}")
        print(f"✅ Figures: {len(paper.get('figures', []))}")
        print(f"✅ Keywords: {paper.get('keywords', [])}")
        print(f"✅ Parsed by: {paper.get('parsed_by', 'unknown')}")
        print(f"✅ Output saved to: {output_path}")

        print("\n✅ ALL CHECKS PASSED")
        return True

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    else:
        pdf_path = download_sample_paper()
        if not pdf_path:
            print("❌ Could not get sample paper. Provide a PDF path as argument.")
            sys.exit(1)

    success = test_ingest(pdf_path)
    sys.exit(0 if success else 1)
