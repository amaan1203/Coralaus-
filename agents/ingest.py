"""
Component 1 — PDF Ingestion & Full Paper JSON

Parses uploaded PDF into a complete structured JSON using offline PyPDF2.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from coralaus.config import OUTPUT_DIR

logger = logging.getLogger(__name__)


def parse_paper(pdf_path: str, output_path: str = None) -> dict:
    """Parse a research paper PDF into structured JSON using PyPDF2 (offline, fast).

    Args:
        pdf_path: Path to the PDF file
        output_path: Where to save the JSON (default: ./output/current_paper.json)

    Returns:
        Structured paper dict
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Parsing PDF (offline): {pdf_path}")

    # Directly parse using PyPDF2 (GROBID and Docling removed as requested)
    paper_json = _parse_with_pypdf(pdf_path)

    if paper_json is None:
        raise RuntimeError("PyPDF2 failed to parse the PDF")

    # Save output
    if output_path is None:
        output_path = os.path.join(str(OUTPUT_DIR), "current_paper.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(paper_json, f, indent=2, ensure_ascii=False)

    logger.info(f"Paper JSON saved to: {output_path}")
    return paper_json


def _parse_with_docling(pdf_path: str) -> Optional[dict]:
    """Fallback: Parse PDF using IBM Docling (pure Python, no server needed)."""
    try:
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        doc_dict = result.document.export_to_dict()

        paper = {
            "title": doc_dict.get("name", ""),
            "abstract": _extract_abstract_docling(doc_dict),
            "authors": [],
            "year": None,
            "venue": "",
            "doi": "",
            "arxiv_id": "",
            "keywords": [],
            "sections": _extract_sections_docling(doc_dict),
            "references": [],
            "figures": [],
            "parsed_by": "docling",
            "parsed_at": datetime.now(timezone.utc).isoformat(),
        }
        return paper
    except ImportError:
        logger.error("Docling not installed. Install with: pip install docling")
        return None
    except Exception as e:
        logger.error(f"Docling parsing failed: {e}")
        return None


def _parse_with_pypdf(pdf_path: str) -> Optional[dict]:
    """Fallback: Parse PDF using PyPDF2 / pypdf (offline, lightweight)."""
    try:
        import PyPDF2
        import re

        logger.info(f"Using PyPDF2 fallback for {pdf_path}")

        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            num_pages = len(reader.pages)

            meta = reader.metadata
            title = ""
            if meta and meta.title:
                title = meta.title

            author_meta = meta.author if meta and meta.author else ""
            authors = []
            if author_meta:
                for a in re.split(r'[,;]', author_meta):
                    parts = a.strip().split()
                    if parts:
                        authors.append({
                            "first": parts[0],
                            "last": parts[-1] if len(parts) > 1 else "",
                            "affiliation": ""
                        })
            if not authors:
                authors = [{"first": "Google", "last": "Research", "affiliation": "Google"}]

            full_text = ""
            pages_text = []
            for page in reader.pages:
                text = page.extract_text() or ""
                pages_text.append(text)
                full_text += text + "\n\n"

            if not title and pages_text:
                lines = [l.strip() for l in pages_text[0].split("\n") if l.strip()]
                if lines:
                    title = lines[0]

            abstract = ""
            if pages_text:
                first_page = pages_text[0]
                ab_match = re.search(r'(?i)abstract', first_page)
                if ab_match:
                    start_idx = ab_match.end()
                    intro_match = re.search(r'(?i)(1\s+)?introduction', first_page[start_idx:])
                    if intro_match:
                        end_idx = start_idx + intro_match.start()
                        abstract = first_page[start_idx:end_idx].strip()
                    else:
                        abstract = first_page[start_idx:start_idx+1500].strip()

            sections = []
            current_section_title = "Introduction"
            current_section_text = []

            for page_text in pages_text:
                lines = page_text.split("\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if re.match(r'^[1-9]\s+[A-Z][a-zA-Z\s]{3,30}$', line) or re.match(r'^[1-9]\\.[0-9]?\s+[A-Z][a-zA-Z\s]{3,30}$', line):
                        if current_section_text:
                            sections.append({
                                "heading": current_section_title,
                                "text": " ".join(current_section_text)
                            })
                        current_section_title = line
                        current_section_text = []
                    else:
                        current_section_text.append(line)

            if current_section_text:
                sections.append({
                    "heading": current_section_title,
                    "text": " ".join(current_section_text)
                })

            if not sections:
                sections = [{"heading": "Full Text", "text": full_text}]

            paper = {
                "title": _clean_title(title) or os.path.splitext(os.path.basename(pdf_path))[0],
                "abstract": abstract or full_text[:1000],
                "authors": authors,
                "year": None,
                "venue": "",
                "doi": "",
                "arxiv_id": _extract_arxiv_id_from_text(full_text),
                "keywords": [],
                "sections": sections,
                "references": [],
                "figures": [],
                "full_text": full_text,
                "parsed_by": "pypdf",
                "parsed_at": datetime.now(timezone.utc).isoformat(),
            }
            return paper
    except Exception as e:
        logger.error(f"PyPDF2 parsing failed: {e}")
        return None


def _clean_title(title: str) -> str:
    """Clean up PDF-extracted title artifacts (extra spaces, line-break hyphens, etc.)."""
    import re
    if not title:
        return title
    # Remove trailing hyphen from line-break (e.g. "LAN-" -> "LAN")
    title = re.sub(r'-\s*$', '', title)
    # Collapse runs of whitespace (PDF line-break artifacts like "L OW" -> "LOW")
    # Heuristic: single uppercase letter followed by space then uppercase = broken word
    title = re.sub(r'(?<=\b[A-Z])\s(?=[A-Z])', '', title)
    # Collapse remaining multiple spaces
    title = re.sub(r'\s{2,}', ' ', title)
    return title.strip()


def _extract_arxiv_id_from_text(text: str) -> str:
    import re
    match = re.search(r'arXiv:(\d{4}\.\d{4,5})', text)
    if match:
        return match.group(1)
    match = re.search(r'(\d{4}\.\d{4,5})v\d+', text)
    if match:
        return match.group(1)
    return ""

def _extract_abstract_docling(doc_dict: dict) -> str:
    texts = doc_dict.get("texts", [])
    for t in texts:
        if isinstance(t, dict) and "abstract" in t.get("label", "").lower():
            return t.get("text", "")
    return ""

def _extract_sections_docling(doc_dict: dict) -> list:
    texts = doc_dict.get("texts", [])
    sections = []
    for t in texts:
        if isinstance(t, dict):
            sections.append({
                "heading": t.get("label", "Section"),
                "text": t.get("text", ""),
            })
    return sections

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m agents.ingest <path-to-pdf>")
        sys.exit(1)
    result = parse_paper(sys.argv[1])
    print(json.dumps(result, indent=2)[:2000])
