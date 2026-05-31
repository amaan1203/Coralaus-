"""
Component 1.1 — PDF Ingestion & Full Paper JSON

Parses uploaded PDF into a complete structured JSON using a three-phase pipeline:

    Phase 1  extract_identifiers()   PDF → reliable signals only (arXiv ID,
                                     DOI, title, authors, year) via PyPDF2.
    Phase 2  resolve_paper()         IDs → confirmed canonical metadata via
                                     external APIs (arXiv, Semantic Scholar,
                                     OpenAlex, CrossRef).
    Phase 3  fetch_fulltext()        metadata → clean full text from the best
                                     available source (arXiv HTML/LaTeX,
                                     Semantic Scholar, open-access PDF).

PDF content is used as a fallback only if all API sources fail.
"""

import os
import json
import logging
from collections import Counter
import re
from datetime import datetime, timezone
from typing import Optional
from pathlib import Path

from coralaus.config import OUTPUT_DIR

logger = logging.getLogger(__name__)

_KEYWORD_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "in", "into", "is", "it", "of", "on", "or", "that", "the", "their",
    "this", "to", "we", "with", "using", "use", "via", "our",
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def parse_paper(pdf_path: str, output_path: str = None) -> dict:
    """Parse a research paper PDF into structured JSON.

    Orchestrates the three-phase pipeline:
        1. Extract identifiers from PDF (offline, fast, PyPDF2).
        2. Resolve canonical metadata via external APIs.
        3. Fetch clean full text from the best available source.

    Args:
        pdf_path:    Path to the PDF file.
        output_path: Where to save the JSON
                     (default: ``<OUTPUT_DIR>/current_paper.json``).

    Returns:
        Structured paper dict containing title, abstract, authors, year,
        venue, doi, arxiv_id, sections, references, figures, full_text,
        and provenance fields (parsed_by, parsed_at).
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    logger.info(f"Starting paper ingestion pipeline for: {pdf_path}")

    # Phase 1 — extract reliable identifiers from the PDF
    identifiers = extract_identifiers(pdf_path)
    logger.info(
        f"Extracted identifiers — arXiv: {identifiers['arxiv_id']!r}  "
        f"DOI: {identifiers['doi']!r}  title: {identifiers['title_raw'][:60]!r}"
    )

    # Phase 2 — resolve canonical metadata via APIs
    # Import lazily so that resolver/fetcher can be used independently
    from agents.resolver import resolve_paper  # noqa: PLC0415
    resolved = resolve_paper(identifiers)
    logger.info(f"Resolved paper via: {resolved.get('source')}  title: {resolved.get('title', '')[:60]!r}")

    # Phase 3 — fetch clean full text
    from agents.fetcher import fetch_fulltext  # noqa: PLC0415
    fulltext = fetch_fulltext(resolved)
    logger.info(f"Full text fetched via: {fulltext.get('fetched_by')}")

    # Merge all three phases; PDF full-text sections are used as fallback
    paper_json = _merge(identifiers, resolved, fulltext, pdf_path)

    # Save output
    if output_path is None:
        output_path = os.path.join(str(OUTPUT_DIR), "current_paper.json")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(paper_json, f, indent=2, ensure_ascii=False)

    logger.info(f"Paper JSON saved to: {output_path}")
    return paper_json


# ---------------------------------------------------------------------------
# Phase 1 — identifier extraction
# ---------------------------------------------------------------------------

def extract_identifiers(pdf_path: str) -> dict:
    """Extract only reliable identifying signals from a PDF.

    Deliberately ignores body text, equations, and section content —
    those are fetched from authoritative sources in Phase 3.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        Dict with keys:
            arxiv_id     – arXiv ID string or ""
            doi          – DOI string or ""
            title_raw    – best-effort title from PDF metadata or first line
            authors_raw  – list of {"first": str, "last": str, "affiliation": str}
            year         – int or None
    """
    try:
        import PyPDF2

        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            meta = reader.metadata or {}
            first_page = reader.pages[0] if reader.pages else None

            # Read only the first three pages — enough for all identifying signals
            first_pages_text = ""
            for page in reader.pages[:3]:
                first_pages_text += (page.extract_text() or "") + "\n"

        arxiv_id = _extract_arxiv_id(first_pages_text)
        if not arxiv_id:
            # Fallback to filename parsing
            fn = os.path.basename(pdf_path)
            m = re.search(r"(\d{4}\.\d{4,5})", fn)
            if m:
                arxiv_id = m.group(1)
        doi = _extract_doi(first_pages_text)
        title_raw = _extract_title(meta, first_pages_text, first_page)
        authors_raw = _extract_authors(meta)
        year = _extract_year(first_pages_text)

        return {
            "arxiv_id": arxiv_id,
            "doi": doi,
            "title_raw": title_raw,
            "authors_raw": authors_raw,
            "year": year,
        }
    except Exception as e:
        logger.error(f"Identifier extraction failed: {e}")
        return {
            "arxiv_id": "",
            "doi": "",
            "title_raw": os.path.splitext(os.path.basename(pdf_path))[0],
            "authors_raw": [],
            "year": None,
        }


# ---------------------------------------------------------------------------
# Identifier sub-extractors
# ---------------------------------------------------------------------------

def _extract_arxiv_id(text: str) -> str:
    """Return the first arXiv ID found in the text, or an empty string."""
    m = re.search(r'arXiv:(\d{4}\.\d{4,5})', text)
    if m:
        return m.group(1)
    m = re.search(r'(\d{4}\.\d{4,5})v\d+', text)
    if m:
        return m.group(1)
    return ""


def _extract_doi(text: str) -> str:
    """Return the first DOI found in the text, or an empty string."""
    m = re.search(r'\b(10\.\d{4,9}/[^\s,;"\'<>]+)', text, re.IGNORECASE)
    return m.group(1).rstrip('.,') if m else ""


def _extract_title(meta: object, first_pages_text: str, first_page: object = None) -> str:
    """Return the best-available title from metadata, font cues, or page text."""
    raw_title = getattr(meta, 'title', '') or (meta.get('/Title', '') if hasattr(meta, 'get') else '')
    cleaned_meta_title = _clean_title(str(raw_title)) if raw_title else ""
    if cleaned_meta_title and not _is_weak_title(cleaned_meta_title):
        return cleaned_meta_title

    if first_page is not None:
        font_title = _extract_title_from_largest_font(first_page)
        if font_title and not _is_weak_title(font_title):
            return font_title

    lines = [l.strip() for l in first_pages_text.split("\n") if l.strip()]
    for line in lines:
        candidate = _clean_title(line)
        if candidate and not _is_weak_title(candidate):
            return candidate

    return cleaned_meta_title or (_clean_title(lines[0]) if lines else "")


def _extract_authors(meta: object) -> list:
    """Parse author names from PDF metadata, or return an empty list."""
    author_meta = getattr(meta, 'author', '') or (meta.get('/Author', '') if hasattr(meta, 'get') else '')
    if not author_meta:
        return []

    authors = []
    for name in re.split(r'[,;]', str(author_meta)):
        parts = name.strip().split()
        if parts:
            authors.append({
                "first": parts[0],
                "last": parts[-1] if len(parts) > 1 else "",
                "affiliation": "",
            })
    return authors


def _extract_year(text: str) -> Optional[int]:
    """Return the first plausible publication year found in the text."""
    for m in re.finditer(r'\b(20\d{2}|19\d{2})\b', text):
        year = int(m.group(1))
        if 1950 <= year <= datetime.now(timezone.utc).year + 1:
            return year
    return None


def _is_weak_title(title: str) -> bool:
    candidate = _clean_title(title)
    if not candidate:
        return True

    lower = candidate.lower()
    if len(candidate.split()) < 3:
        return True
    if len(candidate) > 250:
        return True
    if re.search(r'(abstract|introduction|copyright|proceedings|www\.|http)', lower):
        return True
    if re.fullmatch(r'[\W\d_]+', candidate):
        return True

    alpha_ratio = sum(1 for ch in candidate if ch.isalpha()) / max(len(candidate), 1)
    return alpha_ratio < 0.45


def _extract_title_from_largest_font(first_page: object) -> str:
    """Use the largest-font text near the top of page 1 as a title candidate."""
    fragments: list[dict] = []

    def visitor_text(text, cm, tm, font_dict, font_size):
        cleaned = _clean_title(text or "")
        if not cleaned:
            return
        y_pos = None
        if isinstance(tm, (list, tuple)) and len(tm) >= 6:
            y_pos = tm[5]
        fragments.append(
            {
                "text": cleaned,
                "font_size": float(font_size or 0),
                "y": float(y_pos or 0),
            }
        )

    try:
        first_page.extract_text(visitor_text=visitor_text)
    except Exception:
        return ""

    if not fragments:
        return ""

    top_fragments = [
        item for item in fragments
        if item["y"] >= 350 and len(item["text"]) >= 3
    ]
    candidates = top_fragments or fragments
    if not candidates:
        return ""

    max_font_size = max(item["font_size"] for item in candidates)
    title_fragments = [
        item for item in candidates
        if item["font_size"] >= max_font_size - 0.5
    ]
    title_fragments.sort(key=lambda item: (-item["y"], item["text"]))

    parts = []
    seen = set()
    for item in title_fragments:
        text = item["text"]
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        parts.append(text)

    candidate = _clean_title(" ".join(parts))
    return candidate


# ---------------------------------------------------------------------------
# Merge helper
# ---------------------------------------------------------------------------

def _merge(identifiers: dict, resolved: dict, fulltext: dict, pdf_path: str) -> dict:
    """Combine the three pipeline phases into a single output dict.

    Precedence:
        - Title, authors, year, venue, doi, arxiv_id → resolved (API) wins
        - Sections, references, figures, abstract     → fulltext (API) wins
        - Fallback to PDF identifiers / empty values if keys are missing
    """
    # Core metadata from resolver (authoritative)
    paper = {
        "title": resolved.get("title") or identifiers.get("title_raw", ""),
        "abstract": fulltext.get("abstract") or resolved.get("abstract", ""),
        "authors": resolved.get("authors") or identifiers.get("authors_raw", []),
        "year": resolved.get("year") or identifiers.get("year"),
        "venue": resolved.get("venue", ""),
        "doi": resolved.get("doi") or identifiers.get("doi", ""),
        "arxiv_id": resolved.get("arxiv_id") or identifiers.get("arxiv_id", ""),
        "keywords": [],
        # Full text content from fetcher (authoritative)
        "sections": fulltext.get("sections", []),
        "references": fulltext.get("references", []),
        "figures": fulltext.get("figures", []),
        "full_text": fulltext.get("full_text", ""),
        # Provenance
        "parsed_by": fulltext.get("fetched_by", "unknown"),
        "resolved_by": resolved.get("source", "unknown"),
        "parsed_at": datetime.now(timezone.utc).isoformat(),
    }

    # If the fetcher returned nothing, fall back to full PyPDF2 parse
    if not paper["sections"]:
        logger.warning("Full-text fetch returned no sections — falling back to PyPDF2")
        pdf_fallback = _parse_with_pypdf(pdf_path)
        if pdf_fallback:
            paper["sections"] = pdf_fallback.get("sections", [])
            paper["full_text"] = pdf_fallback.get("full_text", "")
            paper["abstract"] = paper["abstract"] or pdf_fallback.get("abstract", "")
            paper["parsed_by"] = "pypdf_fallback"

    paper["keywords"] = _extract_keywords(
        paper.get("title", ""),
        paper.get("abstract", ""),
        paper.get("sections", []),
    )

    return paper


# ---------------------------------------------------------------------------
# PyPDF2 full-parse fallback (retained from original, used only as last resort)
# ---------------------------------------------------------------------------

def _parse_with_pypdf(pdf_path: str) -> Optional[dict]:
    """Parse PDF using PyPDF2 (offline, lightweight) as a last-resort fallback.

    This is the same implementation as in the original ingest.py and is called
    only when all API-based full-text sources have been exhausted.
    """
    try:
        import PyPDF2

        logger.info(f"Using PyPDF2 fallback for {pdf_path}")

        with open(pdf_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)

            full_text = ""
            pages_text = []
            for page in reader.pages:
                raw_text = page.extract_text() or ""
                text = _clean_pdf_page_text(raw_text)
                if not text and _sanitize_pdf_text(raw_text):
                    logger.warning("Discarding low-quality text from one PDF page")
                pages_text.append(text)
                if text:
                    full_text += text + "\n\n"

            pages_text = [text for text in pages_text if text]
            full_text = full_text.strip()

            if not pages_text or not full_text:
                logger.warning("PyPDF2 fallback produced no usable text")
                return None

            # Abstract
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
                        abstract = first_page[start_idx:start_idx + 1500].strip()

            # Sections
            sections = []
            current_section_title = "Introduction"
            current_section_text = []

            for page_text in pages_text:
                lines = page_text.split("\n")
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    if re.match(r'^[1-9]\s+[A-Z][a-zA-Z\s]{3,30}$', line) or \
                            re.match(r'^[1-9]\.[0-9]?\s+[A-Z][a-zA-Z\s]{3,30}$', line):
                        if current_section_text:
                            sections.append({
                                "heading": current_section_title,
                                "text": " ".join(current_section_text),
                            })
                        current_section_title = line
                        current_section_text = []
                    else:
                        current_section_text.append(line)

            if current_section_text:
                sections.append({
                    "heading": current_section_title,
                    "text": " ".join(current_section_text),
                })

            if not sections:
                sections = [{"heading": "Full Text", "text": full_text}]

            return {
                "abstract": abstract or full_text[:1000],
                "sections": sections,
                "references": [],
                "figures": [],
                "full_text": full_text,
            }
    except Exception as e:
        logger.error(f"PyPDF2 fallback failed: {e}")
        return None


def _sanitize_pdf_text(text: str) -> str:
    """Remove control-byte artifacts from PDF extraction while keeping prose."""
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = text.replace("\ufffd", " ")
    text = "".join(
        ch for ch in text
        if ch in "\n\r\t" or ord(ch) >= 32
    )
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _looks_like_garbled_text(text: str) -> bool:
    """Heuristic for PDF extraction that returned binary-looking junk."""
    if not text:
        return True

    stripped = text.strip()
    if not stripped:
        return True

    letters = sum(1 for ch in stripped if ch.isalpha())
    weird = sum(
        1 for ch in stripped
        if (ord(ch) < 32 and ch not in "\n\r\t") or ch == "\ufffd"
    )
    punctuation = sum(1 for ch in stripped if not ch.isalnum() and not ch.isspace())

    letter_ratio = letters / max(len(stripped), 1)
    weird_ratio = weird / max(len(stripped), 1)
    punctuation_ratio = punctuation / max(len(stripped), 1)

    if weird_ratio > 0.02:
        return True
    if letter_ratio < 0.35 and len(stripped) > 120:
        return True
    if punctuation_ratio > 0.45 and len(stripped) > 120:
        return True

    return False


def _clean_pdf_page_text(text: str) -> str:
    """Clean a page and drop obviously corrupted lines."""
    cleaned = _sanitize_pdf_text(text)
    if not cleaned:
        return ""

    kept_lines = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if _looks_like_garbled_text(line):
            continue
        kept_lines.append(line)

    return "\n".join(kept_lines).strip()


def _extract_keywords(title: str, abstract: str, sections: list[dict]) -> list[str]:
    """Build a small keyword list from the best local text we have."""
    text_parts = [title, abstract]
    text_parts.extend(section.get("heading", "") for section in sections[:6])
    text_parts.extend(section.get("text", "")[:600] for section in sections[:3])
    text = " ".join(part for part in text_parts if part)
    if not text:
        return []

    candidates = re.findall(r"\b[a-zA-Z][a-zA-Z0-9-]{3,}\b", text.lower())
    counts = Counter(
        token for token in candidates
        if token not in _KEYWORD_STOPWORDS and not token.isdigit()
    )

    keywords = []
    for token, _ in counts.most_common(12):
        if token not in keywords:
            keywords.append(token)
        if len(keywords) == 6:
            break

    return keywords


# ---------------------------------------------------------------------------
# Shared text-cleaning utilities
# ---------------------------------------------------------------------------

def _clean_title(title: str) -> str:
    """Clean PDF-extracted title artifacts (extra spaces, line-break hyphens)."""
    if not title:
        return title
    # Remove trailing hyphen from line-break (e.g. "LAN-" → "LAN")
    title = re.sub(r'-\s*$', '', title)
    # Collapse broken single uppercase letters (e.g. "L OW" → "LOW")
    title = re.sub(r'(?<=\b[A-Z])\s(?=[A-Z])', '', title)
    # Collapse remaining multiple spaces
    title = re.sub(r'\s{2,}', ' ', title)
    return title.strip()


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m agents.ingest <path-to-pdf>")
        sys.exit(1)
    result = parse_paper(sys.argv[1])
    print(json.dumps(result, indent=2)[:2000])
