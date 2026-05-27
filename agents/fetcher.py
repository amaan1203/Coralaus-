"""
Component 1.3 — Full-Text Fetcher

Given a resolved paper identity dict (output of resolver.py), fetches the
complete paper content from the best available source, in priority order:

    arXiv HTML  →  arXiv LaTeX  →  Semantic Scholar full-text  →  Unpaywall PDF

Returns a standardised dict with ``sections``, ``references``, ``abstract``,
and ``figures`` keys — the same schema as the legacy PyPDF2 parser — so the
rest of the pipeline does not need to change.
"""

import io
import json
import logging
import os
import re
import gzip
import tarfile
import urllib.parse
import urllib.request
import urllib.error
import zipfile
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fetch_fulltext(resolved: dict) -> dict:
    """Fetch full paper content from the best available source.

    Priority:
        1. arXiv HTML (clean, math in MathML, no OCR needed)
        2. arXiv LaTeX source (semantic equations, explicit structure)
        3. Semantic Scholar full-text JSON (structured sections)
        4. Unpaywall open-access PDF URL → re-parse with PyPDF2 (last resort)

    Args:
        resolved: Output of ``resolve_paper()`` from resolver.py.

    Returns:
        Dict with keys:
            sections          – list of {"heading": str, "text": str}
            references        – list of {"title": str, "authors": [...], ...}
            abstract          – str (may be richer than resolver abstract)
            figures           – list of {"caption": str, "label": str}
            full_text         – plain concatenation of all section texts
            fetched_by        – source identifier string
            fetched_at        – ISO 8601 UTC timestamp
    """
    arxiv_id = resolved.get("arxiv_id", "")
    s2_paper_id = resolved.get("s2_paper_id", "")
    oa_pdf = resolved.get("open_access_pdf", "")
    doi = resolved.get("doi", "")
    logger.info(
        "Fetching full text for paper with arXiv ID '%s', S2 Paper ID '%s', OA PDF '%s', DOI '%s'",
        arxiv_id,
        s2_paper_id,
        oa_pdf,
        doi,
    )

    result: Optional[dict] = None

    # --- 1. arXiv HTML ---
    if arxiv_id:
        logger.info(f"Fetching arXiv HTML: {arxiv_id}")
        result = _fetch_arxiv_html(arxiv_id)

    # --- 2. arXiv LaTeX source ---
    if result is None and arxiv_id:
        logger.info(f"Fetching arXiv LaTeX source: {arxiv_id}")
        result = _fetch_arxiv_latex(arxiv_id)

    # --- 3. Semantic Scholar full-text ---
    if result is None and s2_paper_id:
        logger.info(f"Fetching Semantic Scholar full-text: {s2_paper_id}")
        result = _fetch_s2_fulltext(s2_paper_id)

    # --- 4. DOI -> Semantic Scholar full-text ---
    if result is None and doi and not s2_paper_id:
        bridged_s2_paper_id = _lookup_s2_paper_id_by_doi(doi)
        if bridged_s2_paper_id:
            logger.info(
                "Resolved DOI %s to Semantic Scholar paper %s",
                doi,
                bridged_s2_paper_id,
            )
            result = _fetch_s2_fulltext(bridged_s2_paper_id)

    # --- 5. Unpaywall / open-access PDF ---
    if result is None and oa_pdf:
        logger.info(f"Fetching open-access PDF: {oa_pdf}")
        result = _fetch_oa_pdf(oa_pdf)

    # --- 6. DOI landing page / Unpaywall ---
    if result is None and doi:
        logger.info(f"Fetching DOI full text: {doi}")
        result = _fetch_doi_fulltext(doi)

    if result is None:
        logger.warning("All full-text sources exhausted — returning empty content")
        result = _empty_fulltext("none")

    return result


# ---------------------------------------------------------------------------
# Strategy 1 — arXiv HTML
# ---------------------------------------------------------------------------

def _fetch_arxiv_html(arxiv_id: str) -> Optional[dict]:
    """Parse the arXiv HTML rendering (available for papers since ~late 2023)."""
    try:
        url = f"https://arxiv.org/html/{arxiv_id}"
        html = _http_get(url)
        if not html:
            return None

        # arXiv HTML renders a 404-style page for papers without HTML versions
        if "not available" in html.lower() or "<h1>404" in html:
            logger.info(f"arXiv HTML not available for {arxiv_id}")
            return None

        sections = _parse_html_sections(html)
        references = _parse_html_references(html)
        figures = _parse_html_figures(html)
        abstract = _parse_html_abstract(html)

        if not sections:
            return None

        return _make_fulltext(sections, references, abstract, figures, "arxiv_html")
    except Exception as e:
        logger.error(f"arXiv HTML fetch error: {e}")
        return None


def _parse_html_sections(html: str) -> list:
    """Extract sections from arXiv HTML.  Headings are in <h2>/<h3> tags."""
    sections = []
    # arXiv HTML wraps sections in <section> tags with an id attribute
    section_blocks = re.findall(
        r'<section[^>]*>(.*?)</section>', html, re.DOTALL | re.IGNORECASE
    )
    if section_blocks:
        for block in section_blocks:
            heading = _strip_tags(re.search(r'<h[1-4][^>]*>(.*?)</h[1-4]>', block, re.DOTALL | re.IGNORECASE).group(1) if re.search(r'<h[1-4][^>]*>(.*?)</h[1-4]>', block, re.DOTALL | re.IGNORECASE) else "")
            if heading.lower() in ("references", "bibliography", "acknowledgements", "acknowledgments"):
                continue
            # Strip all tags, collapse whitespace
            text = _strip_tags(block)
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                sections.append({"heading": heading or "Section", "text": text})
        return sections

    # Fallback: paragraph-level extraction when <section> tags are absent
    paragraphs = re.findall(r'<p[^>]*>(.*?)</p>', html, re.DOTALL | re.IGNORECASE)
    if paragraphs:
        combined = " ".join(_strip_tags(p) for p in paragraphs)
        sections = [{"heading": "Full Text", "text": re.sub(r'\s+', ' ', combined).strip()}]
    return sections


def _parse_html_references(html: str) -> list:
    """Extract references from arXiv HTML reference list."""
    references = []
    ref_section = re.search(
        r'(?:id="references"|id="bib")[^>]*>(.*?)(?:</section>|</div>)',
        html, re.DOTALL | re.IGNORECASE
    )
    if not ref_section:
        return references
    items = re.findall(r'<li[^>]*>(.*?)</li>', ref_section.group(1), re.DOTALL | re.IGNORECASE)
    for item in items[:100]:  # cap at 100 refs
        text = re.sub(r'\s+', ' ', _strip_tags(item)).strip()
        if text:
            references.append({"raw": text})
    return references


def _parse_html_figures(html: str) -> list:
    """Extract figure captions from arXiv HTML."""
    figures = []
    for fig in re.finditer(r'<figure[^>]*>(.*?)</figure>', html, re.DOTALL | re.IGNORECASE):
        caption_m = re.search(r'<figcaption[^>]*>(.*?)</figcaption>', fig.group(1), re.DOTALL | re.IGNORECASE)
        caption = re.sub(r'\s+', ' ', _strip_tags(caption_m.group(1))).strip() if caption_m else ""
        label_m = re.search(r'(?:Figure|Fig\.?)\s+(\d+)', caption, re.IGNORECASE)
        label = label_m.group(0) if label_m else ""
        if caption:
            figures.append({"caption": caption, "label": label})
    return figures


def _parse_html_abstract(html: str) -> str:
    """Extract abstract from arXiv HTML."""
    m = re.search(r'<blockquote[^>]*class="[^"]*abstract[^"]*"[^>]*>(.*?)</blockquote>', html, re.DOTALL | re.IGNORECASE)
    if m:
        return re.sub(r'\s+', ' ', _strip_tags(m.group(1))).strip()
    m = re.search(r'(?:id|class)="abstract"[^>]*>(.*?)</(?:div|section|p)>', html, re.DOTALL | re.IGNORECASE)
    if m:
        return re.sub(r'\s+', ' ', _strip_tags(m.group(1))).strip()
    return ""


# ---------------------------------------------------------------------------
# Strategy 2 — arXiv LaTeX source
# ---------------------------------------------------------------------------

def _fetch_arxiv_latex(arxiv_id: str) -> Optional[dict]:
    """Download the arXiv LaTeX source tarball and parse the main .tex file."""
    try:
        url = f"https://arxiv.org/src/{arxiv_id}"
        raw_bytes = _http_get_bytes(url)
        if not raw_bytes:
            return None

        tex_source = _extract_main_tex(raw_bytes)
        if not tex_source:
            return None

        sections = _parse_latex_sections(tex_source)
        abstract = _parse_latex_abstract(tex_source)
        references = _parse_latex_references(tex_source)

        if not sections:
            return None

        return _make_fulltext(sections, references, abstract, [], "arxiv_latex")
    except Exception as e:
        logger.error(f"arXiv LaTeX fetch error: {e}")
        return None


def _extract_main_tex(raw_bytes: bytes) -> Optional[str]:
    """Extract the main .tex file from common arXiv source archive formats."""
    extractors = [
        _extract_main_tex_from_zip,
        _extract_main_tex_from_tar,
        _extract_main_tex_from_gzip,
        _decode_text_if_likely_tex,
    ]

    for extractor in extractors:
        try:
            tex_source = extractor(raw_bytes)
            if tex_source:
                return tex_source
        except Exception as e:
            logger.debug("LaTeX extractor %s failed: %s", extractor.__name__, e)

    return None


def _extract_main_tex_from_zip(raw_bytes: bytes) -> Optional[str]:
    with zipfile.ZipFile(io.BytesIO(raw_bytes)) as zf:
        tex_members = [
            (name, zf.getinfo(name).file_size)
            for name in zf.namelist()
            if name.lower().endswith(".tex")
        ]
        if not tex_members:
            return None

        main_name = max(tex_members, key=lambda item: item[1])[0]
        return _decode_tex_bytes(zf.read(main_name))


def _extract_main_tex_from_tar(raw_bytes: bytes) -> Optional[str]:
    with tarfile.open(fileobj=io.BytesIO(raw_bytes), mode="r:*") as tf:
        tex_members = [
            member for member in tf.getmembers()
            if member.isfile() and member.name.lower().endswith(".tex")
        ]
        if not tex_members:
            return None

        main_member = max(tex_members, key=lambda member: member.size)
        extracted = tf.extractfile(main_member)
        if extracted is None:
            return None

        return _decode_tex_bytes(extracted.read())


def _extract_main_tex_from_gzip(raw_bytes: bytes) -> Optional[str]:
    if raw_bytes[:2] != b"\x1f\x8b":
        return None

    decompressed = gzip.decompress(raw_bytes)

    # Some sources are a gzipped tar, some are a gzipped single .tex file.
    try:
        tex_source = _extract_main_tex_from_tar(decompressed)
        if tex_source:
            return tex_source
    except tarfile.TarError:
        pass

    return _decode_text_if_likely_tex(decompressed)


def _decode_text_if_likely_tex(raw_bytes: bytes) -> Optional[str]:
    text = _decode_tex_bytes(raw_bytes)
    if not text:
        return None

    latex_markers = (
        "\\documentclass",
        "\\begin{document}",
        "\\section",
        "\\title",
        "\\author",
        "\\begin{abstract}",
    )
    if any(marker in text for marker in latex_markers):
        return text

    return None


def _decode_tex_bytes(raw_bytes: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return raw_bytes.decode(encoding, errors="replace")
        except Exception:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def _parse_latex_sections(tex: str) -> list:
    """Split LaTeX source into sections on \\section / \\subsection commands."""
    sections = []
    pattern = re.compile(
        r'\\(?:sub)*section\*?\{([^}]+)\}(.*?)(?=\\(?:sub)*section|\Z)',
        re.DOTALL
    )
    for m in pattern.finditer(tex):
        heading = m.group(1).strip()
        body = _clean_latex(m.group(2))
        if body:
            sections.append({"heading": heading, "text": body})
    if not sections:
        # No section commands — return the whole document
        body = _clean_latex(tex)
        if body:
            sections = [{"heading": "Full Text", "text": body}]
    return sections


def _parse_latex_abstract(tex: str) -> str:
    """Extract the \\begin{abstract}...\\end{abstract} block."""
    m = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', tex, re.DOTALL)
    return _clean_latex(m.group(1)) if m else ""


def _parse_latex_references(tex: str) -> list:
    """Extract \\bibitem entries as raw reference strings."""
    references = []
    for m in re.finditer(r'\\bibitem(?:\[[^\]]*\])?\{[^}]+\}(.*?)(?=\\bibitem|\Z)', tex, re.DOTALL):
        raw = _clean_latex(m.group(1))
        if raw:
            references.append({"raw": raw})
    return references[:100]


def _clean_latex(tex: str) -> str:
    """Strip common LaTeX commands and normalise whitespace."""
    # Remove comments
    tex = re.sub(r'%[^\n]*', '', tex)
    # Remove common formatting commands, keep content: \textbf{x} → x
    tex = re.sub(r'\\(?:textbf|textit|emph|text|mathrm|mathbf|mathit)\{([^}]+)\}', r'\1', tex)
    # Remove \label, \cite, \ref etc.
    tex = re.sub(r'\\(?:label|cite|ref|eqref|autoref)\{[^}]*\}', '', tex)
    # Remove equation environments (replace with placeholder)
    tex = re.sub(r'\\begin\{(?:equation|align|gather|multline)\*?\}.*?\\end\{(?:equation|align|gather|multline)\*?\}', '[EQUATION]', tex, flags=re.DOTALL)
    # Remove inline math
    tex = re.sub(r'\$[^$]+\$', '[MATH]', tex)
    # Remove remaining backslash commands
    tex = re.sub(r'\\[a-zA-Z]+\*?\{[^}]*\}', '', tex)
    tex = re.sub(r'\\[a-zA-Z]+\s*', ' ', tex)
    # Collapse whitespace
    return re.sub(r'\s+', ' ', tex).strip()


# ---------------------------------------------------------------------------
# Strategy 3 — Semantic Scholar full-text
# ---------------------------------------------------------------------------

def _fetch_s2_fulltext(s2_paper_id: str) -> Optional[dict]:
    """Fetch structured full text from the Semantic Scholar internal endpoint."""
    try:
        # The S2 recommendations API exposes tldr + open access; the /paper
        # endpoint with tldr,sections is only in private beta as of 2024.
        # We use the public /paper endpoint and include available fields.
        url = (
            f"https://api.semanticscholar.org/graph/v1/paper/{s2_paper_id}"
            f"?fields=abstract,tldr,references.title,references.authors,references.year"
        )
        data = _http_get_json(url)
        if not data:
            return None

        abstract = data.get("abstract", "") or ""
        tldr = (data.get("tldr") or {}).get("text", "")

        sections = []
        if abstract:
            sections.append({"heading": "Abstract", "text": abstract})
        if tldr and tldr != abstract:
            sections.append({"heading": "TL;DR", "text": tldr})

        references = []
        for ref in (data.get("references") or [])[:100]:
            title = ref.get("title", "")
            year = ref.get("year")
            authors = [a.get("name", "") for a in (ref.get("authors") or [])]
            references.append({"title": title, "authors": authors, "year": year})

        if not sections:
            return None

        return _make_fulltext(sections, references, abstract, [], "semantic_scholar")
    except Exception as e:
        logger.error(f"Semantic Scholar full-text error: {e}")
        return None


def _lookup_s2_paper_id_by_doi(doi: str) -> str:
    """Resolve a Semantic Scholar paperId from a DOI."""
    encoded_identifier = urllib.parse.quote(f"DOI:{doi}", safe="")
    url = (
        "https://api.semanticscholar.org/graph/v1/paper/"
        f"{encoded_identifier}?fields=paperId"
    )
    data = _http_get_json(url, timeout=20, extra_headers=_s2_headers())
    if not data:
        return ""
    return data.get("paperId", "") or ""


# ---------------------------------------------------------------------------
# Strategy 4 — Open-access PDF re-parse
# ---------------------------------------------------------------------------

def _fetch_oa_pdf(pdf_url: str) -> Optional[dict]:
    """Download an open-access PDF and parse it with PyPDF2 as a last resort."""
    try:
        pdf_bytes = _http_get_bytes(pdf_url)
        if not pdf_bytes:
            return None

        # Write to a temp file, then reuse the PyPDF2 parser from ingest.py
        tmp_path = "/tmp/_coralaus_oa_paper.pdf"
        with open(tmp_path, "wb") as f:
            f.write(pdf_bytes)

        # Import lazily to avoid circular dependency
        from agents.ingest import _parse_with_pypdf  # noqa: PLC0415

        paper = _parse_with_pypdf(tmp_path)
        if paper is None:
            return None

        return _make_fulltext(
            paper.get("sections", []),
            paper.get("references", []),
            paper.get("abstract", ""),
            paper.get("figures", []),
            "oa_pdf_pypdf",
        )
    except Exception as e:
        logger.error(f"OA PDF fetch/parse error: {e}")
        return None


def _fetch_doi_fulltext(doi: str) -> Optional[dict]:
    """Try DOI landing-page HTML first, then Unpaywall OA-PDF lookup."""
    landing_url = f"https://doi.org/{urllib.parse.quote(doi, safe='/:')}"
    html = _http_get(landing_url, timeout=25)
    if html:
        parsed = _parse_generic_html_fulltext(html)
        if parsed:
            return _make_fulltext(
                parsed.get("sections", []),
                [],
                parsed.get("abstract", ""),
                [],
                "doi_html",
            )

    oa_pdf = _lookup_unpaywall_pdf_url(doi)
    if oa_pdf:
        logger.info("Found OA PDF via Unpaywall for DOI %s: %s", doi, oa_pdf)
        return _fetch_oa_pdf(oa_pdf)

    return None


def _lookup_unpaywall_pdf_url(doi: str) -> str:
    """Resolve an open-access PDF URL from Unpaywall when DOI is known."""
    email = os.environ.get("UNPAYWALL_EMAIL", "coralaus@example.com").strip()
    if not email:
        return ""

    url = (
        "https://api.unpaywall.org/v2/"
        f"{urllib.parse.quote(doi, safe='')}"
        f"?email={urllib.parse.quote(email, safe='')}"
    )
    data = _http_get_json(url, timeout=20)
    if not data:
        return ""

    best_oa_location = data.get("best_oa_location") or {}
    if best_oa_location.get("url_for_pdf"):
        return best_oa_location["url_for_pdf"]

    oa_locations = data.get("oa_locations") or []
    for location in oa_locations:
        if not isinstance(location, dict):
            continue
        if location.get("url_for_pdf"):
            return location["url_for_pdf"]

    return ""


def _parse_generic_html_fulltext(html: str) -> Optional[dict]:
    """Extract usable text from a generic publisher/landing-page HTML document."""
    abstract = _parse_generic_html_abstract(html)

    sections = _parse_html_sections(html)
    if not sections:
        sections = _parse_generic_html_sections(html)

    cleaned_sections = []
    for section in sections:
        heading = (section.get("heading") or "Section").strip()
        text = re.sub(r"\s+", " ", section.get("text", "")).strip()
        if len(text) < 120:
            continue
        cleaned_sections.append({"heading": heading, "text": text})

    if not cleaned_sections and abstract:
        cleaned_sections = [{"heading": "Abstract", "text": abstract}]

    if not cleaned_sections:
        return None

    return {
        "abstract": abstract,
        "sections": cleaned_sections,
    }


def _parse_generic_html_abstract(html: str) -> str:
    patterns = [
        r'<meta[^>]+name="citation_abstract"[^>]+content="([^"]+)"',
        r'<meta[^>]+property="og:description"[^>]+content="([^"]+)"',
        r'<section[^>]*class="[^"]*abstract[^"]*"[^>]*>(.*?)</section>',
        r'<div[^>]*class="[^"]*abstract[^"]*"[^>]*>(.*?)</div>',
    ]

    for pattern in patterns:
        match = re.search(pattern, html, re.DOTALL | re.IGNORECASE)
        if not match:
            continue
        abstract = re.sub(r"\s+", " ", _strip_tags(match.group(1))).strip()
        if len(abstract) >= 40:
            return abstract

    return ""


def _parse_generic_html_sections(html: str) -> list[dict]:
    paragraphs = re.findall(r"<p[^>]*>(.*?)</p>", html, re.DOTALL | re.IGNORECASE)
    text_chunks = []
    for paragraph in paragraphs:
        text = re.sub(r"\s+", " ", _strip_tags(paragraph)).strip()
        if len(text) >= 80:
            text_chunks.append(text)

    if not text_chunks:
        return []

    body = " ".join(text_chunks[:30]).strip()
    if len(body) < 200:
        return []

    return [{"heading": "Full Text", "text": body}]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_fulltext(
    sections: list,
    references: list,
    abstract: str,
    figures: list,
    source: str,
) -> dict:
    """Assemble the standardised fulltext dict."""
    full_text = "\n\n".join(
        f"{s.get('heading', '')}\n{s.get('text', '')}" for s in sections
    )
    return {
        "sections": sections,
        "references": references,
        "abstract": abstract,
        "figures": figures,
        "full_text": full_text,
        "fetched_by": source,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
    }


def _empty_fulltext(source: str) -> dict:
    return _make_fulltext([], [], "", [], source)


def _strip_tags(html: str) -> str:
    """Remove all HTML/XML tags from a string."""
    return re.sub(r'<[^>]+>', '', html)


_HEADERS = {
    "User-Agent": "Coralaus/1.0 (research paper analyser; mailto:coralaus@example.com)"
}


def _http_get(
    url: str,
    timeout: int = 20,
    extra_headers: Optional[dict] = None,
) -> Optional[str]:
    """GET a URL and return the response body as text, or None on error."""
    try:
        headers = dict(_HEADERS)
        if extra_headers:
            headers.update(extra_headers)
        req = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        logger.warning(f"HTTP {e.code} for {url}")
        return None
    except Exception as e:
        logger.warning(f"Request failed for {url}: {e}")
        return None


def _http_get_bytes(url: str, timeout: int = 30) -> Optional[bytes]:
    """GET a URL and return the raw response bytes, or None on error."""
    try:
        req = urllib.request.Request(url, headers=_HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read()
    except urllib.error.HTTPError as e:
        logger.warning(f"HTTP {e.code} for {url}")
        return None
    except Exception as e:
        logger.warning(f"Request failed for {url}: {e}")
        return None


def _http_get_json(
    url: str,
    timeout: int = 15,
    extra_headers: Optional[dict] = None,
) -> Optional[dict]:
    """GET a URL and parse the response as JSON, or None on error."""
    text = _http_get(url, timeout=timeout, extra_headers=extra_headers)
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning(f"JSON decode error for {url}: {e}")
        return None


def _s2_headers() -> dict:
    headers = {}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    if api_key:
        headers["x-api-key"] = api_key
    return headers


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) < 2:
        print("Usage: python -m agents.fetcher <arxiv_id>")
        sys.exit(1)
    resolved_stub = {
        "arxiv_id": sys.argv[1],
        "s2_paper_id": "",
        "open_access_pdf": "",
    }
    result = fetch_fulltext(resolved_stub)
    print(json.dumps(result, indent=2)[:3000])
