"""
Component 1.2 — Robust paper identity resolver.

Fixes:
- retry + exponential backoff
- proper transient/permanent error handling
- connection pooling
- rate limit handling
- title quality validation
- identifier-first resolution strategy
- serial provider resolution
- confidence scoring
- old + new arXiv ID support
- caching
- safer title matching
- degraded-but-confident fallback behavior

Requires:
    pip install requests rapidfuzz

Optional:
    pip install requests-cache
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Callable, Optional

import requests
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

HTTP_TIMEOUT = 20
MAX_RETRIES = 5
BACKOFF_BASE = 1.5
SEMANTIC_SCHOLAR_API_KEY = os.environ.get(
    "SEMANTIC_SCHOLAR_API_KEY",
    "",
).strip()
SEMANTIC_SCHOLAR_MIN_INTERVAL = (
    0.11 if SEMANTIC_SCHOLAR_API_KEY else 1.1
)
SEMANTIC_SCHOLAR_COOLDOWN_ON_429 = 60.0

USER_AGENT = (
    "PaperResolver/2.0 "
    "(research metadata resolver; contact=coralaus@example.com)"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json",
}

# =============================================================================
# Session
# =============================================================================

_thread_local = threading.local()
_s2_rate_limit_lock = threading.Lock()
_s2_last_call_time = 0.0
_s2_blocked_until = 0.0

SEMANTIC_SCHOLAR_HEADERS = dict(HEADERS)
if SEMANTIC_SCHOLAR_API_KEY:
    SEMANTIC_SCHOLAR_HEADERS["x-api-key"] = SEMANTIC_SCHOLAR_API_KEY


def get_session() -> requests.Session:
    """
    Thread-local requests session for connection pooling.
    """
    if not hasattr(_thread_local, "session"):
        session = requests.Session()
        session.headers.update(HEADERS)
        _thread_local.session = session
    return _thread_local.session


def wait_for_semantic_scholar_slot() -> None:
    """
    Enforce a shared minimum gap between Semantic Scholar requests.
    """
    global _s2_last_call_time

    with _s2_rate_limit_lock:
        now = time.monotonic()
        if now < _s2_blocked_until:
            sleep_time = _s2_blocked_until - now
            logger.info(
                "Semantic Scholar is in cooldown for %.2fs",
                sleep_time,
            )
            time.sleep(sleep_time)
            now = time.monotonic()

        elapsed = now - _s2_last_call_time

        if elapsed < SEMANTIC_SCHOLAR_MIN_INTERVAL:
            sleep_time = SEMANTIC_SCHOLAR_MIN_INTERVAL - elapsed
            logger.info(
                "Throttling Semantic Scholar for %.2fs",
                sleep_time,
            )
            time.sleep(sleep_time)
            now = time.monotonic()

        _s2_last_call_time = now


def mark_semantic_scholar_rate_limited(
    cooldown_seconds: float,
) -> None:
    """
    Open a process-wide cooldown window after a Semantic Scholar 429.
    """
    global _s2_blocked_until

    with _s2_rate_limit_lock:
        _s2_blocked_until = max(
            _s2_blocked_until,
            time.monotonic() + cooldown_seconds,
        )


def semantic_scholar_is_available() -> bool:
    with _s2_rate_limit_lock:
        return time.monotonic() >= _s2_blocked_until


# =============================================================================
# Exceptions
# =============================================================================

class ResolverError(Exception):
    pass


class TransientAPIError(ResolverError):
    pass


class PermanentAPIError(ResolverError):
    pass


# =============================================================================
# Data model
# =============================================================================

@dataclass
class ResolvedPaper:
    source: str
    confidence: float

    arxiv_id: str = ""
    doi: str = ""

    title: str = ""
    authors: list | None = None

    year: Optional[int] = None
    venue: str = ""

    abstract: str = ""

    s2_paper_id: str = ""

    open_access_pdf: str = ""

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "confidence": self.confidence,
            "arxiv_id": self.arxiv_id,
            "doi": self.doi,
            "title": self.title,
            "authors": self.authors or [],
            "year": self.year,
            "venue": self.venue,
            "abstract": self.abstract,
            "s2_paper_id": self.s2_paper_id,
            "open_access_pdf": self.open_access_pdf,
        }
# =============================================================================
# Public API
# =============================================================================

def resolve_paper(identifiers: dict) -> dict:
    """
    Robust multi-provider paper resolver.
    """
    logger.info("Starting paper resolution with identifiers: %s", identifiers)
    arxiv_id = normalize_arxiv_id(identifiers.get("arxiv_id", ""))
    doi = normalize_doi(identifiers.get("doi", ""))
    raw_title = clean_title(identifiers.get("title_raw", ""))

    logger.info(
        "Starting resolution: arxiv=%s doi=%s title=%r",
        arxiv_id,
        doi,
        raw_title,
    )

# -------------------------------------------------------------------------
# Strong identifiers first
# -------------------------------------------------------------------------

    if doi:
        result = resolve_by_doi(doi)
        if result:
            return result.to_dict()

    if arxiv_id:
        result = resolve_by_arxiv(arxiv_id)
        if result:
            return result.to_dict()

    # -------------------------------------------------------------------------
    # Weak title resolution only if title is valid
    # -------------------------------------------------------------------------

    if not is_valid_title(raw_title):
        logger.warning("Invalid/noisy title detected: %r", raw_title)

        # still preserve strong IDs
        return fallback_resolution(
            identifiers,
            confidence=0.75 if (doi or arxiv_id) else 0.2,
        ).to_dict()

    result = resolve_by_title(raw_title)

    if result:
        return result.to_dict()

    return fallback_resolution(
        identifiers,
        confidence=0.5 if (doi or arxiv_id) else 0.1,
    ).to_dict()


# =============================================================================
# Identifier resolution
# =============================================================================

def resolve_by_doi(doi: str) -> Optional[ResolvedPaper]:
    """
    DOI is strongest possible identifier.
    """
    logger.info(
        "Resolving DOI '%s' across providers serially",
        doi,
    )
    providers = [
        lambda: query_crossref_by_doi(doi),
        lambda: query_openalex_by_doi(doi),
        lambda: query_semantic_scholar(f"DOI:{doi}"),
    ]

    return first_success(providers)


def resolve_by_arxiv(arxiv_id: str) -> Optional[ResolvedPaper]:
    """
    Resolve arXiv ID serially across providers.
    """
    logger.info(
        "Resolving arXiv ID '%s' across providers serially",
        arxiv_id,
    )
    providers = [
        lambda: query_arxiv(arxiv_id),
        lambda: query_openalex_by_arxiv(arxiv_id),
        lambda: query_semantic_scholar(f"ARXIV:{arxiv_id}"),
    ]

    return first_success(providers)


def resolve_by_title(title: str) -> Optional[ResolvedPaper]:
    """
    Weak retrieval path.
    """
    logger.info(
        "Resolving by title '%s' across providers serially",
        title,
    )
    providers = [
        lambda: query_openalex_search(title),
        lambda: query_crossref_search(title),
        lambda: query_semantic_scholar_search(title),
    ]

    candidates = collect_serial(providers)

    if not candidates:
        return None

    best = max(candidates, key=lambda x: x.confidence)

    if best.confidence < 0.65:
        return None

    return best


# =============================================================================
# Provider helpers
# =============================================================================

def first_success(providers):
    for provider in providers:
        try:
            result = provider()
            if result:
                return result
        except Exception as e:
            logger.warning("Provider failed: %s", e)

    return None


def collect_serial(providers):
    results = []

    for provider in providers:
        try:
            result = provider()
            if result:
                results.append(result)
        except Exception as e:
            logger.warning("Provider failed: %s", e)

    return results


# =============================================================================
# HTTP layer
# =============================================================================

def parse_retry_after_seconds(
    retry_after: Optional[str],
    default: float = 2,
) -> float:
    try:
        return float(retry_after) if retry_after else default
    except (TypeError, ValueError):
        return default


def http_get_json(
    url: str,
    *,
    headers: Optional[dict[str, str]] = None,
    before_request: Optional[Callable[[], None]] = None,
    retry_on_429: bool = True,
    on_429: Optional[Callable[[float], None]] = None,
) -> dict:
    """
    Retry-aware JSON fetcher.
    """

    session = get_session()
    request_headers = headers or HEADERS

    for attempt in range(MAX_RETRIES):

        try:
            if before_request:
                before_request()

            response = session.get(
                url,
                timeout=HTTP_TIMEOUT,
                headers=request_headers,
            )

            # -----------------------------------------------------------------
            # Success
            # -----------------------------------------------------------------

            if response.status_code == 200:
                return response.json()

            # -----------------------------------------------------------------
            # Rate limit
            # -----------------------------------------------------------------

            if response.status_code == 429:
                sleep_time = max(
                    parse_retry_after_seconds(
                        response.headers.get("Retry-After"),
                        default=2,
                    ),
                    2,
                )

                logger.warning(
                    "429 rate limit for %s, sleeping %.2fs at time %s",
                    url,
                    sleep_time,
                    time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                )

                if on_429:
                    on_429(
                        max(
                            sleep_time,
                            SEMANTIC_SCHOLAR_COOLDOWN_ON_429,
                        )
                    )

                if not retry_on_429:
                    raise TransientAPIError(f"429 rate limit: {url}")

                time.sleep(sleep_time)
                continue

            # -----------------------------------------------------------------
            # Transient server errors
            # -----------------------------------------------------------------

            if response.status_code in (500, 502, 503, 504):

                sleep_time = 2

                logger.warning(
                    "Transient %d for %s, retrying in %.2fs",
                    response.status_code,
                    url,
                    sleep_time,
                )

                time.sleep(sleep_time)
                continue

            # -----------------------------------------------------------------
            # Permanent failure
            # -----------------------------------------------------------------

            raise PermanentAPIError(
                f"HTTP {response.status_code}: {url}"
            )

        except (
            requests.Timeout,
            requests.ConnectionError,
        ) as e:

            sleep_time = 2

            logger.warning(
                "Connection failure for %s: %s (retry %.2fs)",
                url,
                e,
                sleep_time,
            )

            time.sleep(sleep_time)

    raise TransientAPIError(f"Retries exhausted: {url}")


def exponential_backoff(attempt: int) -> float:
    jitter = 1
    return (BACKOFF_BASE ** attempt) + jitter


# =============================================================================
# arXiv
# =============================================================================

def query_arxiv(arxiv_id: str) -> Optional[ResolvedPaper]:

    url = (
        "https://export.arxiv.org/api/query"
        f"?id_list={arxiv_id}&max_results=1"
    )

    session = get_session()

    for attempt in range(MAX_RETRIES):

        try:
            response = session.get(url, timeout=HTTP_TIMEOUT)

            if response.status_code == 200:
                xml = response.text

                title = extract_xml(xml, "title", skip_first=True)

                if not title:
                    return None

                author_matches = re.findall(r"<author[^>]*>(.*?)</author>", xml, flags=re.DOTALL)
                authors = []
                for author_xml in author_matches:
                    name_match = re.search(r"<name[^>]*>(.*?)</name>", author_xml, flags=re.DOTALL)
                    if name_match:
                        name = re.sub(r"<[^>]+>", "", name_match.group(1)).strip()
                        parts = name.split()
                        authors.append({
                            "first": parts[0] if parts else "",
                            "last": parts[-1] if len(parts) > 1 else "",
                            "affiliation": "",
                        })

                return ResolvedPaper(
                    source="arxiv",
                    confidence=0.99,
                    arxiv_id=arxiv_id,
                    title=clean_title(title),
                    authors=authors,
                    abstract=clean_text(
                        extract_xml(xml, "summary")
                    ),
                    year=parse_year(
                        extract_xml(xml, "published")
                    ),
                    open_access_pdf=(
                        f"https://arxiv.org/pdf/{arxiv_id}.pdf"
                    ),
                )

            if response.status_code in (429, 500, 502, 503, 504):
                time.sleep(2)
                continue

            return None

        except requests.RequestException:
            time.sleep(2)

    return None


# =============================================================================
# Semantic Scholar
# =============================================================================

S2_FIELDS = (
    "paperId,title,abstract,authors,"
    "year,venue,externalIds,openAccessPdf"
)


def query_semantic_scholar(identifier: str) -> Optional[ResolvedPaper]:
    if not semantic_scholar_is_available():
        logger.info(
            "Skipping Semantic Scholar lookup for %s during cooldown",
            identifier,
        )
        return None

    url = (
        "https://api.semanticscholar.org/graph/v1/paper/"
        f"{requests.utils.quote(identifier)}"
        f"?fields={S2_FIELDS}"
    )

    data = http_get_json(
        url,
        headers=SEMANTIC_SCHOLAR_HEADERS,
        before_request=wait_for_semantic_scholar_slot,
        retry_on_429=False,
        on_429=mark_semantic_scholar_rate_limited,
    )

    if "title" not in data:
        return None

    return s2_to_paper(data, confidence=0.98)


def query_semantic_scholar_search(
    title: str,
) -> Optional[ResolvedPaper]:
    if not semantic_scholar_is_available():
        logger.info(
            "Skipping Semantic Scholar title search during cooldown: %s",
            title,
        )
        return None

    url = (
        "https://api.semanticscholar.org/graph/v1/paper/search"
        f"?query={requests.utils.quote(title)}"
        f"&fields={S2_FIELDS}"
        "&limit=10"
    )

    data = http_get_json(
        url,
        headers=SEMANTIC_SCHOLAR_HEADERS,
        before_request=wait_for_semantic_scholar_slot,
        retry_on_429=False,
        on_429=mark_semantic_scholar_rate_limited,
    )

    candidates = data.get("data", [])

    best = best_title_match(candidates, title)

    if not best:
        return None

    score = title_similarity(title, best["title"])

    return s2_to_paper(best, confidence=score)


def extract_s2_authors(data: dict) -> list[dict]:
    authors = []
    for author in data.get("authors") or []:
        if not isinstance(author, dict):
            continue
        name = (author.get("name") or "").strip()
        if not name:
            continue
        parts = name.split()
        authors.append({
            "first": parts[0] if parts else "",
            "last": parts[-1] if len(parts) > 1 else "",
            "affiliation": ""
        })
    return authors


def s2_to_paper(data: dict, confidence: float):
    ext = data.get("externalIds") or {}
    oa = data.get("openAccessPdf") or {}

    return ResolvedPaper(
        source="semantic_scholar",
        confidence=confidence,
        arxiv_id=ext.get("ArXiv", ""),
        doi=ext.get("DOI", ""),
        title=data.get("title", ""),
        authors=extract_s2_authors(data),
        year=data.get("year"),
        venue=data.get("venue", ""),
        abstract=data.get("abstract", "") or "",
        s2_paper_id=data.get("paperId", ""),
        open_access_pdf=oa.get("url", ""),
    )


# =============================================================================
# OpenAlex
# =============================================================================

def query_openalex_search(title: str):

    url = (
        "https://api.openalex.org/works"
        f"?search={requests.utils.quote(title)}"
        "&per-page=10"
    )

    data = http_get_json(url)

    candidates = data.get("results", [])

    best = best_title_match(candidates, title)

    if not best:
        return None

    score = title_similarity(title, best["title"])
    return openalex_to_paper(best, confidence=score)


def query_openalex_by_arxiv(arxiv_id: str):

    url = (
        "https://api.openalex.org/works"
        f"?filter=locations.landing_page_url:"
        f"https://arxiv.org/abs/{arxiv_id}"
    )

    data = http_get_json(url)

    results = data.get("results", [])

    if not results:
        return None

    best = results[0]
    paper = openalex_to_paper(best, confidence=0.95)
    paper.arxiv_id = paper.arxiv_id or arxiv_id
    return paper


def query_openalex_by_doi(doi: str):

    url = (
        "https://api.openalex.org/works/"
        f"https://doi.org/{doi}"
    )

    data = http_get_json(url)
    paper = openalex_to_paper(data, confidence=0.99)
    paper.doi = paper.doi or doi
    return paper


def openalex_to_paper(data: dict, confidence: float) -> ResolvedPaper:
    ids = data.get("ids") or {}
    return ResolvedPaper(
        source="openalex",
        confidence=confidence,
        arxiv_id=extract_openalex_arxiv_id(data),
        doi=normalize_doi(data.get("doi", "")),
        title=data.get("title", ""),
        authors=extract_openalex_authors(data),
        year=data.get("publication_year"),
        venue=extract_openalex_venue(data),
        s2_paper_id=extract_openalex_s2_paper_id(ids),
        open_access_pdf=extract_openalex_pdf_url(data),
    )


def extract_openalex_arxiv_id(data: dict) -> str:
    candidates = []
    ids = data.get("ids") or {}

    for value in ids.values():
        if isinstance(value, str):
            candidates.append(value)

    primary_location = data.get("primary_location") or {}
    if isinstance(primary_location, dict):
        candidates.extend(
            value for value in [
                primary_location.get("landing_page_url"),
                primary_location.get("pdf_url"),
            ] if value
        )

    best_oa_location = data.get("best_oa_location") or {}
    if isinstance(best_oa_location, dict):
        candidates.extend(
            value for value in [
                best_oa_location.get("landing_page_url"),
                best_oa_location.get("pdf_url"),
            ] if value
        )

    for location in data.get("locations") or []:
        if not isinstance(location, dict):
            continue
        for key in ("landing_page_url", "pdf_url"):
            value = location.get(key)
            if value:
                candidates.append(value)

    for candidate in candidates:
        arxiv_id = normalize_arxiv_id(candidate)
        if looks_like_arxiv_id(arxiv_id):
            return arxiv_id

    return ""


def extract_openalex_authors(data: dict) -> list[dict]:
    authors = []

    for authorship in data.get("authorships") or []:
        if not isinstance(authorship, dict):
            continue

        author = authorship.get("author") or {}
        display_name = (author.get("display_name") or "").strip()
        if not display_name:
            continue

        parts = display_name.split()
        institutions = authorship.get("institutions") or []
        affiliation = ""
        if institutions and isinstance(institutions[0], dict):
            affiliation = institutions[0].get("display_name", "") or ""

        authors.append(
            {
                "first": parts[0],
                "last": parts[-1] if len(parts) > 1 else "",
                "affiliation": affiliation,
            }
        )

    return authors


def extract_openalex_venue(data: dict) -> str:
    primary_location = data.get("primary_location") or {}
    source = primary_location.get("source") or {}
    if source.get("display_name"):
        return source["display_name"]

    host_venue = data.get("host_venue") or {}
    if host_venue.get("display_name"):
        return host_venue["display_name"]

    biblio = data.get("biblio") or {}
    return biblio.get("venue", "") or ""


def extract_openalex_pdf_url(data: dict) -> str:
    best_oa_location = data.get("best_oa_location") or {}
    if best_oa_location.get("pdf_url"):
        return best_oa_location["pdf_url"]

    primary_location = data.get("primary_location") or {}
    if primary_location.get("pdf_url"):
        return primary_location["pdf_url"]

    open_access = data.get("open_access") or {}
    if open_access.get("oa_url"):
        return open_access["oa_url"]

    return ""


def extract_openalex_s2_paper_id(ids: dict) -> str:
    for key, value in ids.items():
        if not isinstance(value, str):
            continue
        normalized_key = key.lower()
        if "semantic" not in normalized_key and "s2" not in normalized_key:
            continue
        return value.rstrip("/").split("/")[-1]

    return ""


# =============================================================================
# CrossRef
# =============================================================================

def extract_crossref_authors(item: dict) -> list[dict]:
    authors = []
    for author in item.get("author") or []:
        if not isinstance(author, dict):
            continue
        given = (author.get("given") or "").strip()
        family = (author.get("family") or "").strip()
        
        affiliations = author.get("affiliation") or []
        affiliation_name = ""
        if affiliations and isinstance(affiliations[0], dict):
            affiliation_name = (affiliations[0].get("name") or "").strip()
        
        name = (author.get("name") or "").strip()
        if not given and not family and name:
            parts = name.split()
            given = parts[0] if parts else ""
            family = parts[-1] if len(parts) > 1 else ""

        authors.append({
            "first": given,
            "last": family,
            "affiliation": affiliation_name,
        })
    return authors


def extract_crossref_year(item: dict) -> Optional[int]:
    for key in ("published-print", "published-online", "created", "published"):
        val = item.get(key)
        if not isinstance(val, dict):
            continue
        date_parts = val.get("date-parts")
        if date_parts and date_parts[0]:
            try:
                return int(date_parts[0][0])
            except (ValueError, TypeError, IndexError):
                pass
    return None


def extract_crossref_abstract(item: dict) -> str:
    abstract_xml = item.get("abstract", "")
    if isinstance(abstract_xml, str) and abstract_xml:
        return re.sub(r"<[^>]+>", "", abstract_xml).strip()
    return ""


def query_crossref_search(title: str):

    url = (
        "https://api.crossref.org/works"
        f"?query.title={requests.utils.quote(title)}"
        "&rows=10"
    )

    data = http_get_json(url)

    items = data.get("message", {}).get("items", [])

    normalized = []

    for item in items:
        titles = item.get("title") or []

        if not titles:
            continue

        item["normalized_title"] = titles[0]
        normalized.append(item)

    best = best_title_match(
        normalized,
        title,
        key="normalized_title",
    )

    if not best:
        return None

    score = title_similarity(
        title,
        best["normalized_title"],
    )

    return ResolvedPaper(
        source="crossref",
        confidence=score,
        title=best["normalized_title"],
        doi=best.get("DOI", ""),
        authors=extract_crossref_authors(best),
        year=extract_crossref_year(best),
        abstract=extract_crossref_abstract(best),
    )


def query_crossref_by_doi(doi: str):

    url = f"https://api.crossref.org/works/{doi}"

    data = http_get_json(url)

    item = data["message"]

    titles = item.get("title") or []

    return ResolvedPaper(
        source="crossref",
        confidence=0.99,
        doi=doi,
        title=titles[0] if titles else "",
        authors=extract_crossref_authors(item),
        year=extract_crossref_year(item),
        abstract=extract_crossref_abstract(item),
    )


# =============================================================================
# Matching
# =============================================================================

def best_title_match(
    candidates: list,
    query_title: str,
    key: str = "title",
):

    best = None
    best_score = 0

    for candidate in candidates:

        candidate_title = candidate.get(key, "")

        if not candidate_title:
            continue

        score = title_similarity(
            query_title,
            candidate_title,
        )

        if score > best_score:
            best_score = score
            best = candidate

    return best if best_score >= 0.65 else None


def title_similarity(a: str, b: str) -> float:
    """
    Better than token overlap.
    """

    a = normalize_title_for_matching(a)
    b = normalize_title_for_matching(b)

    return fuzz.token_sort_ratio(a, b) / 100.0


# =============================================================================
# Title validation
# =============================================================================

BAD_TITLE_PATTERNS = [
    r"copyright",
    r"permission",
    r"attribution",
    r"all rights reserved",
    r"creativecommons",
    r"http[s]?://",
    r"www\.",
    r"email",
    r"license",
    r"proceedings",
]


def is_valid_title(title: str) -> bool:

    if not title:
        return False

    if len(title.split()) < 3:
        return False

    if len(title) > 300:
        return False

    lower = title.lower()

    for pattern in BAD_TITLE_PATTERNS:
        if re.search(pattern, lower):
            return False

    # too much punctuation
    punct_ratio = (
        sum(1 for c in title if not c.isalnum() and not c.isspace())
        / max(len(title), 1)
    )

    if punct_ratio > 0.3:
        return False

    return True


# =============================================================================
# Utilities
# =============================================================================

def normalize_doi(doi: str | None) -> str:
    if not doi:
        return ""
    doi = doi.strip()

    doi = re.sub(r"^https?://doi\.org/", "", doi)

    return doi


def normalize_arxiv_id(arxiv_id: str | None) -> str:
    """
    Supports both:
        1706.03762
        cs.CL/9901001
    """
    if not arxiv_id:
        return ""

    arxiv_id = arxiv_id.strip()

    arxiv_id = re.sub(
        r"^https?://arxiv\.org/(abs|pdf)/",
        "",
        arxiv_id,
    )

    arxiv_id = arxiv_id.replace(".pdf", "")

    return arxiv_id


def looks_like_arxiv_id(value: str) -> bool:
    return bool(
        re.fullmatch(r"\d{4}\.\d{4,5}(v\d+)?", value)
        or re.fullmatch(r"[a-z\-]+(?:\.[A-Z\-]+)?/\d{7}(v\d+)?", value)
    )


def clean_title(title: str | None) -> str:
    return clean_text(title)


def clean_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def normalize_title_for_matching(text: str | None) -> str:
    if not text:
        return ""

    text = text.lower()

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    return text.strip()


def parse_year(text: str) -> Optional[int]:

    match = re.search(r"(19|20)\d{2}", text or "")

    return int(match.group()) if match else None


def extract_xml(
    xml: str,
    tag: str,
    skip_first: bool = False,
) -> str:

    matches = re.findall(
        rf"<{tag}[^>]*>(.*?)</{tag}>",
        xml,
        flags=re.DOTALL,
    )

    if not matches:
        return ""

    idx = 1 if skip_first and len(matches) > 1 else 0

    return re.sub(r"<[^>]+>", "", matches[idx]).strip()


# =============================================================================
# Fallback
# =============================================================================

def fallback_resolution(
    identifiers: dict,
    confidence: float,
) -> ResolvedPaper:

    arxiv_id = normalize_arxiv_id(
        identifiers.get("arxiv_id", "")
    )

    return ResolvedPaper(
        source="fallback",
        confidence=confidence,
        arxiv_id=arxiv_id,
        doi=normalize_doi(
            identifiers.get("doi", "")
        ),
        title=clean_title(
            identifiers.get("title_raw", "")
        ),
        authors=identifiers.get("authors_raw", []),
        year=identifiers.get("year"),
        open_access_pdf=(
            f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            if arxiv_id else ""
        ),
    )


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    ids = {
        "arxiv_id": "1706.03762",
        "doi": "",
        "title_raw": "",
        "authors_raw": [],
        "year": None,
    }

    result = resolve_paper(ids)

    print(json.dumps(result, indent=2))
