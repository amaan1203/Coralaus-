"""
Semantic Scholar API Client
Replaces the dead PapersWithCode MCP client.

Searches for research papers using the Semantic Scholar Graph API,
then discovers implementations via GitHub repository search.

API docs: https://api.semanticscholar.org/graph/v1
Rate limits:
  - Unauthenticated: ~1 req/s
  - With SEMANTIC_SCHOLAR_API_KEY: 10 req/s

Input:  Paper title or arXiv ID
Output: Paper metadata + best GitHub repo URL
"""

import os
import time
import logging
import requests
from typing import Optional

logger = logging.getLogger(__name__)


class SemanticScholarClient:
    """
    Drop-in replacement for the old PapersWithCodeMCPClient.

    Uses the Semantic Scholar Graph API for paper lookup and GitHub
    search for finding implementations (S2 does not expose code links directly).

    Optional: set SEMANTIC_SCHOLAR_API_KEY in .env for higher rate limits.
    """

    S2_API_BASE = "https://api.semanticscholar.org/graph/v1"
    PAPER_FIELDS = "paperId,title,year,authors,externalIds,openAccessPdf,url"

    # Unauthenticated S2 allows ~1 req/s; add a gap between consecutive calls
    _MIN_INTERVAL = 1.1  # seconds

    def __init__(self):
        self.call_count = 0
        self.available = True
        self._api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY", "")
        self._last_call_time: float = 0.0
        if self._api_key:
            logger.info("Semantic Scholar client ready (authenticated — 10 req/s)")
        else:
            logger.info("Semantic Scholar client ready (unauthenticated — ~1 req/s)")

    # ------------------------------------------------------------------
    # Public API — same interface as old PapersWithCodeMCPClient
    # ------------------------------------------------------------------

    def search_papers(self, query: str, items_per_page: int = 5) -> list[dict]:
        """Search for papers by title or keyword."""
        self.call_count += 1
        logger.info(f"[S2 #{self.call_count}] Searching papers: '{query[:80]}'")

        resp = self._get(
            f"{self.S2_API_BASE}/paper/search",
            params={"query": query, "limit": items_per_page, "fields": self.PAPER_FIELDS},
        )
        if resp is None:
            return []
        if resp.status_code == 200:
            results = resp.json().get("data", [])
            logger.info(f"  Found {len(results)} papers")
            return [self._normalize_paper(p) for p in results]
        else:
            logger.error(f"  S2 search error: {resp.status_code} — {resp.text[:200]}")
            return []

    def get_paper_by_arxiv_id(self, arxiv_id: str) -> Optional[dict]:
        """Look up a specific paper by its arXiv ID."""
        self.call_count += 1
        logger.info(f"[S2 #{self.call_count}] Looking up arXiv:{arxiv_id}")

        resp = self._get(
            f"{self.S2_API_BASE}/paper/arXiv:{arxiv_id}",
            params={"fields": self.PAPER_FIELDS},
        )
        if resp is None:
            return None
        if resp.status_code == 200:
            paper = resp.json()
            logger.info(f"  Found: {paper.get('title', 'Unknown')}")
            return self._normalize_paper(paper)
        elif resp.status_code == 404:
            logger.info(f"  Paper arXiv:{arxiv_id} not found in Semantic Scholar")
            return None
        else:
            logger.error(f"  S2 lookup error: {resp.status_code} — {resp.text[:200]}")
            return None

    def get_paper_repositories(self, paper_id: str) -> list[dict]:
        """
        Find GitHub repositories implementing this paper.

        Strategy:
          1. Hugging Face Papers API  — extracts the official GitHub URL
             from the abstract as written by the authors themselves.
             This is the most authoritative source (is_official=True).
          2. GitHub search fallback   — searches repos that reference
             the arXiv ID (less precise, is_official=False).

        paper_id is expected to be the arXiv ID (set by _normalize_paper).
        """
        if not paper_id:
            return []

        # Strategy 1: HF Papers API — official link from authors' abstract
        self.call_count += 1
        logger.info(f"[HF #{self.call_count}] Checking HF Papers API for official repo: {paper_id}")
        official = self._get_official_repo_from_hf(paper_id)
        if official:
            return [official]

        # Strategy 2: GitHub search fallback
        self.call_count += 1
        logger.info(f"[S2 #{self.call_count}] No official repo found — falling back to GitHub search: {paper_id}")
        repos = self._search_github(f'"{paper_id}" language:python')
        if not repos:
            repos = self._search_github(f'"{paper_id}"')

        return repos

    def _pick_best_repo(self, repos: list[dict]) -> Optional[dict]:
        """Pick the best repo: prefer official implementations, then most stars."""
        if not repos:
            return None
        official = [r for r in repos if r.get("is_official")]
        if official:
            return max(official, key=lambda r: r.get("stars", 0))
        return max(repos, key=lambda r: r.get("stars", 0))

    def get_call_count(self) -> int:
        return self.call_count

    def reset_call_count(self):
        self.call_count = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _headers(self) -> dict:
        """Build request headers, adding API key if available."""
        h = {"Accept": "application/json"}
        if self._api_key:
            h["x-api-key"] = self._api_key
        return h

    def _get(self, url: str, params: dict = None, retries: int = 3) -> Optional[requests.Response]:
        """
        Rate-limit-aware GET with automatic retry on 429.
        Enforces a minimum interval between calls so we stay under the
        unauthenticated 1 req/s limit.
        """
        # Throttle: wait if we're calling too fast
        if not self._api_key:
            elapsed = time.time() - self._last_call_time
            if elapsed < self._MIN_INTERVAL:
                time.sleep(self._MIN_INTERVAL - elapsed)

        for attempt in range(retries):
            try:
                resp = requests.get(url, params=params, headers=self._headers(), timeout=15)
                self._last_call_time = time.time()

                if resp.status_code == 429:
                    wait = int(resp.headers.get("Retry-After", 2)) + 1
                    logger.warning(f"  S2 rate-limited (429). Waiting {wait}s before retry {attempt + 1}/{retries}...")
                    time.sleep(wait)
                    continue

                return resp
            except requests.RequestException as e:
                logger.error(f"  S2 request error (attempt {attempt + 1}): {e}")
                time.sleep(1)

        logger.error(f"  S2 request failed after {retries} retries: {url}")
        return None

    def _get_official_repo_from_hf(self, arxiv_id: str) -> Optional[dict]:
        """
        Call the Hugging Face Papers API and extract the official GitHub URL
        from the paper's abstract (summary). Authors typically link their
        own implementation in their abstract, making this the most reliable
        source for an official repo.

        API: https://huggingface.co/api/papers/{arxiv_id}
        No authentication required.
        """
        try:
            resp = requests.get(
                f"https://huggingface.co/api/papers/{arxiv_id}",
                timeout=10,
            )
            if resp.status_code == 200:
                data = resp.json()
                summary = data.get("summary", "")
                github_url = self._extract_github_url(summary)
                if github_url:
                    logger.info(f"  Official repo found in HF Papers abstract: {github_url}")
                    return {
                        "url": github_url,
                        "stars": 0,  # not available from this source
                        "framework": self._detect_framework(summary),
                        "is_official": True,
                    }
                else:
                    logger.info(f"  HF Papers: paper found but no GitHub URL in abstract")
            elif resp.status_code == 404:
                logger.info(f"  HF Papers: paper {arxiv_id} not indexed")
            else:
                logger.warning(f"  HF Papers API error: {resp.status_code}")
        except Exception as e:
            logger.error(f"  HF Papers API failed: {e}")
        return None

    def _extract_github_url(self, text: str) -> Optional[str]:
        """Extract the first GitHub repository URL from a block of text."""
        import re
        match = re.search(
            r'https?://github\.com/([a-zA-Z0-9_.-]+)/([a-zA-Z0-9_.-]+)',
            text
        )
        if match:
            owner, repo = match.group(1), match.group(2)
            return f"https://github.com/{owner}/{repo}"
        return None

    def _search_github(self, query: str) -> list[dict]:
        """Search GitHub repositories and normalize results."""
        headers = {}
        token = os.environ.get("GITHUB_TOKEN", "")
        if token and "dummy" not in token.lower() and "your_" not in token.lower():
            headers["Authorization"] = f"token {token}"

        try:
            resp = requests.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "sort": "stars", "per_page": 5},
                headers=headers,
                timeout=10,
            )
            if resp.status_code == 200:
                items = resp.json().get("items", [])
                logger.info(f"  GitHub search returned {len(items)} repos for: {query[:60]}")
                return [self._normalize_github_repo(r) for r in items]
            else:
                logger.error(f"  GitHub search error: {resp.status_code}")
        except Exception as e:
            logger.error(f"  GitHub search failed: {e}")

        return []

    def _normalize_paper(self, paper: dict) -> dict:
        """
        Normalize a Semantic Scholar paper response to a consistent internal format.
        The 'id' field is set to the arXiv ID so get_paper_repositories can
        use it directly for GitHub search.
        """
        external_ids = paper.get("externalIds") or {}
        arxiv_id = external_ids.get("ArXiv", "")
        return {
            # 'id' is used as the key passed to get_paper_repositories
            "id": arxiv_id or paper.get("paperId", ""),
            "paper_id": paper.get("paperId", ""),
            "arxiv_id": arxiv_id,
            "title": paper.get("title", ""),
            "year": paper.get("year"),
            "url": paper.get("url", ""),
        }

    def _normalize_github_repo(self, repo: dict) -> dict:
        """Normalize a GitHub API repo result to the internal repo format."""
        return {
            "url": repo.get("html_url", ""),
            "stars": repo.get("stargazers_count", 0),
            "framework": self._detect_framework(repo.get("description") or ""),
            "is_official": False,  # GitHub search cannot determine officiality
        }

    def _detect_framework(self, description: str) -> str:
        """Detect ML framework from repository description."""
        desc = description.lower()
        if "pytorch" in desc or "torch" in desc:
            return "PyTorch"
        if "tensorflow" in desc or " tf " in desc:
            return "TensorFlow"
        if "jax" in desc:
            return "JAX"
        return "Python"


# ---------------------------------------------------------------------------
# Singleton
# ---------------------------------------------------------------------------
_client: Optional[SemanticScholarClient] = None


def get_s2_client() -> SemanticScholarClient:
    """Get or create the global Semantic Scholar client."""
    global _client
    if _client is None:
        _client = SemanticScholarClient()
    return _client
