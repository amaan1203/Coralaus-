"""
Component 2 — PapersWithCode Search

Searches PapersWithCode for existing implementations of a paper.
Uses the PapersWithCode API (same data exposed by the MCP server).

Input:  Paper JSON (title, arxiv_id from Component 1)
Output: Best repo URL + metadata, or None if no implementation found
"""

import logging
from typing import Optional
from agents.pwc_mcp_client import get_pwc_client

logger = logging.getLogger(__name__)


def search_implementation(paper_json: dict) -> dict:
    """
    Search PapersWithCode for implementations of the given paper.

    Args:
        paper_json: Parsed paper JSON from Component 1

    Returns:
        Dict with:
            - found (bool): Whether an implementation was found
            - repo_url (str): GitHub repo URL
            - stars (int): Repo star count
            - framework (str): ML framework used
            - is_official (bool): Official implementation flag
            - paper_id (str): PapersWithCode paper ID
            - search_method (str): How the repo was found
    """
    title = paper_json.get("title", "")
    arxiv_id = paper_json.get("arxiv_id", "")
    pwc = get_pwc_client()

    result = {
        "found": False,
        "repo_url": None,
        "stars": 0,
        "framework": None,
        "is_official": False,
        "paper_id": None,
        "search_method": None,
    }

    # Strategy 1: Search by arXiv ID (most precise)
    if arxiv_id:
        logger.info(f"Searching PWC by arXiv ID: {arxiv_id}")
        paper = pwc.get_paper_by_arxiv_id(arxiv_id)
        if paper:
            paper_id = paper.get("id")
            result["paper_id"] = paper_id
            repos = pwc.get_paper_repositories(paper_id) if paper_id else []
            best = pwc._pick_best_repo(repos)
            if best:
                result.update({
                    "found": True,
                    "repo_url": best.get("url"),
                    "stars": best.get("stars", 0),
                    "framework": best.get("framework"),
                    "is_official": best.get("is_official", False),
                    "search_method": "arxiv_id",
                })
                logger.info(f"Found repo via arXiv ID: {result['repo_url']}")
                return result

    # Strategy 2: Search by title
    if title:
        logger.info(f"Searching PWC by title: '{title[:60]}'")
        papers = pwc.search_papers(title, items_per_page=5)
        for paper in papers:
            paper_id = paper.get("id")
            if paper_id:
                repos = pwc.get_paper_repositories(paper_id)
                best = pwc._pick_best_repo(repos)
                if best:
                    result.update({
                        "found": True,
                        "repo_url": best.get("url"),
                        "stars": best.get("stars", 0),
                        "framework": best.get("framework"),
                        "is_official": best.get("is_official", False),
                        "paper_id": paper_id,
                        "search_method": "title",
                    })
                    logger.info(f"Found repo via title search: {result['repo_url']}")
                    return result

    # Strategy 3: Search with keywords from title
    if title:
        # Use first few significant words as search terms
        keywords = _extract_search_terms(title)
        if keywords:
            logger.info(f"Searching PWC by keywords: '{keywords}'")
            papers = pwc.search_papers(keywords, items_per_page=3)
            for paper in papers:
                paper_id = paper.get("id")
                if paper_id:
                    repos = pwc.get_paper_repositories(paper_id)
                    best = pwc._pick_best_repo(repos)
                    if best:
                        result.update({
                            "found": True,
                            "repo_url": best.get("url"),
                            "stars": best.get("stars", 0),
                            "framework": best.get("framework"),
                            "is_official": best.get("is_official", False),
                            "paper_id": paper_id,
                            "search_method": "keywords",
                        })
                        logger.info(f"Found repo via keyword search: {result['repo_url']}")
                        return result

    logger.info("No implementation found on PapersWithCode. Trying direct GitHub search fallback...")

    # Strategy 4: Check if the abstract/paper body mentions a GitHub URL directly
    # This is the most authoritative signal — the authors link to their own code
    abstract = paper_json.get("abstract", "")
    full_text = paper_json.get("full_text", "")
    mentioned_url = _extract_github_url(abstract) or _extract_github_url(full_text)
    if mentioned_url:
        logger.info(f"Found GitHub URL mentioned in paper: {mentioned_url}")
        result.update({
            "found": True,
            "repo_url": mentioned_url,
            "stars": 0,
            "framework": None,
            "is_official": True,
            "search_method": "paper_body_url",
        })
        return result

    # Strategy 5: Direct GitHub search by arXiv ID
    if arxiv_id:
        logger.info(f"Searching GitHub directly for arXiv ID: {arxiv_id}")
        repos = _search_github_repositories(f"\"{arxiv_id}\"")
        if repos:
            best = _pick_best_github_repo(repos, paper_json)
            if best:
                result.update({
                    "found": True,
                    "repo_url": best["html_url"],
                    "stars": best.get("stargazers_count", best.get("stars", 0)) or 0,
                    "framework": _detect_framework(best.get("description", "")),
                    "is_official": best.get("_is_official", False),
                    "search_method": "direct_github_arxiv_id",
                })
                logger.info(f"Found repo via direct GitHub arXiv ID search: {result['repo_url']}")
                return result

    # Strategy 6: Direct GitHub search by paper title keywords
    if title:
        keywords = _extract_search_terms(title)
        if keywords:
            logger.info(f"Searching GitHub directly for title keywords: {keywords}")
            repos = _search_github_repositories(f"{keywords} language:python")
            if repos:
                best = _pick_best_github_repo(repos, paper_json)
                if best:
                    result.update({
                        "found": True,
                        "repo_url": best["html_url"],
                        "stars": best.get("stargazers_count", best.get("stars", 0)) or 0,
                        "framework": _detect_framework(best.get("description", "")),
                        "is_official": best.get("_is_official", False),
                        "search_method": "direct_github_title",
                    })
                    logger.info(f"Found repo via direct GitHub title search: {result['repo_url']}")
                    return result

    logger.info("No implementation found on PapersWithCode or GitHub search")
    return result


def _search_github_repositories(query: str) -> list:
    """Search GitHub for repositories using Coral SQL (or API fallback)."""
    import os
    from agents.coral_utils import get_coral_client

    coral = get_coral_client()
    if coral.available:
        # Run Coral SQL — ORDER BY stars to get the most popular first
        res = coral.sql(f"""
            SELECT full_name, html_url, stargazers_count, description
            FROM github.search_repositories(q => '{query}')
            ORDER BY stargazers_count DESC
            LIMIT 10
        """)
        if res and "results" in res:
            mapped = []
            for row in res["results"]:
                mapped.append({
                    "full_name": row.get("full_name"),
                    "html_url": row.get("html_url") or f"https://github.com/{row.get('full_name')}",
                    "stargazers_count": row.get("stargazers_count", 0),
                    "description": row.get("description", "")
                })
            return mapped

    # Fallback to GitHub REST API
    import requests
    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token and "dummy" not in token.lower() and "your_" not in token.lower():
        headers["Authorization"] = f"token {token}"

    try:
        resp = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "stars", "order": "desc", "per_page": 10},
            headers=headers, timeout=10
        )
        if resp.status_code == 200:
            return resp.json().get("items", [])
    except Exception as e:
        logger.error(f"GitHub search fallback failed: {e}")

    return []


def _pick_best_github_repo(repos: list, paper_json: dict = None) -> Optional[dict]:
    """
    Pick the best repo from GitHub search results.
    Prefers: repos mentioned in the paper > repos by known orgs > most stars.
    """
    if not repos:
        return None

    # Check if the paper mentions a specific GitHub URL
    if paper_json:
        abstract = paper_json.get("abstract", "")
        full_text = paper_json.get("full_text", "")
        mentioned_url = _extract_github_url(abstract) or _extract_github_url(full_text)
        if mentioned_url:
            # Try to match the mentioned URL to one of our search results
            for repo in repos:
                repo_url = repo.get("html_url", "")
                if repo_url and mentioned_url.rstrip("/").lower() == repo_url.rstrip("/").lower():
                    repo["_is_official"] = True
                    logger.info(f"Matched paper-mentioned repo: {repo_url}")
                    return repo

    # Score repos: prefer orgs, high stars, non-fork/homework names
    def _repo_score(repo):
        stars = repo.get("stargazers_count", repo.get("stars", 0)) or 0
        name = (repo.get("full_name") or "").lower()
        desc = (repo.get("description") or "").lower()

        score = stars

        # Penalize repos that look like homework/reimplementations
        penalty_words = ["homework", "assignment", "coursework", "cs4", "cs2",
                         "course", "class", "tutorial", "my-", "fork"]
        for word in penalty_words:
            if word in name or word in desc:
                score -= 10000

        # Boost repos from known ML orgs
        known_orgs = ["microsoft", "google", "meta", "facebook", "openai",
                      "huggingface", "pytorch", "tensorflow", "deepmind",
                      "nvidia", "apple", "amazon", "aws"]
        owner = name.split("/")[0] if "/" in name else ""
        if owner in known_orgs:
            score += 50000

        # Boost repos whose name matches a well-known abbreviation from the title
        if paper_json:
            title_lower = paper_json.get("title", "").lower()
            repo_name = name.split("/")[-1] if "/" in name else name
            if repo_name and repo_name in title_lower:
                score += 20000

        return score

    best = max(repos, key=_repo_score)
    best_score = _repo_score(best)
    # Mark as official if it's from a known org
    owner = (best.get("full_name") or "").split("/")[0].lower()
    known_orgs = ["microsoft", "google", "meta", "facebook", "openai",
                  "huggingface", "pytorch", "tensorflow", "deepmind"]
    best["_is_official"] = owner in known_orgs
    return best


def _extract_github_url(text: str) -> Optional[str]:
    """Extract a GitHub repo URL from text (e.g. from abstract or paper body)."""
    import re
    if not text:
        return None
    match = re.search(r'https?://github\.com/([\w.-]+/[\w.-]+)', text)
    if match:
        return f"https://github.com/{match.group(1)}"
    return None


def _detect_framework(description: str) -> str:
    """Detect ML framework from repository description."""
    description = (description or "").lower()
    if "pytorch" in description or "torch" in description:
        return "PyTorch"
    if "tensorflow" in description or "tf" in description:
        return "TensorFlow"
    if "jax" in description:
        return "JAX"
    return "Python"


def _extract_search_terms(title: str) -> str:
    """Extract significant search terms from a paper title."""
    stop_words = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "for", "and", "nor", "but",
        "or", "yet", "so", "at", "by", "from", "in", "of", "on", "to", "with",
        "as", "into", "through", "during", "before", "after", "above", "below",
        "between", "under", "over", "all", "each", "every", "both", "few",
        "more", "most", "other", "some", "such", "no", "not", "only", "own",
        "same", "than", "too", "very", "just", "about", "up", "out", "off",
        "its", "it", "this", "that", "these", "those", "what", "which", "who",
        "whom", "how", "when", "where", "why", "new", "via", "using",
    }
    words = [w for w in title.lower().split() if w not in stop_words and len(w) > 2]
    return " ".join(words[:5])


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    # Quick test with a known paper
    test_paper = {
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
    }
    result = search_implementation(test_paper)
    print(json.dumps(result, indent=2))
