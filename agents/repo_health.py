"""
Component 3 — Repo Health Check (Coral GitHub Query)

Given a GitHub repo URL, fetches health signals using Coral's built-in
GitHub connector. Pure SQL queries — no LLM.

Input:  GitHub repo URL
Output: health_score (0–100) + signals JSON
"""

import re
import logging
from datetime import date, datetime
from typing import Optional
from agents.coral_utils import get_coral_client

logger = logging.getLogger(__name__)


def check_repo_health(repo_url: str) -> dict:
    """
    Check the health of a GitHub repository using Coral SQL queries.

    Args:
        repo_url: GitHub repository URL (e.g. https://github.com/user/repo)

    Returns:
        Dict with health_score, signals, and raw query results
    """
    owner, repo = _parse_github_url(repo_url)
    if not owner or not repo:
        logger.error(f"Could not parse GitHub URL: {repo_url}")
        return _fallback_health(repo_url, "Invalid GitHub URL")

    coral = get_coral_client()

    # If Coral is not available, use GitHub API fallback
    if not coral.available:
        logger.info("Coral not available, using GitHub API fallback")
        return _github_api_fallback(owner, repo)

    signals = {
        "last_commit_days_ago": None,
        "open_issues": None,
        "stars": None,
        "forks": None,
        "archived": None,
        "year": None,  # Year of last commit — used for temporal version pinning
    }

    # Query 1: Last commit date
    commit_result = coral.sql(f"""
        SELECT commit__committer__date as committed_date
        FROM github.commits
        WHERE owner = '{owner}' AND repo = '{repo}'
        ORDER BY commit__committer__date DESC
        LIMIT 1
    """)
    if commit_result and "results" in commit_result:
        rows = commit_result["results"]
        if rows:
            commit_date_str = rows[0].get("committed_date", "")
            if commit_date_str:
                try:
                    commit_date = datetime.fromisoformat(commit_date_str.replace("Z", "+00:00")).date()
                    signals["last_commit_days_ago"] = (date.today() - commit_date).days
                    signals["year"] = str(commit_date.year)
                except (ValueError, TypeError):
                    pass

    # Query 2: Open issues count
    issues_result = coral.sql(f"""
        SELECT COUNT(*) as open_issues
        FROM github.issues
        WHERE owner = '{owner}' AND repo = '{repo}'
          AND state = 'open'
    """)
    if issues_result and "results" in issues_result:
        rows = issues_result["results"]
        if rows:
            signals["open_issues"] = rows[0].get("open_issues", 0)

    # Query 3: Repo metadata (stars, forks, archived) — use repos_get for fast index lookup
    meta_result = coral.sql(f"""
        SELECT stargazers_count, forks_count, archived
        FROM github.repos_get
        WHERE owner = '{owner}' AND repo = '{repo}'
    """)
    if meta_result and "results" in meta_result:
        rows = meta_result["results"]
        if rows:
            signals["stars"] = rows[0].get("stargazers_count", 0)
            signals["forks"] = rows[0].get("forks_count", 0)
            signals["archived"] = rows[0].get("archived", False)

    # If we couldn't get key signals via Coral (due to auth failure), use GitHub API fallback
    if signals["stars"] is None or signals["last_commit_days_ago"] is None:
        logger.info("Coral query returned no data (possibly auth failure). Falling back to GitHub API...")
        return _github_api_fallback(owner, repo)

    # Compute health score
    health_score = compute_health_score(
        last_commit_days=signals.get("last_commit_days_ago"),
        open_issues=signals.get("open_issues", 0),
        stars=signals.get("stars", 0),
        archived=signals.get("archived", False),
    )

    result = {
        "repo_url": repo_url,
        "owner": owner,
        "repo": repo,
        "health_score": health_score,
        "health_label": _score_label(health_score),
        "health_emoji": _score_emoji(health_score),
        "signals": signals,
        "coral_queries_used": 3,
    }

    logger.info(f"Health score for {owner}/{repo}: {health_score} {_score_emoji(health_score)}")
    return result


def compute_health_score(
    last_commit_days: Optional[int],
    open_issues: int = 0,
    stars: int = 0,
    archived: bool = False
) -> int:
    """
    Compute a health score (0–100) from repo signals.
    Pure Python — no LLM needed.
    """
    if archived:
        return 0

    score = 100

    # Commit recency
    if last_commit_days is not None:
        if last_commit_days > 730:
            score -= 50
        elif last_commit_days > 365:
            score -= 25
        elif last_commit_days > 180:
            score -= 10

    # Issue load
    if open_issues is not None:
        if open_issues > 100:
            score -= 20
        elif open_issues > 50:
            score -= 10

    # Popularity (low stars = less maintained)
    if stars is not None and stars < 10:
        score -= 10

    return max(0, score)


def _parse_github_url(url: str) -> tuple:
    """Extract owner and repo from a GitHub URL."""
    if not url:
        return None, None

    # Handle various GitHub URL formats
    patterns = [
        r"github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
        r"github\.com/([^/]+)/([^/]+?)(?:/.*)?$",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1), match.group(2)

    return None, None


def _score_label(score: int) -> str:
    if score >= 80:
        return "Healthy"
    elif score >= 60:
        return "Fair"
    elif score >= 30:
        return "Stale"
    else:
        return "Dead"


def _score_emoji(score: int) -> str:
    if score >= 80:
        return "🟢"
    elif score >= 60:
        return "🟡"
    elif score >= 30:
        return "🟠"
    else:
        return "🔴"


def _github_api_fallback(owner: str, repo: str) -> dict:
    """Fallback: use GitHub REST API directly when Coral is unavailable."""
    import requests
    import os

    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token and "dummy" not in token.lower() and "your_" not in token.lower():
        headers["Authorization"] = f"token {token}"

    signals = {
        "last_commit_days_ago": None,
        "open_issues": None,
        "stars": None,
        "forks": None,
        "archived": None,
        "year": None,
    }

    try:
        # Get repo metadata
        resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}",
            headers=headers, timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            signals["stars"] = data.get("stargazers_count", 0)
            signals["forks"] = data.get("forks_count", 0)
            signals["archived"] = data.get("archived", False)
            signals["open_issues"] = data.get("open_issues_count", 0)

            pushed_at = data.get("pushed_at")
            if pushed_at:
                try:
                    push_date = datetime.fromisoformat(pushed_at.replace("Z", "+00:00")).date()
                    signals["last_commit_days_ago"] = (date.today() - push_date).days
                    signals["year"] = str(push_date.year)
                except (ValueError, TypeError):
                    pass
    except Exception as e:
        logger.error(f"GitHub API fallback failed: {e}")

    health_score = compute_health_score(
        last_commit_days=signals.get("last_commit_days_ago"),
        open_issues=signals.get("open_issues", 0),
        stars=signals.get("stars", 0),
        archived=signals.get("archived", False),
    )

    return {
        "repo_url": f"https://github.com/{owner}/{repo}",
        "owner": owner,
        "repo": repo,
        "health_score": health_score,
        "health_label": _score_label(health_score),
        "health_emoji": _score_emoji(health_score),
        "signals": signals,
        "coral_queries_used": 0,
        "fallback": True,
    }


def _fallback_health(repo_url: str, reason: str) -> dict:
    return {
        "repo_url": repo_url,
        "health_score": -1,
        "health_label": "Unknown",
        "health_emoji": "⚪",
        "signals": {},
        "error": reason,
        "coral_queries_used": 0,
    }


if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    result = check_repo_health("https://github.com/tensorflow/tensor2tensor")
    print(json.dumps(result, indent=2))
