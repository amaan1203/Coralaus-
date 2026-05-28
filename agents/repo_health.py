"""
Component 3 — Repo Health Check (Coral GitHub Query)

Given a GitHub repo URL, fetches health signals using Coral's built-in
GitHub connector. Pure SQL queries — no LLM.

Input:  GitHub repo URL
Output: health_score (0–100) + signals JSON

Scoring signals (v2):
  - Commit recency           (primary decay signal)
  - Issue resolution ratio   (closed / total — replaces raw open count)
  - Star velocity            (stars/year — replaces hard star cutoff)
  - Fork count               (community adoption bonus)
  - Contributor count        (bus-factor / maintenance risk)
  - CI/CD presence           (quality signal)
  - Archived flag            (instant zero)
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
        "closed_issues": None,
        "stars": None,
        "forks": None,
        "archived": None,
        "year": None,
        "repo_age_days": None,
        "contributor_count": None,
        "has_ci": None,
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

    # Query 3: Repo metadata (stars, forks, archived, created_at)
    meta_result = coral.sql(f"""
        SELECT stargazers_count, forks_count, archived, created_at
        FROM github.repos_get
        WHERE owner = '{owner}' AND repo = '{repo}'
    """)
    if meta_result and "results" in meta_result:
        rows = meta_result["results"]
        if rows:
            signals["stars"] = rows[0].get("stargazers_count", 0)
            signals["forks"] = rows[0].get("forks_count", 0)
            signals["archived"] = rows[0].get("archived", False)
            created_at = rows[0].get("created_at")
            if created_at:
                try:
                    created_date = datetime.fromisoformat(created_at.replace("Z", "+00:00")).date()
                    signals["repo_age_days"] = (date.today() - created_date).days
                except (ValueError, TypeError):
                    pass

    # If we couldn't get key signals via Coral (due to auth failure), use GitHub API fallback
    if signals["stars"] is None or signals["last_commit_days_ago"] is None:
        logger.info("Coral query returned no data (possibly auth failure). Falling back to GitHub API...")
        return _github_api_fallback(owner, repo)

    # Compute health score
    health_score = compute_health_score(
        last_commit_days=signals.get("last_commit_days_ago"),
        open_issues=signals.get("open_issues", 0),
        closed_issues=signals.get("closed_issues"),
        stars=signals.get("stars", 0),
        forks=signals.get("forks", 0),
        archived=signals.get("archived", False),
        repo_age_days=signals.get("repo_age_days"),
        contributor_count=signals.get("contributor_count"),
        has_ci=signals.get("has_ci", False),
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
    closed_issues: Optional[int] = None,
    stars: int = 0,
    forks: int = 0,
    archived: bool = False,
    repo_age_days: Optional[int] = None,
    contributor_count: Optional[int] = None,
    has_ci: Optional[bool] = None,
) -> int:
    """
    Compute a health score (0–100) from repo signals.

    Scoring breakdown:
      - Starts at 100, penalties applied, bonuses can recover up to 100.
      - Archived repos always score 0.
      - Commit recency is the heaviest single signal.
      - Issue resolution ratio replaces the raw open-issue count.
      - Star velocity (stars/year) replaces the hard star cutoff so new
        official repos are not unfairly penalised.
      - Forks, contributor count, and CI presence provide smaller bonuses/penalties.

    Pure Python — no LLM needed.
    """
    if archived:
        return 0

    score = 100

    # ------------------------------------------------------------------
    # 1. Commit recency  (max penalty: −50)
    # ------------------------------------------------------------------
    if last_commit_days is not None:
        if last_commit_days > 730:    # > 2 years
            score -= 50
        elif last_commit_days > 365:  # > 1 year
            score -= 25
        elif last_commit_days > 180:  # > 6 months
            score -= 10

    # ------------------------------------------------------------------
    # 2. Issue resolution ratio  (max penalty: −20)
    #    Replaces raw open_issues count — a repo that closes most issues
    #    is healthy even with a large backlog.
    # ------------------------------------------------------------------
    open_issues = open_issues or 0
    closed_issues_count = closed_issues if closed_issues is not None else 0
    total_issues = open_issues + closed_issues_count

    if total_issues > 0 and closed_issues is not None:
        close_rate = closed_issues_count / total_issues
        if close_rate < 0.3:
            score -= 20   # severely under-resolved backlog
        elif close_rate < 0.5:
            score -= 10   # moderate backlog pressure
        # ≥ 0.5 close rate → no penalty
    else:
        # No closed-issue data available — fall back to raw count
        if open_issues > 100:
            score -= 20
        elif open_issues > 50:
            score -= 10

    # ------------------------------------------------------------------
    # 3. Star velocity  (max penalty: −10)
    #    stars/year avoids penalising new official repos with few stars.
    # ------------------------------------------------------------------
    if stars is not None:
        if repo_age_days and repo_age_days > 30:
            stars_per_year = stars / max(1.0, repo_age_days / 365.0)
            if stars < 10 and stars_per_year < 10:
                score -= 10   # truly low-traction and new
            # else: recent repo or growing fast — no penalty
        elif stars < 10:
            score -= 10   # no age data, use absolute cutoff as before

    # ------------------------------------------------------------------
    # 4. Forks  (bonus: +5)
    #    High fork count signals community adoption.
    # ------------------------------------------------------------------
    if forks and forks > 50:
        score = min(100, score + 5)

    # ------------------------------------------------------------------
    # 5. Contributor count  (penalty/bonus: −10 / +5)
    #    Single-maintainer repos carry higher bus-factor risk.
    # ------------------------------------------------------------------
    if contributor_count is not None:
        if contributor_count >= 5:
            score = min(100, score + 5)   # community-maintained
        elif contributor_count == 1:
            score -= 10                   # single point of failure

    # ------------------------------------------------------------------
    # 6. CI/CD presence  (bonus: +5)
    #    A working CI pipeline is a proxy for code quality and maintenance.
    # ------------------------------------------------------------------
    if has_ci:
        score = min(100, score + 5)

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
    """
    Fallback: use GitHub REST API directly when Coral is unavailable.

    Makes up to 4 API calls:
      1. /repos/{owner}/{repo}             — core metadata
      2. /search/issues?q=...state=closed  — closed issue count (resolution ratio)
      3. /repos/{owner}/{repo}/contributors — contributor count (bus-factor)
      4. /repos/{owner}/{repo}/contents/.github/workflows — CI/CD presence
    """
    import requests
    import os

    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token and "dummy" not in token.lower() and "your_" not in token.lower():
        headers["Authorization"] = f"token {token}"

    signals = {
        "last_commit_days_ago": None,
        "open_issues": None,
        "closed_issues": None,
        "stars": None,
        "forks": None,
        "archived": None,
        "year": None,
        "repo_age_days": None,
        "contributor_count": None,
        "has_ci": None,
    }

    # --- Call 1: Core repo metadata ---
    try:
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

            created_at = data.get("created_at")
            if created_at:
                try:
                    created_date = datetime.fromisoformat(created_at.replace("Z", "+00:00")).date()
                    signals["repo_age_days"] = (date.today() - created_date).days
                except (ValueError, TypeError):
                    pass
    except Exception as e:
        logger.error(f"GitHub API (repo metadata) failed: {e}")

    # --- Call 2: Closed issue count (for resolution ratio) ---
    try:
        resp = requests.get(
            "https://api.github.com/search/issues",
            params={"q": f"repo:{owner}/{repo} type:issue state:closed", "per_page": 1},
            headers=headers, timeout=10,
        )
        if resp.status_code == 200:
            signals["closed_issues"] = resp.json().get("total_count", 0)
            logger.info(f"  Closed issues: {signals['closed_issues']}")
    except Exception as e:
        logger.warning(f"GitHub API (closed issues) failed: {e}")

    # --- Call 3: Contributor count (bus-factor) ---
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/contributors",
            params={"per_page": 30, "anon": "true"},
            headers=headers, timeout=10,
        )
        if resp.status_code == 200:
            contributors = resp.json()
            # If the page is full (30), there are likely more — flag as "30+"
            signals["contributor_count"] = len(contributors) if len(contributors) < 30 else 30
            logger.info(f"  Contributors: {signals['contributor_count']}")
    except Exception as e:
        logger.warning(f"GitHub API (contributors) failed: {e}")

    # --- Call 4: CI/CD presence ---
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/contents/.github/workflows",
            headers=headers, timeout=10,
        )
        signals["has_ci"] = resp.status_code == 200
        logger.info(f"  CI/CD present: {signals['has_ci']}")
    except Exception as e:
        logger.warning(f"GitHub API (CI check) failed: {e}")
        signals["has_ci"] = False

    health_score = compute_health_score(
        last_commit_days=signals.get("last_commit_days_ago"),
        open_issues=signals.get("open_issues", 0),
        closed_issues=signals.get("closed_issues"),
        stars=signals.get("stars", 0),
        forks=signals.get("forks", 0),
        archived=signals.get("archived", False),
        repo_age_days=signals.get("repo_age_days"),
        contributor_count=signals.get("contributor_count"),
        has_ci=signals.get("has_ci", False),
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
