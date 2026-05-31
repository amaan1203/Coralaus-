"""
Component 2 — Implementation Search (Semantic Scholar + GitHub)

Searches Semantic Scholar for known papers, then finds implementations
via GitHub repository search (arXiv ID-based).

Input:  Paper JSON (title, arxiv_id from Component 1)
Output: Best repo URL + metadata, or None if no implementation found
"""

import logging
from typing import Optional
from agents.pwc_mcp_client import get_s2_client

logger = logging.getLogger(__name__)


def _is_same_paper(title1: str, title2: str) -> bool:
    if not title1 or not title2:
        return False
    t1 = "".join(c for c in title1.lower() if c.isalnum())
    t2 = "".join(c for c in title2.lower() if c.isalnum())
    return t1 == t2


def search_implementation(paper_json: dict) -> dict:
    """
    Search Semantic Scholar + GitHub for implementations of the given paper.
    Accumulates candidates using all search strategies to find at least 10 repositories,
    including arXiv ID, title, keywords, direct GitHub search, and GitHub Code Search.

    Args:
        paper_json: Parsed paper JSON from Component 1

    Returns:
        Dict with:
            - found (bool): Whether any implementation was found
            - repo_url (str): Best guess GitHub repo URL (highest priority fallback)
            - stars (int): Repo star count
            - framework (str): ML framework used
            - is_official (bool): Official implementation flag
            - paper_id (str): Semantic Scholar / arXiv paper ID
            - search_method (str): How the primary repo was found
            - candidates (list): List of unique candidates found across all strategies
    """
    title = paper_json.get("title", "")
    arxiv_id = paper_json.get("arxiv_id", "")
    if arxiv_id:
        import re
        arxiv_id = re.sub(r'v\d+$', '', arxiv_id)
    pwc = get_s2_client()

    candidates = []

    def add_candidate(url, stars=0, framework=None, is_official=False, method=None, is_official_context=False, pre_boost_score=0.0):
        if not url:
            return
        # Normalize URL: remove trailing slashes, periods, and whitespace
        url = url.rstrip("/").rstrip(".").strip()
        norm_url = url.rstrip("/").lower()
        if norm_url.endswith(".git"):
            norm_url = norm_url[:-4]
        for c in candidates:
            c_norm = c["repo_url"].rstrip("/").lower()
            if c_norm.endswith(".git"):
                c_norm = c_norm[:-4]
            if c_norm == norm_url:
                # Merge fields if the new ones have more information
                if stars and not c["stars"]:
                    c["stars"] = stars
                if is_official and not c["is_official"]:
                    c["is_official"] = True
                if is_official_context and not c.get("is_official_context"):
                    c["is_official_context"] = True
                if pre_boost_score and pre_boost_score > c.get("pre_boost_score", 0.0):
                    c["pre_boost_score"] = pre_boost_score
                return
        candidates.append({
            "repo_url": url,
            "stars": stars or 0,
            "framework": framework or _detect_framework(url),
            "is_official": is_official,
            "is_official_context": is_official_context,
            "search_method": method,
            "pre_boost_score": pre_boost_score,
        })

    paper_id = None

    # Strategy 0: Direct Hugging Face Papers API lookup
    if arxiv_id:
        logger.info(f"Searching Hugging Face Papers API directly for arXiv ID: {arxiv_id}")
        if hasattr(pwc, "_get_official_repo_from_hf"):
            hf_repo = pwc._get_official_repo_from_hf(arxiv_id)
            if hf_repo:
                add_candidate(
                    url=hf_repo.get("url"),
                    stars=hf_repo.get("stars", 0),
                    framework=hf_repo.get("framework"),
                    is_official=True,
                    is_official_context=True,
                    method="hf_direct",
                    pre_boost_score=1.0
                )

    # Strategy 1: Search Semantic Scholar by arXiv ID (most precise)
    if arxiv_id:
        logger.info(f"Searching Semantic Scholar by arXiv ID: {arxiv_id}")
        paper = pwc.get_paper_by_arxiv_id(arxiv_id)
        if paper:
            paper_id = paper.get("id")
            repos = pwc.get_paper_repositories(paper_id) if paper_id else []
            for r in repos:
                add_candidate(
                    url=r.get("url"),
                    stars=r.get("stars", 0),
                    framework=r.get("framework"),
                    is_official=r.get("is_official", False),
                    is_official_context=r.get("is_official", False),
                    method="arxiv_id",
                    pre_boost_score=1.0 if r.get("is_official", False) else 0.0
                )

    # Strategy 2: Search Semantic Scholar by title
    if title:
        logger.info(f"Searching Semantic Scholar by title: '{title[:60]}'")
        papers = pwc.search_papers(title, items_per_page=5)
        for paper in papers:
            p_id = paper.get("id")
            if p_id:
                if not paper_id:
                    paper_id = p_id
                repos = pwc.get_paper_repositories(p_id)
                same_paper = _is_same_paper(title, paper.get("title", ""))
                for r in repos:
                    is_off = r.get("is_official", False) if same_paper else False
                    add_candidate(
                        url=r.get("url"),
                        stars=r.get("stars", 0),
                        framework=r.get("framework"),
                        is_official=is_off,
                        is_official_context=is_off,
                        method="title",
                        pre_boost_score=1.0 if is_off else 0.0
                    )

    # Strategy 3: Search Semantic Scholar with keywords from title
    if title:
        keywords = _extract_search_terms(title)
        if keywords:
            logger.info(f"Searching Semantic Scholar by keywords: '{keywords}'")
            papers = pwc.search_papers(keywords, items_per_page=3)
            for paper in papers:
                p_id = paper.get("id")
                if p_id:
                    repos = pwc.get_paper_repositories(p_id)
                    same_paper = _is_same_paper(title, paper.get("title", ""))
                    for r in repos:
                        is_off = r.get("is_official", False) if same_paper else False
                        add_candidate(
                            url=r.get("url"),
                            stars=r.get("stars", 0),
                            framework=r.get("framework"),
                            is_official=is_off,
                            is_official_context=is_off,
                            method="keywords",
                            pre_boost_score=1.0 if is_off else 0.0
                        )

    # Strategy 4: Check if abstract/paper body mentions a GitHub URL directly
    abstract = paper_json.get("abstract", "") or ""
    full_text = paper_json.get("full_text", "") or ""
    combined_text = f"{abstract}\n{full_text}"
    
    import re
    combined_cleaned = re.sub(r'github\.com/\s+', 'github.com/', combined_text)
    combined_cleaned = re.sub(r'/\s+', '/', combined_cleaned)
    
    # Extract all unique GitHub URLs
    all_urls = []
    seen_norm = set()
    for match in re.finditer(r'https?://github\.com/([\w.-]+/[\w.-]+)(?=[\s\)\]\},;:]|$)', combined_cleaned):
        url = f"https://github.com/{match.group(1)}"
        url_norm = url.rstrip("/").lower()
        if url_norm not in seen_norm:
            seen_norm.add(url_norm)
            all_urls.append(url)
            
    if all_urls:
        if len(all_urls) == 1:
            # If ONLY a single GitHub URL is present in the paper, it is automatically treated as official (1.00)
            mentioned_url = all_urls[0]
            is_official_context = True
            logger.info(f"Found exactly ONE GitHub URL in paper: {mentioned_url}. Treating as official repo (1.00 boost).")
        else:
            # Fall back to checking surrounding context for keywords
            res = _extract_github_url_with_context(abstract) or _extract_github_url_with_context(full_text)
            mentioned_url, is_official_context = res if res else (all_urls[0], False)
            
        add_candidate(
            url=mentioned_url,
            stars=0,
            framework=None,
            is_official=True,
            method="paper_body_url",
            is_official_context=is_official_context,
            pre_boost_score=1.0
        )
        # Add special context flag to the candidate we just added/updated
        for c in candidates:
            if c["repo_url"].rstrip("/").lower() == mentioned_url.rstrip("/").lower():
                c["is_official_context"] = is_official_context

    # Strategy 5: Direct GitHub search by arXiv ID
    if arxiv_id:
        logger.info(f"Searching GitHub directly for arXiv ID: {arxiv_id}")
        repos = _search_github_repositories(f'"{arxiv_id}" in:name,description,readme')
        for r in repos:
            add_candidate(
                url=r.get("html_url") or f"https://github.com/{r.get('full_name')}",
                stars=r.get("stargazers_count", r.get("stars", 0)) or 0,
                framework=_detect_framework(r.get("description", "")),
                is_official=r.get("_is_official", False),
                is_official_context=r.get("_is_official", False),
                method="direct_github_arxiv_id",
                pre_boost_score=0.85 if r.get("_is_official", False) else 0.0
            )

    # Strategy 6: Direct GitHub search by paper title keywords
    if title:
        queries = _get_expanded_search_queries(title)
        for q in queries:
            logger.info(f"Searching GitHub directly for title keywords: {q}")
            repos = _search_github_repositories(f"{q} in:name,description,readme")
            for r in repos:
                add_candidate(
                    url=r.get("html_url") or f"https://github.com/{r.get('full_name')}",
                    stars=r.get("stargazers_count", r.get("stars", 0)) or 0,
                    framework=_detect_framework(r.get("description", "")),
                    is_official=r.get("_is_official", False),
                    is_official_context=r.get("_is_official", False),
                    method="direct_github_title",
                    pre_boost_score=0.85 if r.get("_is_official", False) else 0.0
                )

    # Strategy 7: Direct GitHub Code Search on .md files (NEW)
    if arxiv_id or title:
        logger.info("Searching GitHub Code (.md files) for arXiv ID / Title")
        queries = _get_expanded_search_queries(title) if title else []
        repos = _search_github_code(arxiv_id, queries)
        for r in repos:
            add_candidate(
                url=r.get("html_url") or f"https://github.com/{r.get('full_name')}",
                stars=r.get("stargazers_count", r.get("stars", 0)) or 0,
                framework=_detect_framework(r.get("description", "")),
                is_official=False,
                method="direct_github_code",
                pre_boost_score=0.0
            )

    # Prioritize the candidate list to choose a smart fallback best URL
    def _rank_score(c):
        score = c.get("stars", 0)
        url = c.get("repo_url", "").lower()
        if c.get("is_official"):
            score += int((c.get("pre_boost_score", 0.0) or 0.0) * 100000)

        # Penalties
        penalty_words = ["homework", "assignment", "coursework", "cs4", "cs2",
                         "course", "class", "tutorial", "my-", "fork"]
        for word in penalty_words:
            if word in url:
                score -= 50000

        # Boost known ML orgs
        known_orgs = ["microsoft", "google", "meta", "facebook", "openai",
                      "huggingface", "pytorch", "tensorflow", "deepmind",
                      "nvidia", "apple", "amazon", "aws"]
        owner = url.split("github.com/")[-1].split("/")[0] if "github.com/" in url else ""
        if owner in known_orgs:
            score += 20000

        return score

    candidates.sort(key=_rank_score, reverse=True)

    result = {
        "found": len(candidates) > 0,
        "repo_url": None,
        "stars": 0,
        "framework": None,
        "is_official": False,
        "paper_id": paper_id,
        "search_method": None,
        "candidates": candidates,
    }

    if candidates:
        best = candidates[0]
        result.update({
            "repo_url": best["repo_url"],
            "stars": best["stars"],
            "framework": best["framework"],
            "is_official": best["is_official"],
            "search_method": best["search_method"],
        })
        logger.info(f"Total {len(candidates)} candidates accumulated. Default best candidate: {result['repo_url']} via {result['search_method']}")
    else:
        logger.info("No candidates found across Semantic Scholar or GitHub searches.")

    return result


def _fetch_repo_stars(owner: str, repo: str) -> int:
    """
    Fetch the actual star count for a GitHub repository from API.
    Returns 0 if fetch fails.
    """
    import requests
    import os
    
    # Normalize: strip trailing slashes, periods, and whitespace
    owner = owner.rstrip("./").strip()
    repo = repo.rstrip("./").strip()
    
    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token and "dummy" not in token.lower() and "your_" not in token.lower():
        headers["Authorization"] = f"token {token}"
    
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}",
            headers=headers, timeout=5
        )
        if resp.status_code == 200:
            return resp.json().get("stargazers_count", 0)
    except Exception as e:
        logger.debug(f"Failed to fetch stars for {owner}/{repo}: {e}")
    
    return 0


def _search_github_code(arxiv_id: str, title_queries: list) -> list:
    """
    Search GitHub code specifically in .md files for the arXiv ID and/or title keywords.
    Returns both the root repository and the specific subdirectory candidate (if matched in a subfolder).
    Fetches actual star counts from GitHub API for each repo.
    """
    import os
    from agents.coral_utils import get_coral_client

    coral = get_coral_client()
    queries = []
    
    if arxiv_id:
        queries.append(f'"{arxiv_id}" extension:md')
        queries.append(f'"{arxiv_id}" archived:true extension:md')
    if title_queries:
        for tq in title_queries:
            queries.append(f'{tq} extension:md')
            queries.append(f'{tq} archived:true extension:md')

    mapped = []
    seen_urls = set()

    def add_code_candidate(repo_full, path):
        if not repo_full:
            return
        
        # Normalize: remove trailing slashes, periods, and whitespace
        repo_full = repo_full.rstrip("/").rstrip(".").strip()
        
        main_url = f"https://github.com/{repo_full}"
        if main_url.lower() not in seen_urls:
            seen_urls.add(main_url.lower())
            # Fetch actual star count from GitHub API
            parts = repo_full.split("/")
            if len(parts) >= 2:
                owner = parts[0]
                repo = parts[1]
                stars = _fetch_repo_stars(owner, repo)
            else:
                stars = 0
            mapped.append({
                "full_name": repo_full,
                "html_url": main_url,
                "stargazers_count": stars,
                "description": f"Found via Code Search: {path}"
            })

        # Subdirectory logic: if path has subfolders, add a specific subdirectory URL
        if "/" in path:
            subdir_path = "/".join(path.split("/")[:-1])
            if subdir_path:
                subdir_url = f"https://github.com/{repo_full}/tree/master/{subdir_path}"
                if subdir_url.lower() not in seen_urls:
                    seen_urls.add(subdir_url.lower())
                    # Use same stars count for subdirectory (they belong to same repo)
                    parts = repo_full.split("/")
                    if len(parts) >= 2:
                        owner = parts[0]
                        repo = parts[1]
                        stars = _fetch_repo_stars(owner, repo)
                    else:
                        stars = 0
                    mapped.append({
                        "full_name": f"{repo_full}/{subdir_path}",
                        "html_url": subdir_url,
                        "stargazers_count": stars,
                        "description": f"Found via Code Search Subdir: {path}"
                    })

    for query in queries:
        import time
        time.sleep(1.5) # sleep to respect rate limit
        # Coral code search
        if coral.available:
            try:
                res = coral.sql(f"""
                    SELECT repository_full_name, path
                    FROM github.search_code(q => '{query}')
                    LIMIT 8
                """)
                if res and "results" in res:
                    for row in res["results"]:
                        add_code_candidate(row.get("repository_full_name"), row.get("path"))
            except Exception as e:
                logger.debug(f"Coral code search failed for query '{query}': {e}")

        # Fallback to GitHub REST API
        import requests
        headers = {}
        token = os.environ.get("GITHUB_TOKEN")
        if token and "dummy" not in token.lower() and "your_" not in token.lower():
            headers["Authorization"] = f"token {token}"

        for attempt in range(3):
            try:
                resp = requests.get(
                    "https://api.github.com/search/code",
                    params={"q": query, "per_page": 8},
                    headers=headers, timeout=10
                )
                if resp.status_code == 200:
                    items = resp.json().get("items", [])
                    for item in items:
                        repo = item.get("repository", {})
                        add_code_candidate(repo.get("full_name"), item.get("path"))
                    break
                elif resp.status_code in (403, 429):
                    wait_time = 5 * (attempt + 1)
                    logger.warning(f"GitHub Code Search API rate-limited ({resp.status_code}). Waiting {wait_time}s before retry {attempt + 1}/3...")
                    time.sleep(wait_time)
                else:
                    logger.debug(f"GitHub code search API failed for query '{query}': status {resp.status_code}")
                    break
            except Exception as e:
                logger.debug(f"GitHub code search fallback failed for query '{query}' (attempt {attempt + 1}): {e}")
                time.sleep(2)

    return mapped


def _search_github_repositories(query: str) -> list:
    """Search GitHub for repositories using Coral SQL (or API fallback)."""
    import os
    import time
    from agents.coral_utils import get_coral_client

    time.sleep(1.0) # sleep to avoid secondary rate limit
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

    for attempt in range(3):
        try:
            resp = requests.get(
                "https://api.github.com/search/repositories",
                params={"q": query, "sort": "stars", "order": "desc", "per_page": 10},
                headers=headers, timeout=10
            )
            if resp.status_code == 200:
                return resp.json().get("items", [])
            elif resp.status_code in (403, 429):
                wait_time = 5 * (attempt + 1)
                logger.warning(f"GitHub Search API rate-limited ({resp.status_code}). Waiting {wait_time}s before retry {attempt + 1}/3...")
                time.sleep(wait_time)
            else:
                logger.error(f"GitHub search API failed with status {resp.status_code}: {resp.text[:200]}")
                break
        except Exception as e:
            logger.error(f"GitHub search fallback attempt {attempt + 1} failed: {e}")
            time.sleep(2)

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
    cleaned = re.sub(r'github\.com/\s+', 'github.com/', text)
    cleaned = re.sub(r'/\s+', '/', cleaned)
    match = re.search(r'https?://github\.com/([\w.-]+/[\w.-]+)(?=[\s\)\]\},;:]|$)', cleaned)
    if match:
        url = f"https://github.com/{match.group(1)}"
        return url.rstrip(".,;:()[]{}")
    return None


def _extract_github_url_with_context(text: str) -> Optional[tuple]:
    """
    Extract a GitHub repo URL from text and check if the surrounding context (e.g. 150 characters before/after)
    contains words like 'official implementation', 'official code', 'our implementation', etc.
    Returns: (url, is_official_context_matched) or None
    """
    import re
    if not text:
        return None
    
    cleaned = re.sub(r'github\.com/\s+', 'github.com/', text)
    cleaned = re.sub(r'/\s+', '/', cleaned)
    
    matches = list(re.finditer(r'https?://github\.com/([\w.-]+/[\w.-]+)(?=[\s\)\]\},;:]|$)', cleaned))
    if not matches:
        return None
        
    for match in matches:
        url = f"https://github.com/{match.group(1)}".rstrip(".,;:()[]{}")
        start = max(0, match.start() - 150)
        end = min(len(cleaned), match.end() + 150)
        context = cleaned[start:end].lower()
        
        official_keywords = [
            # Direct "official" language
            "official",
            "official implementation",
            "official code",
            "official repository",
            "official repo",
            "official github",
            "official source code",
            "official pytorch implementation",
            "official tensorflow implementation",
            "official jax implementation",
            "project page",
            "project code",
            "project repository",
            "project repo",
            "is available",
            "source code is available",
            "implementation is available",
            "available at github",
            # Ownership / authorship claims
            "our code",
            "our implementation",
            "our repository",
            "our repo",
            "our github",
            "our codebase",
            "we release",
            "we open-source",
            "we provide",
            "we publicly release",
            "we make the code available",
            "we make our code available",
            "we release the source code",
            "we release our implementation",
            "we provide the implementation",
            "we provide code",
            "the authors release",
            "authors release",
            "released by the authors",
            "provided by the authors",
            "created by the authors",
            "the author"

            # Availability statements (very common in papers)
            "code is available at",
            "source code is available at",
            "implementation is available at",
            "available at github",
            "available on github",
            "can be found at",
            "is publicly available",
            "publicly available at",
            "repository is available at",
            "repo is available at",
            "github repository",
            "github repo",

            # Canonical / primary wording
            "original implementation",
            "original codebase",
            "canonical implementation",
            "reference implementation",
            "main repository",
            "primary repository",
            "maintained by the authors",

            # Release / publication phrasing
            "accompanying code",
            "accompanying implementation",
            "supplementary code",
            "project page",
            "project repository",
            "released with the paper",
            "released alongside the paper",
            "paper implementation",
            "implementation accompanying this paper",

            # Benchmark / framework style wording
            "this repository contains the official implementation",
            "this repo contains the official implementation",
            "this repository provides",
            "this repo provides",
            "implementation of the paper",
            "implementation for the paper",

            # Organization / affiliation clues
            "maintained by",
            "developed by",
            "created by the authors",
            "from the authors",
            "author's implementation",
        ]
        is_official = any(kw in context for kw in official_keywords)
        return url, is_official
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
    return " ".join(words)

def _get_expanded_search_queries(title: str) -> list[str]:
    import re
    queries = []
    if not title:
        return queries
    title_clean = title.strip()
    
    # Acronyms
    acronyms = re.findall(r"\b[A-Z0-9]{2,8}\b", title)
    for acr in acronyms:
        queries.append(acr)
        
    # Part before colon
    if ":" in title:
        prefix = title.split(":")[0].strip()
        prefix_clean = re.sub(r'[^\w\s-]', '', prefix).strip()
        if prefix_clean:
            queries.append(prefix_clean)
        prefix_words = prefix_clean.split()
        if len(prefix_words) > 1:
            queries.append("".join(w for w in prefix_words if w.isalnum()))
            
    # First words cleaned of punctuation
    words = [re.sub(r'[^\w\s-]', '', w).strip() for w in title_clean.split()]
    words = [w for w in words if w]
    if words:
        queries.append(words[0])
        if len(words) > 1:
            queries.append(" ".join(words[:2]))
            queries.append("".join(words[:2]))
        if len(words) > 2:
            queries.append(" ".join(words[:3]))
            queries.append("".join(words[:3]))
            
    # Standard keywords
    keywords = _extract_search_terms(title)
    if keywords:
        queries.append(keywords)
        
    seen = set()
    unique = []
    for q in queries:
        q_clean = q.lower().strip()
        if q_clean and q_clean not in seen and len(q_clean) >= 3:
            seen.add(q_clean)
            unique.append(q)
    return unique


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
