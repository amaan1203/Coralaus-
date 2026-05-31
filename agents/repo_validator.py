"""
Component 2.5 — Repository Validator (Component 2.5)

Validates discovered repositories against the research paper metadata to verify
whether the repository actually implements the paper it claims to represent.

Heuristics:
1. Semantic Similarity (Weight: 0.35): Cosine similarity between abstract and README.
2. Concept Matching (Weight: 0.40): Fuzzy matching of paper keywords against README.
3. Dependency Matching (Weight: 0.25): Required frameworks checked against requirements text.
"""

import os
import re
import json
import logging
import threading
from typing import Optional, List, Tuple, Dict
from agents.coral_utils import get_coral_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hugging Face Authentication & Log Silencing Setup
# Resolves rate-limit warnings and hides noisy 404 HEAD checks
# ---------------------------------------------------------------------------
hf_token = os.environ.get("HUGGING_FACE_ACCESS_TOKEN", "")
if hf_token:
    os.environ["HF_TOKEN"] = hf_token

# Silence noisy optional dependency 404 loggers and HF rate-limit warnings
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub.utils._http").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)
logging.getLogger("requests").setLevel(logging.ERROR)

import warnings
warnings.filterwarnings("ignore", message=".*unauthenticated requests to the HF Hub.*")

# Try to import rapidfuzz for fuzzy matching
try:
    from rapidfuzz import fuzz
except ImportError:
    logger.warning("rapidfuzz not installed. Falling back to simple substring matching for keywords.")
    fuzz = None

# Try to import sentence-transformers for semantic similarity
try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("sentence-transformers not installed. Semantic Similarity will use Jaccard fallback.")
    SENTENCE_TRANSFORMERS_AVAILABLE = False


# ---------------------------------------------------------------------------
# Global Session Caches & Singletons
# Prevents duplicate requests, blind 404s, and duplicate model reloads
# ---------------------------------------------------------------------------
# Map: (owner, repo, path) -> file_content (README/requirements/setup.py)
_fetched_content_cache: Dict[Tuple[str, str, str], str] = {}

# Map: (owner, repo, subdir) -> set of lowercase filenames present in the directory
_directory_contents_cache: Dict[Tuple[str, str, Optional[str]], set] = {}

# Map: (owner, repo) -> list of all file paths in the repo tree
_tree_contents_cache: Dict[Tuple[str, str], list] = {}

# Reusable SentenceTransformer singleton
_embedding_model = None
_embedding_model_lock = threading.Lock()


def _get_embedding_model() -> Optional['SentenceTransformer']:
    """Load and return the reusable SentenceTransformer model singleton."""
    global _embedding_model
    if _embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
        with _embedding_model_lock:
            if _embedding_model is None:
                try:
                    logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2' (reusable singleton)...")
                    _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
                except Exception as e:
                    logger.debug(f"Failed to load SentenceTransformer model singleton: {e}")
    return _embedding_model


def _escape_sql_literal(value: str) -> str:
    return value.replace("'", "''")


def _get_repo_contributors(owner: str, repo: str) -> list:
    """
    Get list of contributors/commit authors from Coral or GitHub API.
    """
    names = set()
    coral = get_coral_client()
    if coral.available:
        try:
            res = coral.sql(f"""
                SELECT commit__author__name as author_name, commit__committer__name as committer_name
                FROM github.commits
                WHERE owner = '{_escape_sql_literal(owner)}' AND repo = '{_escape_sql_literal(repo)}'
                LIMIT 50
            """)
            if res and "results" in res:
                for row in res["results"]:
                    if row.get("author_name"):
                        names.add(row["author_name"].lower())
                    if row.get("committer_name"):
                        names.add(row["committer_name"].lower())
        except Exception as e:
            logger.debug(f"Coral contributor fetch failed: {e}")
            
    import requests
    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token and "dummy" not in token.lower() and "your_" not in token.lower():
        headers["Authorization"] = f"token {token}"
        
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/contributors",
            headers=headers, timeout=5
        )
        if resp.status_code == 200:
            for c in resp.json():
                if c.get("login"):
                    names.add(c["login"].lower())
    except Exception as e:
        logger.debug(f"GitHub API contributors fetch failed: {e}")
        
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/commits",
            headers=headers, timeout=5
        )
        if resp.status_code == 200:
            for item in resp.json():
                commit = item.get("commit", {})
                author = commit.get("author", {})
                committer = commit.get("committer", {})
                if author.get("name"):
                    names.add(author["name"].lower())
                if committer.get("name"):
                    names.add(committer["name"].lower())
    except Exception as e:
        logger.debug(f"GitHub API commits fetch failed: {e}")
        
    return list(names)


def _match_authors_to_contributors(paper_authors: list, contributors: list) -> Tuple[bool, str]:
    """
    Match paper authors to repository contributors/committers.
    Returns (is_matched, matched_author_name).
    """
    if not paper_authors or not contributors:
        return False, ""
        
    for author in paper_authors:
        if isinstance(author, dict):
            name_val = author.get("name", "").strip().lower()
            if name_val:
                full = name_val
                parts = full.split()
                first = parts[0] if parts else ""
                last = parts[-1] if len(parts) > 1 else ""
            else:
                first = author.get("first", "").strip().lower()
                last = author.get("last", "").strip().lower()
                full = f"{first} {last}".strip()
        elif isinstance(author, str):
            full = author.strip().lower()
            parts = full.split()
            first = parts[0] if parts else ""
            last = parts[-1] if len(parts) > 1 else ""
        else:
            continue
            
        if not full:
            continue
            
        for c in contributors:
            c_clean = c.strip().lower()
            # 1. Exact match of full name, or full name is a substring of contributor name
            if full == c_clean or full in c_clean:
                return True, full.title()
            # 2. Match both first and last name as substrings of contributor name
            if first and last:
                if first in c_clean and last in c_clean:
                    return True, full.title()
                # First initial + last name (e.g. yburda)
                if len(first) >= 1 and c_clean == (first[0] + last):
                    return True, full.title()
                # First name + last initial (e.g. aletheap)
                if len(last) >= 1 and c_clean == (first + last[0]):
                    return True, full.title()
                # Last name starts with first initial (e.g. yburda)
                if len(first) >= 1 and last in c_clean and c_clean.startswith(first[0]):
                    return True, full.title()
                # Last + first (e.g. caoyunkang)
                if c_clean == (last + first):
                    return True, full.title()
                
    return False, ""


def _repo_name_matches_paper(repo_url: str, paper_json: dict) -> bool:
    owner, repo, subdir = _parse_github_url(repo_url)
    if not repo:
        return False
    
    # Check if owner matches an author
    paper_authors = paper_json.get("authors", [])
    author_names = []
    for author in paper_authors:
        if isinstance(author, dict):
            first = author.get("first", "").strip().lower()
            last = author.get("last", "").strip().lower()
            if first: author_names.append(first)
            if last: author_names.append(last)
            name_val = author.get("name", "").strip().lower()
            if name_val:
                author_names.extend(name_val.split())
        elif isinstance(author, str):
            author_names.extend(author.strip().lower().split())
            
    owner_lower = owner.lower() if owner else ""
    if owner_lower in author_names:
        return True
        
    # Check if repo name or subdir matches paper title words or acronyms
    repo_name_lower = repo.lower()
    subdir_lower = subdir.lower() if subdir else ""
    
    title = paper_json.get("title", "") or ""
    title_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', title).lower()
    title_words = [w for w in title_clean.split() if len(w) > 2]
    
    # Check acronyms of the title
    acronyms = re.findall(r"\b[A-Z0-9]{2,8}\b", title)
    acronyms_lower = [acr.lower() for acr in acronyms]
    
    # Generate main title acronym (first letter of capitalized words)
    words_orig = title.split()
    cap_letters = "".join(w[0] for w in words_orig if w and w[0].isupper())
    if len(cap_letters) >= 2:
        acronyms_lower.append(cap_letters.lower())
        
    # Check if any title word or acronym is in repo name or subdir name
    for word in title_words:
        if word in repo_name_lower or (subdir_lower and word in subdir_lower):
            return True
            
    for acr in acronyms_lower:
        if acr in repo_name_lower or (subdir_lower and acr in subdir_lower):
            return True
            
    for word in title_words:
        if len(word) >= 3 and len(repo_name_lower) >= 3:
            if word.startswith(repo_name_lower) or repo_name_lower.startswith(word):
                return True
                
    return False


def _determine_official_status(c_metadata: Optional[dict], repo_url: str, paper_json: dict, contributors: list) -> Tuple[bool, List[str]]:
    """
    Unifies officiality validation under a single logical check.
    Returns (is_official, list_of_reasons).
    Only returns True for strong officiality signals:
    - Metadata explicit flag (is_official)
    - Contextual direct URL extraction (is_official_context)
    - Author match on contributors (validated by name match)
    """
    reasons = []
    
    paper_authors = paper_json.get("authors", [])
    matched, matched_author = _match_authors_to_contributors(paper_authors, contributors)
    has_name_alignment = _repo_name_matches_paper(repo_url, paper_json)
    
    # 1. Check PWC/S2 metadata
    if c_metadata and c_metadata.get("is_official"):
        if has_name_alignment or matched:
            reasons.append("Flagged as official by PapersWithCode/Semantic Scholar metadata.")
        else:
            logger.info(f"Ignored metadata official flag for {repo_url} because it lacks name alignment or contributor match with paper.")
        
    # 2. Check Contextual URL extraction
    if c_metadata and c_metadata.get("is_official_context"):
        if has_name_alignment or matched:
            reasons.append("Identified as canonical/official repository from paper text context.")
        else:
            logger.info(f"Ignored contextual official flag for {repo_url} because it lacks name alignment or contributor match with paper.")
        
    # 3. Check Author-Contributor match
    if matched:
        if has_name_alignment:
            reasons.append(f"Paper author '{matched_author}' matches a repository contributor.")
        else:
            logger.info(f"Ignored contributor match for {repo_url} because repository name/owner does not match paper title/acronym.")
        
    is_off = len(reasons) > 0
    return is_off, reasons


def _get_recursive_files(owner: str, repo: str) -> list:
    """
    List all file paths recursively under a repository.
    Cached globally to prevent redundant tree network requests.
    """
    key = (owner.lower(), repo.lower())
    if key in _tree_contents_cache:
        return _tree_contents_cache[key]
    
    paths = []
    coral = get_coral_client()
    if coral.available:
        try:
            res = coral.sql(f"""
                SELECT path
                FROM github.trees
                WHERE owner = '{_escape_sql_literal(owner)}' AND repo = '{_escape_sql_literal(repo)}' AND tree_sha = 'HEAD' AND recursive = 'true' AND type = 'blob'
            """)
            if res and "results" in res:
                paths = [row.get("path", "") for row in res["results"] if row.get("path")]
        except Exception as e:
            logger.debug(f"Coral tree search failed for {owner}/{repo}: {e}")
            
    if not paths:
        import requests
        headers = {}
        token = os.environ.get("GITHUB_TOKEN")
        if token and "dummy" not in token.lower() and "your_" not in token.lower():
            headers["Authorization"] = f"token {token}"
        try:
            resp = requests.get(
                f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1",
                headers=headers, timeout=10
            )
            if resp.status_code == 200:
                tree = resp.json().get("tree", [])
                paths = [item.get("path", "") for item in tree if item.get("type") == "blob"]
        except Exception as e:
            logger.debug(f"GitHub REST tree query failed for {owner}/{repo}: {e}")
            
    _tree_contents_cache[key] = paths
    return paths


def _batch_get_recursive_files(repos: list[tuple[str, str]]) -> dict:
    """
    Fetch recursive file trees for multiple repositories in one Coral query.
    """
    if not repos:
        return {}

    unique_repos = []
    missing = []
    for owner, repo in repos:
        key = (owner.lower(), repo.lower())
        if key not in _tree_contents_cache:
            missing.append((owner, repo))
        if key not in unique_repos:
            unique_repos.append(key)

    if missing:
        coral = get_coral_client()
        if coral.available:
            from collections import defaultdict
            by_owner = defaultdict(list)
            for owner, repo in missing:
                by_owner[owner.lower()].append(repo)
            
            for owner_lower, repo_list in by_owner.items():
                repo_clauses = " OR ".join(f"repo = '{_escape_sql_literal(r)}'" for r in repo_list)
                try:
                    res = coral.sql(f"""
                        SELECT owner, repo, path
                        FROM github.trees
                        WHERE owner = '{_escape_sql_literal(owner_lower)}' AND tree_sha = 'HEAD' AND recursive = 'true' AND type = 'blob'
                          AND ({repo_clauses})
                    """)
                    if res and "results" in res:
                        for row in res["results"]:
                            owner_val = row.get("owner", "")
                            repo_val = row.get("repo", "")
                            path_val = row.get("path", "")
                            if owner_val and repo_val and path_val:
                                key = (owner_val.lower(), repo_val.lower())
                                _tree_contents_cache.setdefault(key, []).append(path_val)
                except Exception as e:
                    logger.debug(f"Batch Coral tree search failed for owner {owner_lower}: {e}")

    results = {}
    for owner, repo in unique_repos:
        results[(owner.lower(), repo.lower())] = _tree_contents_cache.get((owner.lower(), repo.lower()), [])
    return results


def _pre_filter_candidates(candidates: list, paper_json: Optional[dict] = None) -> Tuple[list, list]:
    """
    Skeptically prune obvious non-implementations based strictly on local metadata
    (URL owner, URL name, description) before any network fetches are run.
    Prunes curated awesome lists and obvious university homework/lab coursework assignments.
    Also prunes repositories with fewer than 5 stars to focus on more mature projects.
    Returns (kept_candidates, pruned_candidates)
    """
    if not candidates:
        return [], []

    kept = []
    pruned = []
    for c in candidates:
        c_copy = dict(c)
        url = c_copy.get("repo_url", "").lower()
        clean_url = c_copy.get("repo_url", "").rstrip("/").rstrip(".").strip()
        desc = c_copy.get("description", "") or ""
        desc_lower = desc.lower()

        # Don't prune official candidates
        if c_copy.get("is_official") or c_copy.get("is_official_context"):
            kept.append(c_copy)
            continue

        # 1. Prune awesome lists and curated collections
        if "awesome-" in url or "awesome list" in desc_lower or "curated list" in desc_lower:
            logger.info(f"Pre-filtering: Skeptically pruned awesome list repository: {clean_url}")
            c_copy["pruned"] = True
            c_copy["prune_reason"] = "Awesome list / curated collection"
            pruned.append(c_copy)
            continue

        # 2. Prune obvious university homework/coursework assignments
        course_patterns = [
            r"cs\d{3}.*assignment", r"assignment.*cs\d{3}",
            r"homework.*cs\d{3}", r"coursework", r"university assignment",
            r"homework assignment", r"lab assignment", r"class assignment", r"students", r"curriculum", r"syllabus", r"from-scratch",
            r"cs-\d{3}", r"cs\d{3}-"
        ]
        is_course = False
        for pat in course_patterns:
            if re.search(pat, url) or re.search(pat, desc_lower):
                is_course = True
                break

        if is_course:
            logger.info(f"Pre-filtering: Skeptically pruned obvious coursework/homework assignment: {clean_url}")
            c_copy["pruned"] = True
            c_copy["prune_reason"] = "University coursework/assignment"
            pruned.append(c_copy)
            continue

        # 3. Prune repositories with fewer than 5 stars, checking main repository stars
        owner, repo_name, _ = _parse_github_url(clean_url)
        if owner and repo_name:
            stars = c_copy.get("stars")
            if stars is None or stars == 0:
                try:
                    from agents.pwc_search import _fetch_repo_stars
                    stars = _fetch_repo_stars(owner, repo_name)
                except Exception as e:
                    logger.debug(f"Could not fetch stars in pre-filtering for {owner}/{repo_name}: {e}")
                    stars = 0
                c_copy["stars"] = stars

            if stars < 5:
                # If we have paper_json, check if the low-star repo is actually official via contributor/name match
                is_off = False
                if paper_json:
                    try:
                        contributors = _get_repo_contributors(owner, repo_name)
                        is_off, _ = _determine_official_status(c_copy, clean_url, paper_json, contributors)
                    except Exception as e:
                        logger.debug(f"Failed to check low-star official status: {e}")
                
                if is_off:
                    logger.info(f"Pre-filtering: Kept low-star repository because it is verified as OFFICIAL: {clean_url}")
                    c_copy["is_official"] = True
                else:
                    logger.info(f"Pre-filtering: Pruned repository with insufficient stars (<5): {clean_url} (stars: {stars})")
                    c_copy["pruned"] = True
                    c_copy["prune_reason"] = f"Insufficient stars ({stars} < 5)"
                    pruned.append(c_copy)
                    continue

        kept.append(c_copy)

    return kept, pruned


def validate_repo(paper_json: dict, repo_url: str, c_metadata: Optional[dict] = None) -> dict:
    """
    Validate a single GitHub repository against parsed paper metadata.
    Supports both main repositories and specific repository subdirectories.

    Args:
        paper_json: Parsed paper JSON from Component 1
        repo_url: GitHub repository or subdirectory URL
        c_metadata: Optional search metadata dictionary for the candidate

    Returns:
        Dict containing validation scores, confidence_score, and classification.
    """
    owner, repo, subdir = _parse_github_url(repo_url)
    if not owner or not repo:
        logger.error(f"Could not parse GitHub URL: {repo_url}")
        return _error_result(repo_url, "Invalid GitHub URL")

    # Determine official status early so we can bypass pruning/mismatch for official repos
    contributors = _get_repo_contributors(owner, repo)
    is_official, official_reasons = _determine_official_status(c_metadata, repo_url, paper_json, contributors)

    # Fetch recursive tree directory list
    paths = _get_recursive_files(owner, repo)

    # Filter paths if subdir is specified to ensure code files check is accurate for subdirs
    if subdir:
        subdir_prefix = subdir.lower() + "/"
        paths = [p for p in paths if p.lower().startswith(subdir_prefix) or p.lower() == subdir.lower()]

    # 1. Verify code files presence
    has_code = False
    code_exts = {".py", ".ipynb", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs", ".sh", ".scala", ".cs", ".rb", ".php", ".lua", ".pl", ".m", ".kt"}
    for p in paths:
        if any(p.endswith(ext) for ext in code_exts):
            has_code = True
            break

    # If no code files found at all, and NOT official, apply immediate zero score and classify as mismatch
    if not has_code and not is_official:
        logger.warning(f"Validation: {owner}/{repo} (Subdir: {subdir}) contains zero code files. Classifying as MISMATCH.")
        return {
            "repo_url": repo_url,
            "confidence_score": 0.0,
            "classification": "MISMATCH",
            "is_official": False,
            "official_reasons": ["Contains zero code files in its target subdirectory tree."],
            "validator_scores": {
                "semantic": 0.0,
                "concept": 0.0,
                "dependency": 0.0,
                "codebase": 0.0
            },
            "error": "No code files found in repository tree"
        }

    # Fetch README and requirements, taking subdirectory into account
    readme = _fetch_readme(owner, repo, subdir)
    reqs_text = _fetch_requirements(owner, repo, subdir)

    if not readme:
        logger.warning(f"No README found for {owner}/{repo} (Subdir: {subdir}). Giving low validation score.")

    # Check if README contains personal notes/summaries/handwritten indicators
    pruned_by_readme = False
    pruned_by_readme_reason = ""
    if readme:
        readme_lower = readme.lower()
        note_indicators = ["my notes", "reading notes", "summaries", "handwritten"]
        found_indicators = [ind for ind in note_indicators if ind in readme_lower]
        if found_indicators:
            pruned_by_readme = True
            pruned_by_readme_reason = f"README contains personal note/summary/handwritten indicators: {', '.join(found_indicators)}"

    if pruned_by_readme and not is_official:
        result = {
            "repo_url": repo_url,
            "confidence_score": 0.0,
            "classification": "MISMATCH",
            "is_official": False,
            "official_reasons": [],
            "validator_scores": {
                "semantic": 0.0,
                "concept": 0.0,
                "dependency": 0.0,
                "codebase": 0.0
            },
            "pruned_by_readme": True,
            "pruned_by_readme_reason": pruned_by_readme_reason
        }
        logger.info(f"Validated {owner}/{repo} (Subdir: {subdir}) -> Pruned via README note filter: {pruned_by_readme_reason}")
        return result

    # Select up to 3 representative code files for codebase alignment scans
    code_paths = [p for p in paths if p.endswith(".py") or p.endswith(".ipynb")]
    
    def _file_priority(path: str) -> int:
        p_lower = path.lower()
        if "model" in p_lower or "net" in p_lower:
            return 1
        if "train" in p_lower or "algo" in p_lower or "dapo" in p_lower:
            return 2
        if "main" in p_lower or "run" in p_lower or "backtest" in p_lower:
            return 3
        if "loss" in p_lower or "metric" in p_lower or "eval" in p_lower:
            return 4
        return 5
        
    code_paths.sort(key=_file_priority)
    code_contents = []
    for cp in code_paths[:5]:
        content = _fetch_cached_file(owner, repo, cp)
        if content:
            code_contents.append(content)

    # 2. Parse/detect frameworks and keywords from paper
    paper_spec = {
        "abstract": paper_json.get("abstract", "") or paper_json.get("title", ""),
        "title": paper_json.get("title", ""),
        "keywords": _extract_paper_keywords(paper_json),
        "frameworks": _detect_paper_frameworks(paper_json),
        "codebase_terms": _extract_codebase_terms(paper_json)
    }

    # 3. Direct Code Import Dependency Matching Fallback
    # If requirements text is absent, scan python code files for framework imports
    if not reqs_text and code_contents:
        combined_code = "\n".join(code_contents)
        detected_fws = []
        if "import torch" in combined_code or "from torch" in combined_code:
            detected_fws.append("PyTorch")
        if "import tensorflow" in combined_code or "from tensorflow" in combined_code or "import keras" in combined_code:
            detected_fws.append("TensorFlow")
        if "import jax" in combined_code or "from jax" in combined_code:
            detected_fws.append("JAX")
        
        if detected_fws:
            reqs_text = "\n".join(detected_fws)
            logger.info(f"Dependency matching: Detected framework imports directly in code for {owner}/{repo}: {detected_fws}")

    # 4. Run Individual Heuristic Validators
    concept_score = ConceptValidator().validate(paper_spec, readme)
    dependency_score = DependencyValidator().validate(paper_spec, reqs_text)
    codebase_score = CodebaseAlignmentValidator().validate(paper_spec, code_contents, paths)

    # Fast-Check Early Exit for SentenceTransformer
    if concept_score == 0.0 and readme:
        logger.info(f"Fast check: Skeptically skipped SentenceTransformer embedding check for {owner}/{repo} (0 keywords matched in README).")
        semantic_score = 0.0
    else:
        semantic_score = SemanticValidator().validate(paper_spec, readme)

    # 5. Calculate weighted confidence score
    confidence_score = (
        (semantic_score * 0.25) +
        (concept_score * 0.30) +
        (dependency_score * 0.20) +
        (codebase_score * 0.25)
    )

    # 6. Classify based on score using adjusted thresholds:
    # Mismatch < 0.60, Partial Match >= 0.60 and < 0.80, Match >= 0.80
    classification = "UNKNOWN"
    if not readme and not reqs_text and not code_contents:
        classification = "UNKNOWN"
        confidence_score = 0.0
    elif confidence_score >= 0.80:
        classification = "MATCH"
    elif confidence_score >= 0.60:
        classification = "PARTIAL_MATCH"
    else:
        classification = "MISMATCH"

    result = {
        "repo_url": repo_url,
        "confidence_score": round(confidence_score, 4),
        "classification": classification,
        "is_official": is_official,
        "official_reasons": official_reasons,
        "validator_scores": {
            "semantic": round(semantic_score, 4),
            "concept": round(concept_score, 4),
            "dependency": round(dependency_score, 4),
            "codebase": round(codebase_score, 4)
        },
        "readme_content": readme,
        "description": c_metadata.get("description") if c_metadata else ""
    }

    if is_official:
        logger.info(f"Official Repo Verified: {owner}/{repo}. Reasons: {official_reasons}. Boosting to MATCH (1.00).")
        result["confidence_score"] = 1.00
        result["classification"] = "MATCH"

    logger.info(f"Validated {owner}/{repo} (Subdir: {subdir}) -> Score: {result['confidence_score']:.2%} ({result['classification']}) [Sem: {semantic_score:.2f}, Con: {concept_score:.2f}, Dep: {dependency_score:.2f}, Code: {codebase_score:.2f}]")
    return result


def _check_implementation_of_in_same_sentence(text: str, paper_title: str) -> bool:
    if not text or not paper_title:
        return False
    text_clean = text.lower()
    title_clean = paper_title.lower()
    
    # Split text into sentences by punctuation or newlines
    sentences = re.split(r'[.!?\n]+', text_clean)
    
    # Generate match candidates from title
    title_norm = re.sub(r'[^a-z0-9\s]', ' ', title_clean).strip()
    title_words = title_norm.split()
    
    candidates = []
    # 1. Full title
    candidates.append(title_clean)
    # 2. Normalized full title with spaces collapsed
    if title_norm:
        candidates.append(" ".join(title_words))
        candidates.append(re.sub(r'\s+', '', title_norm))
        
    # 3. Part before colon
    if ":" in paper_title:
        prefix = paper_title.split(":")[0].strip().lower()
        if len(prefix) >= 3:
            candidates.append(prefix)
            prefix_norm = re.sub(r'[^a-z0-9\s]', ' ', prefix).strip()
            if prefix_norm:
                candidates.append(" ".join(prefix_norm.split()))
                
    # 4. Main acronyms
    acronyms = re.findall(r"\b[A-Z0-9]{2,8}\b", paper_title)
    for acr in acronyms:
        if len(acr) >= 2:
            candidates.append(acr.lower())
            
    # 5. First 3 words of the title (if long enough)
    if len(title_words) >= 3:
        candidates.append(" ".join(title_words[:3]))
        
    # Deduplicate candidates
    candidates = list(set([c for c in candidates if len(c) >= 3]))
    
    for sentence in sentences:
        sentence = sentence.strip()
        sentence_norm = re.sub(r'[^a-z0-9\s]', ' ', sentence)
        sentence_norm = " ".join(sentence_norm.split())
        
        if "implementation of" in sentence:
            for cand in candidates:
                if cand in sentence or cand in sentence_norm:
                    return True
    return False


def validate_and_rank_candidates(paper_json: dict, candidates: list, include_pruned: bool = False) -> tuple:
    """
    Validate multiple candidate repositories and rank them by confidence score.
    Applies skeptical pre-filtering and metadata boosts (is_official, acronym alignment).

    Args:
        paper_json: Parsed paper JSON
        candidates: List of repository dicts (e.g. from search_implementation)
        include_pruned: If True, include pruned repos with classification "PRUNED" and a reason

    Returns:
        Tuple: (best_candidate_validation_result, list_of_all_validation_results)
    """
    if not candidates:
        return _error_result(None, "No candidates to validate"), []

    # Skeptically prune candidates list first
    filtered_candidates, pre_pruned = _pre_filter_candidates(candidates, paper_json)
    if not filtered_candidates:
        filtered_candidates = candidates
        pre_pruned = []

    # Preload file trees for all candidate repos in one Coral query where possible
    repo_keys = []
    for c in filtered_candidates:
        url = c.get("repo_url")
        if not url:
            continue
        owner, repo, _ = _parse_github_url(url)
        if owner and repo:
            repo_keys.append((owner, repo))
    _batch_get_recursive_files(repo_keys)

    from concurrent.futures import ThreadPoolExecutor, as_completed

    ranked_results = []
    with ThreadPoolExecutor(max_workers=min(len(filtered_candidates), 8)) as executor:
        future_to_cand = {}
        for c in filtered_candidates:
            url = c.get("repo_url")
            if not url:
                continue
            future = executor.submit(validate_repo, paper_json, url, c)
            future_to_cand[future] = c

        for future in as_completed(future_to_cand):
            c = future_to_cand[future]
            url = c.get("repo_url")
            try:
                val_res = future.result()
                # Record pre-boost raw score returned by validate_repo
                val_res["pre_boost_score"] = val_res.get("confidence_score", 0.0)
                # Initialize boost details dictionary
                val_res["boosts"] = {
                    "weak_officiality": 0.0,
                    "acronym": 0.0,
                    "stars": 0.0,
                    "implementation_of_sentence": 0.0,
                    "official_override": False,
                    "applied_total": 0.0,
                }
                # Carry over search metadata
                val_res["search_method"] = c.get("search_method")
                val_res["stars"] = c.get("stars", 0)
                
                # --- Apply Meta-Boosts to confidence score of NON-official repos ---
                if not val_res.get("is_official"):
                    boost = 0.0
                    
                    # 1. Check known organizations or owner name matches author name (weak officiality boost)
                    owner, repo, subdir = _parse_github_url(url)
                    owner_lower = owner.lower() if owner else ""
                    known_orgs = {"microsoft", "google", "meta", "facebook", "openai", "huggingface", "pytorch", "tensorflow", "deepmind", "nvidia", "apple", "amazon", "aws", "salesforce", "ucberkeley", "stanford", "mit", "bytedance", "baidu"}
                    
                    paper_authors = paper_json.get("authors", [])
                    author_names = []
                    for author in paper_authors:
                        if isinstance(author, dict):
                            first = author.get("first", "").strip().lower()
                            last = author.get("last", "").strip().lower()
                            author_names.append(last)
                            author_names.append(first)
                            author_names.append(f"{first}-{last}")
                            if first and last:
                                author_names.extend([
                                    f"{first}_{last}", f"{first}{last}", f"{last}{first}",
                                    f"{first[0]}{last}", f"{last}{first[0]}"
                                ])
                            name_val = author.get("name", "").strip().lower()
                            if name_val:
                                parts = name_val.split()
                                author_names.extend(parts)
                                if len(parts) > 1:
                                    f = parts[0]
                                    l = parts[-1]
                                    author_names.extend([
                                        f"{f}-{l}", f"{f}_{l}", f"{f}{l}", f"{l}{f}",
                                        f"{f[0]}{l}", f"{l}{f[0]}"
                                    ])
                        elif isinstance(author, str):
                            parts = author.strip().lower().split()
                            author_names.extend(parts)
                            if len(parts) > 1:
                                author_names.append("-".join(parts))
                                f = parts[0]
                                l = parts[-1]
                                author_names.extend([
                                    f"{f}-{l}", f"{f}_{l}", f"{f}{l}", f"{l}{f}",
                                    f"{f[0]}{l}", f"{l}{f[0]}"
                                ])
                                
                    is_org_or_owner_match = (owner_lower in known_orgs) or (owner_lower in author_names)
                    if is_org_or_owner_match:
                        logger.info(f"Weak officiality boost: +0.15 boost applied to repo: {url}")
                        boost += 0.15
                        val_res["boosts"]["weak_officiality"] = 0.15
                    
                    # 2. Check Acronym alignment between repo name and paper title
                    title = paper_json.get("title", "")
                    title_words = re.findall(r"\b[a-zA-Z0-9]{3,8}\b", title)
                    repo_name_lower = url.split("/")[-1].lower()
                    acronym_match = False
                    for word in title_words:
                        if len(word) >= 3 and word.lower() in repo_name_lower:
                            acronym_match = True
                            break
                    if acronym_match:
                        logger.info(f"Acronym boost: +0.05 boost applied to aligned repo: {url}")
                        boost += 0.05
                        val_res["boosts"]["acronym"] = 0.05
                        
                    # 3. Add stars count linear scoring boost (up to 0.12 for 1200+ stars)
                    stars = val_res["stars"]
                    star_boost = min(0.12, (stars / 1200) * 0.12)
                    if star_boost > 0.0:
                        logger.info(f"Stars boost: +{star_boost:.3f} boost applied to repo with {stars} stars: {url}")
                        boost += star_boost
                        val_res["boosts"]["stars"] = round(star_boost, 4)
                        
                    # 4. Check "implementation of" and paper title in the same sentence in README or about text
                    readme_content = val_res.get("readme_content", "") or ""
                    about_text = val_res.get("description", "") or ""
                    desc = c.get("description", "") or ""
                    
                    has_sentence_match = (
                        _check_implementation_of_in_same_sentence(readme_content, title) or
                        _check_implementation_of_in_same_sentence(about_text, title) or
                        _check_implementation_of_in_same_sentence(desc, title)
                    )
                    if has_sentence_match:
                        logger.info(f"Implementation of sentence boost: +0.90 boost applied to repo: {url}")
                        boost += 0.90
                        val_res["boosts"]["implementation_of_sentence"] = 0.90
                        
                    if boost > 0.0:
                        # Non-official repos are capped at 0.99 so they can never exceed official repos (1.00)
                        applied = min(0.99, val_res["confidence_score"] + boost)
                        val_res["confidence_score"] = applied
                        val_res["boosts"]["applied_total"] = round(applied - val_res.get("pre_boost_score", 0.0), 4)
                else:
                    # Verified official repos are always set to 1.00
                    val_res["confidence_score"] = 1.00
                    val_res["boosts"]["official_override"] = True
                    val_res["boosts"]["applied_total"] = round(1.00 - val_res.get("pre_boost_score", 0.0), 4)
                    
                # Re-classify based on boosted/final score
                if val_res["confidence_score"] >= 0.80:
                    val_res["classification"] = "MATCH"
                elif val_res["confidence_score"] >= 0.60:
                    val_res["classification"] = "PARTIAL_MATCH"
                else:
                    val_res["classification"] = "MISMATCH"
                    
                ranked_results.append(val_res)
            except Exception as e:
                logger.error(f"Failed to validate candidate {url}: {e}")
                ranked_results.append({
                    "repo_url": url,
                    "confidence_score": 0.0,
                    "classification": "UNKNOWN",
                    "validator_scores": {"semantic": 0.0, "concept": 0.0, "dependency": 0.0, "codebase": 0.0},
                    "search_method": c.get("search_method"),
                    "stars": c.get("stars", 0),
                    "is_official": c.get("is_official", False),
                    "error": str(e)
                })

    # Sort candidates with robust tie-breaking:
    # 1. is_pruned (ascending, 0 for False, 1 for True)
    # 2. confidence_score (descending)
    # 3. is_official (descending, True first)
    # 4. has_subdir (descending, True first to prefer specific subdirectory URLs over root)
    # 5. stars count (descending)
    def _sort_key(x):
        is_pruned = 1 if x.get("classification") == "PRUNED" else 0
        score = x.get("confidence_score", 0.0)
        is_off = 1 if x.get("is_official") else 0
        
        url_val = x.get("repo_url", "")
        _, _, subdir = _parse_github_url(url_val)
        has_sub = 1 if subdir else 0
        
        stars = x.get("stars", 0)
        return (1 - is_pruned, score, is_off, has_sub, stars)

    ranked_results.sort(key=_sort_key, reverse=True)

    if len(ranked_results) >= 1:
        # Dominant candidate boost is only checked/applied on non-pruned candidates
        non_pruned_results = [r for r in ranked_results if r.get("classification") != "PRUNED"]
        if len(non_pruned_results) >= 1:
            best_cand = non_pruned_results[0]
            second_score = non_pruned_results[1].get("confidence_score", 0.0) if len(non_pruned_results) > 1 else 0.0
            if best_cand.get("confidence_score", 0.0) >= 0.65 and (best_cand.get("confidence_score", 0.0) - second_score) >= 0.15:
                logger.info(f"Dominant candidate boost applied: {best_cand['repo_url']} stands out. Boosting to MATCH.")
                best_cand["confidence_score"] = max(best_cand["confidence_score"], 0.85)
                best_cand["classification"] = "MATCH"

    # Filter out repositories that do not meet the minimum scores:
    # codebase score < 0.3, concept score < 0.4, dependency score < 0.4, semantic score < 0.4
    filtered_results = []
    for res in ranked_results:
        # Don't prune official candidates, 1.0 score candidates, or candidates with implementation_of sentence boost
        is_official = res.get("is_official") or res.get("is_official_context")
        has_impl_boost = res.get("boosts", {}).get("implementation_of_sentence", 0.0) > 0.0
        is_high_score = res.get("confidence_score", 0.0) >= 0.90 or res.get("pre_boost_score", 0.0) == 1.0
        
        if is_official or has_impl_boost or is_high_score:
            filtered_results.append(res)
            continue

        if res.get("pruned_by_readme"):
            reason = res.get("pruned_by_readme_reason")
            logger.info(f"Filtering out repository {res.get('repo_url')} due to: {reason}")
            if include_pruned:
                res["classification"] = "PRUNED"
                res["pruned"] = True
                res["prune_reason"] = reason
                filtered_results.append(res)
            continue

        scores = res.get("validator_scores", {})
        codebase = scores.get("codebase", 0.0)
        concept = scores.get("concept", 0.0)
        dependency = scores.get("dependency", 0.0)
        semantic = scores.get("semantic", 0.0)
        
        low_scores = []
        if codebase < 0.3:
            low_scores.append(f"codebase={codebase:.2f} < 0.3")
        if concept < 0.4:
            low_scores.append(f"concept={concept:.2f} < 0.4")
        if dependency < 0.4:
            low_scores.append(f"dependency={dependency:.2f} < 0.4")
        if semantic < 0.4:
            low_scores.append(f"semantic={semantic:.2f} < 0.4")

        if low_scores:
            reason = "Low validation scores: " + ", ".join(low_scores)
            logger.info(f"Filtering out repository {res.get('repo_url')} due to: {reason}")
            if include_pruned:
                res["classification"] = "PRUNED"
                res["pruned"] = True
                res["prune_reason"] = reason
                filtered_results.append(res)
            continue
        filtered_results.append(res)
    ranked_results = filtered_results

    # Include pre_pruned candidates at the bottom if requested
    if include_pruned:
        for c in pre_pruned:
            ranked_results.append({
                "repo_url": c.get("repo_url"),
                "confidence_score": 0.0,
                "classification": "PRUNED",
                "pruned": True,
                "prune_reason": c.get("prune_reason", "Pre-filtered"),
                "validator_scores": {"semantic": 0.0, "concept": 0.0, "dependency": 0.0, "codebase": 0.0},
                "search_method": c.get("search_method"),
                "stars": c.get("stars", 0),
                "is_official": c.get("is_official", False)
            })

    # Sort final results to ensure pruned are at the bottom
    ranked_results.sort(key=_sort_key, reverse=True)

    non_pruned_results = [r for r in ranked_results if r.get("classification") != "PRUNED"]
    best = non_pruned_results[0] if non_pruned_results else _error_result(None, "No candidates passed the validation thresholds")
    return best, ranked_results


# ===========================================================================
# Individual Validator Classes
# ===========================================================================

class SemanticValidator:
    """Validator 1: Semantic similarity of abstract ↔ README."""

    def validate(self, paper_spec: dict, readme_content: str) -> float:
        abstract = paper_spec.get("abstract", "")
        if not abstract or not readme_content:
            return 0.5

        # Clean README slightly for better comparison (cap it to avoid heavy encoding costs)
        readme_trimmed = readme_content[:2500].strip()

        # If sentence-transformers is available, calculate exact cosine similarity
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                # Load the reusable model singleton
                model = _get_embedding_model()
                if model:
                    emb_abstract = model.encode(abstract, convert_to_tensor=True)
                    emb_readme = model.encode(readme_trimmed, convert_to_tensor=True)
                    score = util.pytorch_cos_sim(emb_abstract, emb_readme).item()
                    # Normalize cosine sim from [-1, 1] to [0, 1]
                    normalized_score = max(0.0, min(1.0, (score + 1) / 2))
                    return normalized_score
            except Exception as e:
                logger.debug(f"SentenceTransformer encoding failed: {e}. Falling back to Jaccard bigram similarity.")

        # Jaccard Bigram Fallback
        return self._jaccard_bigram_similarity(abstract, readme_trimmed)

    def _jaccard_bigram_similarity(self, text1: str, text2: str) -> float:
        """Fallback: Computes Jaccard bigram similarity between two text blocks."""
        def get_bigrams(text):
            words = re.findall(r'[a-zA-Z0-9]+', text.lower())
            stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'with', 'by', 'of'}
            words = [w for w in words if w not in stop_words and len(w) > 1]
            return set(zip(words[:-1], words[1:])) if len(words) > 1 else set(words)

        bigrams1 = get_bigrams(text1)
        bigrams2 = get_bigrams(text2)

        if not bigrams1 or not bigrams2:
            return 0.0

        intersection = bigrams1.intersection(bigrams2)
        union = bigrams1.union(bigrams2)
        return len(intersection) / len(union)


class ConceptValidator:
    """Validator 2: Paper technical keywords fuzzy matched in README."""

    def validate(self, paper_spec: dict, readme_content: str) -> float:
        keywords = paper_spec.get("keywords", [])
        if not keywords or not readme_content:
            return 0.5

        readme_lower = readme_content.lower()
        matched = 0

        for keyword in keywords:
            keyword_lower = keyword.lower()
            if fuzz:
                if fuzz.partial_ratio(keyword_lower, readme_lower) >= 65:
                    matched += 1
            else:
                if keyword_lower in readme_lower:
                    matched += 1

        score = matched / len(keywords) if keywords else 0.5
        return score


class DependencyValidator:
    """Validator 3: framework requirements compared with repository requirements."""

    def validate(self, paper_spec: dict, requirements_text: str) -> float:
        frameworks = paper_spec.get("frameworks", [])
        if not frameworks:
            return 1.0

        if not requirements_text:
            return 0.5

        reqs_clean = requirements_text.lower()
        matched = 0

        for fw in frameworks:
            fw_lower = fw.lower()
            search_terms = [fw_lower]
            if fw_lower == "pytorch":
                search_terms.append("torch")
            elif fw_lower == "tensorflow":
                search_terms.extend(["tf-", "tf_"])

            if any(term in reqs_clean for term in search_terms):
                matched += 1

        score = matched / len(frameworks) if frameworks else 1.0
        return score


class CodebaseAlignmentValidator:
    """Validator 4: Architectural names, losses, datasets, benchmarks, and algorithms matched in codebase files."""

    def validate(self, paper_spec: dict, code_files_content: List[str], file_paths: List[str] = None) -> float:
        terms = paper_spec.get("codebase_terms", [])
        if not terms or (not code_files_content and not file_paths):
            return 0.5

        valid_terms = [t for t in terms if len(t) > 2]
        if not valid_terms:
            return 0.5

        combined_code = "\n".join(code_files_content).lower()
        combined_paths = "\n".join(file_paths).lower() if file_paths else ""
        
        # Define generic terms to downweight
        generic_terms = {
            "accuracy", "precision", "recall", "f1", "rmse", "mae", "mse", "auc", "roc", 
            "loss", "adam", "sgd", "rmsprop", "adamw", "dataset", "data", "train", "eval", 
            "test", "model", "net", "network", "layer", "attention", "encoder", "decoder", 
            "module", "block", "backtest", "sharpe", "sortino"
        }
        
        # Identify title words and acronyms to weight higher
        title = paper_spec.get("title", "") or ""
        title_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', title).lower()
        title_words = {w for w in title_clean.split() if len(w) > 2}
        
        acronyms = re.findall(r"\b[A-Z0-9]{2,8}\b", title)
        acronyms_lower = {acr.lower() for acr in acronyms}
        
        total_weight = 0.0
        matched_weight = 0.0
        
        for term in valid_terms:
            term_lower = term.lower()
            
            # Determine term weight
            if term_lower in generic_terms:
                weight = 0.25
            elif term_lower in title_words or term_lower in acronyms_lower or any(acr in term_lower for acr in acronyms_lower if len(acr) >= 3):
                weight = 2.0
            else:
                weight = 1.0
                
            total_weight += weight
            
            # Check match using word boundary pattern
            matched = False
            pattern = r'\b' + re.escape(term_lower) + r'\b'
            if re.search(pattern, combined_code):
                matched = True
            elif ("-" in term_lower or "_" in term_lower or "." in term_lower) and term_lower in combined_code:
                # Fallback for terms with special chars
                matched = True
            elif file_paths:
                # Scan recursive file paths/names too
                if re.search(pattern, combined_paths):
                    matched = True
                elif ("-" in term_lower or "_" in term_lower or "." in term_lower) and term_lower in combined_paths:
                    matched = True
                elif term_lower in combined_paths:
                    # For title words or acronyms, allow substring match in paths as they represent the paper context
                    if term_lower in title_words or term_lower in acronyms_lower:
                        matched = True
                
            if matched:
                matched_weight += weight

        if total_weight == 0.0:
            return 0.5
            
        ratio = matched_weight / total_weight
        # Scaled score: mapping 0.5 ratio to 1.0, generous alignment scaling (ratio * 2.0)
        score = min(1.0, ratio * 2.0)
        # Base floor of 0.2 if at least some weight matched, otherwise 0.0
        if matched_weight > 0.0:
            score = max(0.2, score)
        return score


# ===========================================================================
# Internal Utility Helpers & Caching File Fetchers
# ===========================================================================

def _parse_github_url(url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract owner, repo name, and optional subdirectory path from GitHub URL."""
    if not url:
        return None, None, None

    # Clean trailing punctuation and slashes recursively to handle trailing slash and dots
    url = url.strip()
    while url.endswith("/") or any(url.endswith(c) for c in ".,;:()[]{}"):
        url = url.rstrip("/").rstrip(".,;:()[]{}")

    # Remove protocol and github.com
    match = re.search(r"github\.com/([^/]+)/([^/]+)(.*)$", url, re.IGNORECASE)
    if not match:
        return None, None, None

    owner = match.group(1)
    repo = match.group(2)
    rest = match.group(3).strip("/")

    # Strip .git from repo name if present
    if repo.lower().endswith(".git"):
        repo = repo[:-4]

    if not rest:
        return owner, repo, None

    # If rest starts with tree/<branch> or blob/<branch>, extract the subdir path after that
    parts = rest.split("/")
    if len(parts) >= 2 and parts[0] in ("tree", "blob"):
        subdir = "/".join(parts[2:]) if len(parts) > 2 else None
        return owner, repo, subdir if subdir else None

    return owner, repo, rest


def _get_directory_contents(owner: str, repo: str, subdir: Optional[str] = None) -> set:
    """
    List all filenames at the root or specific subdirectory of a repository.
    Cached globally to prevent blind HTTP 404s for missing files.
    """
    key = (owner.lower(), repo.lower(), subdir.lower() if subdir else None)
    if key in _directory_contents_cache:
        return _directory_contents_cache[key]

    filenames = set()
    coral = get_coral_client()
    path_query = subdir if subdir else ""

    if coral.available:
        try:
            res = coral.sql(f"""
                SELECT path
                FROM github.contents
                WHERE owner = '{owner}' AND repo = '{repo}' AND path = '{path_query}'
            """)
            if res and "results" in res:
                for row in res["results"]:
                    p = row.get("path", "")
                    name = p.split("/")[-1] if p else ""
                    if name:
                        filenames.add(name)
        except Exception as e:
            logger.debug(f"Coral directory list failed for {owner}/{repo}: {e}")

    if not filenames:
        # Fallback to GitHub REST API contents endpoint
        import requests
        headers = {}
        token = os.environ.get("GITHUB_TOKEN")
        if token and "dummy" not in token.lower() and "your_" not in token.lower():
            headers["Authorization"] = f"token {token}"
        try:
            resp = requests.get(
                f"https://api.github.com/repos/{owner}/{repo}/contents/{path_query}",
                headers=headers, timeout=10
            )
            if resp.status_code == 200:
                items = resp.json()
                if isinstance(items, list):
                    for item in items:
                        name = item.get("name", "")
                        if name:
                            filenames.add(name)
            else:
                logger.debug(f"GitHub API directory list returned status {resp.status_code} for {owner}/{repo}")
        except Exception as e:
            logger.debug(f"GitHub API directory list failed for {owner}/{repo}: {e}")

    _directory_contents_cache[key] = filenames
    return filenames


def _fetch_cached_file(owner: str, repo: str, path: str) -> Optional[str]:
    """Fetch file content from Coral SQL or GitHub REST API, wrapping it in a global session cache."""
    key = (owner.lower(), repo.lower(), path.lower())
    if key in _fetched_content_cache:
        return _fetched_content_cache[key]

    content = None
    coral = get_coral_client()
    if coral.available:
        try:
            res = coral.sql(f"""
                SELECT content_text as content
                FROM github.contents
                WHERE owner = '{owner}' AND repo = '{repo}' AND path = '{path}'
            """)
            if res and "results" in res and res["results"]:
                content = res["results"][0].get("content") or res["results"][0].get("content_text")
        except Exception as e:
            logger.debug(f"Coral fetch failed for {path}: {e}")

    if not content:
        # Fallback to GitHub REST API content fetch
        content = _fetch_github_file_direct(owner, repo, path)

    if content is not None:
        _fetched_content_cache[key] = content
    return content


def _fetch_readme(owner: str, repo: str, subdir: Optional[str] = None) -> str:
    """Fetch README.md file content from specific subdirectory or root, checking directory lists first."""
    files = _get_directory_contents(owner, repo, subdir)
    
    # 1. Search in subdirectory first if present
    if subdir:
        readme_name = next((f for f in files if f.lower().startswith("readme")), None)
        if readme_name:
            path = f"{subdir}/{readme_name}"
            content = _fetch_cached_file(owner, repo, path)
            if content:
                return content

    # 2. Fall back to root-level search
    root_files = _get_directory_contents(owner, repo, None)
    readme_name = next((f for f in root_files if f.lower().startswith("readme")), None)
    if readme_name:
        return _fetch_cached_file(owner, repo, readme_name) or ""

    return ""


def _fetch_requirements(owner: str, repo: str, subdir: Optional[str] = None) -> str:
    """Fetch requirements.txt / setup.py dependencies from specific subdirectory or root, checking directory lists first."""
    
    # 1. Search in subdirectory first if present
    if subdir:
        files = _get_directory_contents(owner, repo, subdir)
        # Check requirements.txt
        reqs_name = next((f for f in files if "requirement" in f.lower() and f.lower().endswith(".txt")), None)
        if reqs_name:
            return _fetch_cached_file(owner, repo, f"{subdir}/{reqs_name}") or ""
        # Check setup.py
        setup_name = next((f for f in files if f.lower() == "setup.py"), None)
        if setup_name:
            return _fetch_cached_file(owner, repo, f"{subdir}/{setup_name}") or ""

    # 2. Fall back to root-level search
    root_files = _get_directory_contents(owner, repo, None)
    reqs_name = next((f for f in root_files if "requirement" in f.lower() and f.lower().endswith(".txt")), None)
    if reqs_name:
        return _fetch_cached_file(owner, repo, reqs_name) or ""

    setup_name = next((f for f in root_files if f.lower() == "setup.py"), None)
    if setup_name:
        return _fetch_cached_file(owner, repo, setup_name) or ""

    return ""


def _fetch_github_file_direct(owner: str, repo: str, path: str) -> Optional[str]:
    """Helper to fetch a file's content directly from the GitHub API using requests."""
    import requests
    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token and "dummy" not in token.lower() and "your_" not in token.lower():
        headers["Authorization"] = f"token {token}"
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
            headers=headers, timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                if data.get("encoding") == "base64":
                    import base64
                    return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                elif data.get("download_url"):
                    dl_resp = requests.get(data["download_url"], timeout=10)
                    if dl_resp.status_code == 200:
                        return dl_resp.text
    except Exception as e:
        logger.debug(f"GitHub API fetch of {path} failed: {e}")
    return None


def _extract_paper_keywords(paper_json: dict) -> List[str]:
    """Extract 5-7 core technical keywords from paper metadata."""
    if paper_json.get("keywords"):
        return paper_json["keywords"]

    abstract = paper_json.get("abstract", "")
    title = paper_json.get("title", "")
    
    stop_words = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "for", "and", "nor", "but", "or",
        "yet", "so", "at", "by", "from", "in", "of", "on", "to", "with",
        "as", "this", "that", "these", "those", "we", "our", "it", "its",
        "which", "such", "also", "than", "more", "both", "each", "show",
        "propose", "paper", "approach", "method", "results", "using", "based",
        "new", "model", "data", "use", "used", "work", "proposed", "shown",
    }
    
    text = (title + " " + abstract).lower()
    words = re.findall(r'[a-zA-Z]{4,}', text)
    words = [w for w in words if w not in stop_words]

    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:7]]


def _detect_paper_frameworks(paper_json: dict) -> List[str]:
    """Detect expected deep learning frameworks from paper text."""
    text = (paper_json.get("title", "") + " " + paper_json.get("abstract", "")).lower()
    frameworks = []
    if "pytorch" in text or " torch" in text:
        frameworks.append("PyTorch")
    if "tensorflow" in text or " keras" in text or " tf " in text:
        frameworks.append("TensorFlow")
    if "jax" in text:
        frameworks.append("JAX")
        
    if not frameworks:
        frameworks.append("PyTorch")
    return frameworks


def _extract_codebase_terms(paper_json: dict) -> List[str]:
    """
    Extract unique technical names from paper text representing losses, models,
    datasets, algorithms, benchmarks, and metrics for codebase alignment matching.
    Scans title, abstract, keywords, and first 15k characters of full_text to build a robust query set.
    """
    terms = set()
    title = paper_json.get("title", "") or ""
    abstract = paper_json.get("abstract", "") or ""
    keywords = paper_json.get("keywords", []) or []
    full_text = paper_json.get("full_text", "") or ""
    
    # Combined text for scanning
    text = f"{title} {abstract} " + " ".join(keywords)
    if full_text:
        text += " " + full_text[:15000]

    # 1. Losses (e.g. MSELoss, CrossEntropyLoss, ActorLoss, smooth_l1_loss)
    for match in re.findall(r"\b[a-zA-Z0-9_-]*(?:loss|Loss)\b", text):
        if len(match) > 3 and match.lower() != "loss":
            terms.add(match)
            
    # 2. Model/Layer architectures (e.g. MultiHeadAttention, ActorNet, TransformerEncoder, LSTM, ResNet)
    model_patterns = [
        r"\b\w*(?:Net|Network)\b", 
        r"\b\w*Attention\b", 
        r"\b\w*Layer\b", 
        r"\b\w*(?:Encoder|Decoder)\b",
        r"\b\w*(?:Module|Block)\b",
        r"\b(?:ResNet|ViT|VGG|DenseNet|EfficientNet|BERT|GPT|RoBERTa|T5|BART|LSTM|GRU|RNN|CNN|MLP|GNN|GCN|GAT)\d*\b"
    ]
    for pat in model_patterns:
        for match in re.findall(pat, text):
            if len(match) > 3 and match.lower() not in ["net", "network", "attention", "layer", "encoder", "decoder", "module", "block"]:
                terms.add(match)
                
    # 3. Capitalized/Uppercase Algorithms, Datasets, and Benchmarks (e.g. DAPO, FinRL, PPO, DDPG, SAC, TD3, DQN)
    for match in re.findall(r"\b[A-Z][A-Z0-9a-z_-]{2,12}\b", text):
        if match.lower() not in ["the", "and", "for", "with", "this", "model", "paper", "data", "test", "train", "loss", "code"]:
            terms.add(match)

    # 4. Specific Datasets/Benchmarks ending with dataset, dataset names, or common acronyms
    dataset_patterns = [
        r"\b[A-Za-z0-9_-]+(?:[Dd]ataset|[Dd]ata)\b",
        r"\b(?:CIFAR\d*|ImageNet|MNIST|SQuAD|WikiText|GLUE|SuperGLUE|MMLU|HumanEval|GSM8K|MATH|CoNLL|IMDB|PennTreebank|MSCOCO|LFW|WMT)\b"
    ]
    for pat in dataset_patterns:
        for match in re.findall(pat, text):
            if len(match) > 3 and match.lower() not in ["dataset", "data"]:
                terms.add(match)

    # 5. Reinforcement Learning & Optimization Algorithms
    rl_algo_patterns = [
        r"\b(?:PPO|DPO|DAPO|DDPG|SAC|TD3|DQN|A3C|A2C|TRPO|REINFORCE|QLearning|SARSA)\b",
        r"\b(?:Adam|AdamW|SGD|RMSprop|Adagrad|Adadelta|LBFGS)\b"
    ]
    for pat in rl_algo_patterns:
        for match in re.findall(pat, text, re.IGNORECASE):
            terms.add(match)

    # 6. Evaluation Metrics & Benchmarks
    metric_terms = [
        "accuracy", "precision", "recall", "f1", "rmse", "mae", "mse", "bleu", "rouge", 
        "perplexity", "auc", "roc", "map", "fid", "exact_match", "sharpe", "sortino", 
        "cumulative_return", "max_drawdown", "mdd", "backtest"
    ]
    for t in metric_terms:
        if re.search(rf"\b{t}\b", text, re.IGNORECASE):
            terms.add(t)

    # 7. Direct Keywords
    for kw in keywords:
        if len(kw) > 3:
            terms.add(kw)
            
    return list(terms)


def _error_result(repo_url: Optional[str], message: str) -> dict:
    return {
        "repo_url": repo_url,
        "confidence_score": 0.0,
        "classification": "UNKNOWN",
        "validator_scores": {
            "semantic": 0.0,
            "concept": 0.0,
            "dependency": 0.0
        },
        "error": message
    }


def validate_repo_relevance(repo_url: str, paper_json: dict) -> dict:
    """
    Validate a single GitHub repository's relevance to a paper.
    Exposed for batch testing scripts and integration.
    """
    val_res = validate_repo(paper_json, repo_url)
    score = int(val_res.get("confidence_score", 0.0) * 100)
    classification = val_res.get("classification", "UNKNOWN")
    
    if classification == "MATCH":
        verdict = "relevant"
    elif classification == "PARTIAL_MATCH":
        verdict = "possibly_irrelevant"
    else:
        verdict = "irrelevant"
        
    evidence = []
    warnings = []
    
    v_scores = val_res.get("validator_scores", {})
    sem = v_scores.get("semantic", 0.0)
    con = v_scores.get("concept", 0.0)
    dep = v_scores.get("dependency", 0.0)
    code = v_scores.get("codebase", 0.0)
    
    if sem >= 0.6:
        evidence.append(f"High semantic similarity between abstract and README ({sem:.2f})")
    else:
        warnings.append(f"Low semantic similarity between abstract and README ({sem:.2f})")
        
    if con >= 0.5:
        evidence.append(f"Fuzzy matched technical keywords in README ({con:.2f})")
    else:
        warnings.append(f"Few technical keywords matched in README ({con:.2f})")
        
    if dep >= 0.5:
        evidence.append(f"Framework dependencies match requirements ({dep:.2f})")
    else:
        warnings.append(f"Framework dependencies mismatch or not found ({dep:.2f})")
        
    if code >= 0.5:
        evidence.append(f"Codebase alignment terms found in source code ({code:.2f})")
    else:
        warnings.append(f"Few codebase alignment terms found in code ({code:.2f})")
        
    return {
        "relevance_score": score,
        "verdict": verdict,
        "evidence": evidence,
        "warnings": warnings
    }


def format_relevance_badge(relevance: dict) -> str:
    score = relevance.get("relevance_score", 0)
    verdict = relevance.get("verdict", "irrelevant")
    if verdict == "relevant":
        return f"🎯 Relevant ({score}%)"
    elif verdict == "possibly_irrelevant":
        return f"⚠️ Possibly Irrelevant ({score}%)"
    else:
        return f"❌ Irrelevant ({score}%)"
