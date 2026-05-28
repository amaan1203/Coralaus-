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

# Silence noisy optional dependency 404 loggers
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)

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


def _get_embedding_model() -> Optional['SentenceTransformer']:
    """Load and return the reusable SentenceTransformer model singleton."""
    global _embedding_model
    if _embedding_model is None and SENTENCE_TRANSFORMERS_AVAILABLE:
        try:
            logger.info("Loading SentenceTransformer model 'all-MiniLM-L6-v2' (reusable singleton)...")
            _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            logger.debug(f"Failed to load SentenceTransformer model singleton: {e}")
    return _embedding_model


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
                WHERE owner = '{owner}' AND repo = '{repo}' AND tree_sha = 'HEAD' AND recursive = 'true' AND type = 'blob'
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


def _pre_filter_candidates(candidates: list) -> list:
    """
    Skeptically prune obvious non-implementations based strictly on local metadata
    (URL owner, URL name, description) before any network fetches are run.
    Prunes curated awesome lists and obvious university homework/lab coursework assignments.
    Does NOT prune based on stars or official/unofficial status.
    """
    if not candidates:
        return []

    pruned = []
    for c in candidates:
        url = c.get("repo_url", "").lower()
        desc = c.get("description", "") or ""
        desc_lower = desc.lower()

        # 1. Prune awesome lists and curated collections
        if "awesome-" in url or "awesome list" in desc_lower or "curated list" in desc_lower:
            logger.info(f"Pre-filtering: Skeptically pruned awesome list repository: {c.get('repo_url')}")
            continue

        # 2. Prune obvious university homework/coursework assignments
        course_patterns = [
            r"cs\d{3}.*assignment", r"assignment.*cs\d{3}",
            r"homework.*cs\d{3}", r"coursework", r"university assignment",
            r"homework assignment", r"lab assignment", r"class assignment",
            r"cs-\d{3}", r"cs\d{3}-"
        ]
        is_course = False
        for pat in course_patterns:
            if re.search(pat, url) or re.search(pat, desc_lower):
                is_course = True
                break

        if is_course:
            logger.info(f"Pre-filtering: Skeptically pruned obvious coursework/homework assignment: {c.get('repo_url')}")
            continue

        pruned.append(c)

    return pruned


def validate_repo(paper_json: dict, repo_url: str) -> dict:
    """
    Validate a single GitHub repository against parsed paper metadata.
    Supports both main repositories and specific repository subdirectories.

    Args:
        paper_json: Parsed paper JSON from Component 1
        repo_url: GitHub repository or subdirectory URL

    Returns:
        Dict containing validation scores, confidence_score, and classification.
    """
    owner, repo, subdir = _parse_github_url(repo_url)
    if not owner or not repo:
        logger.error(f"Could not parse GitHub URL: {repo_url}")
        return _error_result(repo_url, "Invalid GitHub URL")

    # Fetch recursive tree directory list
    paths = _get_recursive_files(owner, repo)

    # 1. Verify code files presence
    has_code = False
    code_exts = {".py", ".ipynb", ".js", ".ts", ".java", ".cpp", ".c", ".h", ".hpp", ".go", ".rs", ".sh", ".scala", ".cs", ".rb", ".php", ".lua", ".pl", ".m", ".kt"}
    for p in paths:
        if any(p.endswith(ext) for ext in code_exts):
            has_code = True
            break

    # If no code files found at all, apply immediate zero score and classify as mismatch
    if not has_code:
        logger.warning(f"Validation: {owner}/{repo} contains zero code files in its entire repository tree. Classifying as MISMATCH.")
        return {
            "repo_url": repo_url,
            "confidence_score": 0.0,
            "classification": "MISMATCH",
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
    for cp in code_paths[:3]:
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
    codebase_score = CodebaseAlignmentValidator().validate(paper_spec, code_contents)

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
        "validator_scores": {
            "semantic": round(semantic_score, 4),
            "concept": round(concept_score, 4),
            "dependency": round(dependency_score, 4),
            "codebase": round(codebase_score, 4)
        }
    }

    logger.info(f"Validated {owner}/{repo} (Subdir: {subdir}) -> Score: {confidence_score:.2%} ({classification}) [Sem: {semantic_score:.2f}, Con: {concept_score:.2f}, Dep: {dependency_score:.2f}, Code: {codebase_score:.2f}]")
    return result


def validate_and_rank_candidates(paper_json: dict, candidates: list) -> tuple:
    """
    Validate multiple candidate repositories and rank them by confidence score.
    Applies skeptical pre-filtering and metadata boosts (is_official, acronym alignment).

    Args:
        paper_json: Parsed paper JSON
        candidates: List of repository dicts (e.g. from search_implementation)

    Returns:
        Tuple: (best_candidate_validation_result, list_of_all_validation_results)
    """
    if not candidates:
        return _error_result(None, "No candidates to validate"), []

    # Skeptically prune candidates list first
    filtered_candidates = _pre_filter_candidates(candidates)
    if not filtered_candidates:
        filtered_candidates = candidates

    ranked_results = []
    for c in filtered_candidates:
        url = c.get("repo_url")
        if not url:
            continue
        try:
            val_res = validate_repo(paper_json, url)
            # Carry over search metadata
            val_res["search_method"] = c.get("search_method")
            val_res["stars"] = c.get("stars", 0)
            val_res["is_official"] = c.get("is_official", False)
            
            # --- Apply Meta-Boosts to confidence score ---
            boost = 0.0
            if val_res.get("is_official"):
                logger.info(f"Officiality boost: +0.15 boost applied to official repo: {url}")
                boost += 0.15
                
            # Check Acronym alignment between repo name and paper title
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
                
            if boost > 0.0:
                val_res["confidence_score"] = min(1.0, val_res["confidence_score"] + boost)
                
                # Re-classify based on boosted score
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

    # Sort candidates by confidence score in descending order
    ranked_results.sort(key=lambda x: x["confidence_score"], reverse=True)

    best = ranked_results[0] if ranked_results else _error_result(None, "No candidates validated")
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

    def validate(self, paper_spec: dict, code_files_content: List[str]) -> float:
        terms = paper_spec.get("codebase_terms", [])
        if not terms or not code_files_content:
            return 0.5

        combined_code = "\n".join(code_files_content).lower()
        matched = 0
        for term in terms:
            term_lower = term.lower()
            if fuzz:
                if fuzz.partial_ratio(term_lower, combined_code) >= 75:
                    matched += 1
            else:
                if term_lower in combined_code:
                    matched += 1

        score = matched / len(terms) if terms else 0.5
        return score


# ===========================================================================
# Internal Utility Helpers & Caching File Fetchers
# ===========================================================================

def _parse_github_url(url: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Extract owner, repo name, and optional subdirectory path from GitHub URL."""
    if not url:
        return None, None, None

    # Match owner/repo/tree/branch/path or blob/branch/path
    match = re.search(r"github\.com/([^/]+)/([^/]+)/(?:tree|blob)/[^/]+/(.+)$", url)
    if match:
        owner = match.group(1)
        repo = match.group(2)
        subdir = match.group(3).rstrip("/")
        return owner, repo, subdir

    # Standard formats
    match = re.search(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url)
    if match:
        return match.group(1), match.group(2), None

    match = re.search(r"github\.com/([^/]+)/([^/]+?)(?:/.*)?$", url)
    if match:
        return match.group(1), match.group(2), None

    return None, None, None


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
