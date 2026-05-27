# -*- coding: utf-8 -*-
"""
Repo Relevance Validator - Component 2b (v2)

After search_implementation() returns a GitHub repo, this module verifies
that the repo is actually the implementation of *this specific paper* (not
just a similar-topic repo).

Multi-signal scoring approach (v2):
    Phase 0: Fetch all repo data upfront (Coral SQL / GitHub API)
    Check 1: Repository existence check                      (gate / penalty)
    Check 2: Paper-title / repo-title similarity             (max 30 pts)
    Check 3: README semantic validation                      (max 15 pts)
    Check 4: Code-content validation                         (max 40 pts)
    Check 5: Paper-to-code concept alignment                 (max 20 pts)
    Check 6: Completeness check                              (max 20 pts)
    Check 7: Relevance ratio by file inspection              (max 10 pts)
    Check 8: Execution sanity (static always; dynamic opt-in)(max 10 pts)
    Check 9: Official vs unofficial classifier               (max 10 pts)
    Phase F: Normalize raw score (0–155) → 0–100, apply rubric

    Score bands:
        81–100 → "highly_confident"   (official or near-complete)
        61–80  → "strong_match"       (strong match, likely useful)
        41–60  → "plausible"          (plausible, possibly incomplete)
        21–40  → "weak_match"         (has code, weak paper connection)
         0–20  → "unrelated"          (empty or unrelated)

Public API
----------
    validate_repo_relevance(repo_url, paper_json, *, execution_check=False) -> dict
    format_relevance_badge(result) -> str
"""

import ast
import logging
import os
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Score constants
# ---------------------------------------------------------------------------

MAX_TITLE_SIMILARITY  = 30
MAX_README_SEMANTICS  = 15
MAX_CODE_CONTENT      = 40
MAX_CONCEPT_ALIGNMENT = 20
MAX_COMPLETENESS      = 20
MAX_RELEVANCE_RATIO   = 10
MAX_EXECUTION_SANITY  = 10
MAX_OFFICIAL          = 10
RAW_TOTAL             = 155   # sum of all maximums (normalised to 100)

BAND_HIGHLY_CONFIDENT = 81
BAND_STRONG_MATCH     = 61
BAND_PLAUSIBLE        = 41
BAND_WEAK_MATCH       = 21

# ---------------------------------------------------------------------------
# File classification constants
# ---------------------------------------------------------------------------

CODE_EXTENSIONS = {
    ".py", ".ipynb", ".cpp", ".c", ".h", ".hpp", ".cu", ".cuh",
    ".js", ".ts", ".java", ".go", ".rs", ".sh", ".r", ".m", ".jl",
    ".lua", ".scala", ".kt", ".swift", ".rb", ".cs", ".f90",
}

ENTRYPOINT_NAMES = {
    "train.py", "train.sh", "training.py", "run.py", "run_training.py",
    "main.py", "run_experiment.py", "fit.py", "pretrain.py", "finetune.py",
}
INFERENCE_NAMES = {
    "infer.py", "inference.py", "predict.py", "demo.py", "evaluate.py",
    "eval.py", "test.py", "generate.py", "sample.py", "run_eval.py",
}
MODEL_NAMES = {
    "model.py", "models.py", "network.py", "net.py", "arch.py",
    "architecture.py", "module.py",
}
CONFIG_NAMES = {
    "config.yaml", "config.yml", "config.json", "hparams.yaml",
    "hparams.json", "args.py", "config.py", "params.py", "cfg.py",
    "default.yaml", "conf.yaml",
}
DEPENDENCY_FILES = {
    "requirements.txt", "setup.py", "setup.cfg", "pyproject.toml",
    "environment.yml", "conda.yaml", "pipfile", "pipfile.lock",
}

# Repos containing ONLY these files are considered trivially empty
TRIVIAL_FILES = {
    "readme.md", "readme.rst", "readme", "license", "license.md",
    "license.txt", "licence", "citation.cff", ".gitignore", ".gitattributes",
    "contributing.md", "code_of_conduct.md", "changelog.md",
}

STUB_PATTERNS = [
    r"\bTODO\b", r"\bFIXME\b", r"coming soon", r"to be released",
    r"\bpass\s*$", r"\braise NotImplementedError\b",
    r"#\s*placeholder", r"#\s*TODO", r"will be added", r"work in progress",
]

OFFICIAL_PHRASES = [
    "official implementation", "official code", "official repository",
    "code for our paper", "code for the paper", "as described in our paper",
    "this repository contains the code", "this repo contains the code",
    "code accompanying", "code associated with",
]
UNOFFICIAL_PHRASES = [
    "reimplementation", "re-implementation", "my implementation",
    "unofficial", "reproduc", "third.party", "not official",
    "community implementation",
]

# Frequently-cited ML datasets for concept extraction
KNOWN_DATASETS = {
    "imagenet", "coco", "cifar", "mnist", "squad", "glue", "superglue",
    "wikitext", "openwebtext", "laion", "cc3m", "cc12m", "voc", "ade20k",
    "kinetics", "ucf101", "hmdb51", "librispeech", "voxceleb", "commonvoice",
    "ms coco", "lfw", "celeba", "ffhq", "shapenet", "modelnet", "scannet",
    "nyu depth", "kitti", "waymo", "nuscenes", "bdd100k",
}

# ---------------------------------------------------------------------------
# Data container (populated once in Phase 0, shared by all checks)
# ---------------------------------------------------------------------------


@dataclass
class RepoData:
    """All repository data fetched in Phase 0."""
    owner: str
    repo: str
    readme: str = ""
    description: str = ""
    topics: list = field(default_factory=list)
    file_tree: list = field(default_factory=list)   # [{path, type, size}, ...]
    contributors: list = field(default_factory=list)  # [str login, ...]
    default_branch: str = "main"
    coral_queries_used: int = 0
    fetch_errors: list = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def validate_repo_relevance(
    repo_url: str,
    paper_json: dict,
    *,
    execution_check: bool = False,
) -> dict:
    """
    Validate that a GitHub repo is actually the implementation of the given paper.

    Args:
        repo_url:        GitHub repo URL (e.g. "https://github.com/user/repo")
        paper_json:      Parsed paper dict from Component 1. Expected fields:
                         'title', 'arxiv_id', 'authors', 'keywords', 'abstract'.
        execution_check: If True, run a lightweight dynamic sandbox check
                         (clones repo, attempts dry-run parse). Default: False.

    Returns:
        {
            # Core fields (backward-compatible)
            "relevance_score":             int   (0–100),
            "verdict":                     str   ("highly_confident" | "strong_match"
                                                   | "plausible" | "weak_match"
                                                   | "unrelated"),
            "evidence":                    list[str],
            "warnings":                    list[str],
            "checks_run":                  list[str],

            # Structured summary fields (new)
            "contains_code":               bool,
            "paper_match":                 str   ("weak" | "moderate" | "strong"),
            "implementation_completeness": str   ("low" | "medium" | "high"),
            "official_likelihood":         str   ("low" | "medium" | "high"),
            "overall_confidence_score":    int   (0–100),

            # Per-check score breakdown (new)
            "score_breakdown": {
                "title_similarity":    int,
                "readme_semantics":    int,
                "code_content":        int,
                "concept_alignment":   int,
                "completeness":        int,
                "relevance_ratio":     int,
                "execution_sanity":    int,
                "official_classifier": int,
            },

            # Detail objects (new)
            "file_stats":              dict,
            "official_classification": dict,
            "completeness_signals":    dict,
            "coral_queries_used":      int,
        }
    """
    result = _empty_result()

    if not repo_url:
        result["warnings"].append("No repo URL provided")
        return result

    owner, repo = _parse_github_url(repo_url)
    if not owner or not repo:
        result["warnings"].append(f"Could not parse GitHub URL: {repo_url}")
        return result

    # -----------------------------------------------------------------------
    # Phase 0: Fetch all repo data upfront (one pass, shared by all checks)
    # -----------------------------------------------------------------------
    repo_data = _fetch_repo_data(owner, repo)
    result["coral_queries_used"] = repo_data.coral_queries_used
    result["checks_run"].append("data_fetch")
    if repo_data.fetch_errors:
        result["warnings"].extend(repo_data.fetch_errors)

    # Extract paper concepts once — reused by Checks 5, 6, 7
    paper_concepts = _extract_paper_concepts(paper_json)

    breakdown = result["score_breakdown"]

    # -----------------------------------------------------------------------
    # Check 1: Repository Existence (gate — can trigger early exit)
    # -----------------------------------------------------------------------
    result["checks_run"].append("repo_existence")
    existence = _check_repo_existence(repo_data.file_tree)
    result["file_stats"] = existence["file_stats"]
    result["contains_code"] = existence["has_code"]
    result["evidence"].extend(existence["evidence"])

    if existence["is_trivially_empty"]:
        result["warnings"].append(
            "Repository appears empty or trivially small — no real code files found"
        )
        result["relevance_score"] = 5
        result["overall_confidence_score"] = 5
        result["verdict"] = "unrelated"
        return result

    if not existence["has_code"]:
        result["warnings"].append(
            "Repository has no recognised code files — likely documentation-only"
        )

    # -----------------------------------------------------------------------
    # Check 2: Paper-title / repo-title similarity
    # -----------------------------------------------------------------------
    result["checks_run"].append("title_similarity")
    title_res = _check_title_similarity(
        paper_json.get("title", ""),
        repo,
        repo_data.readme,
        repo_data.topics,
        repo_data.description,
    )
    breakdown["title_similarity"] = title_res["score"]
    result["evidence"].extend(title_res["evidence"])
    result["warnings"].extend(title_res["warnings"])

    # -----------------------------------------------------------------------
    # Check 3: README semantic validation
    # -----------------------------------------------------------------------
    result["checks_run"].append("readme_semantics")
    readme_res = _check_readme_semantics(repo_data.readme, paper_json, paper_concepts)
    breakdown["readme_semantics"] = readme_res["score"]
    result["evidence"].extend(readme_res["evidence"])
    result["warnings"].extend(readme_res["warnings"])

    # -----------------------------------------------------------------------
    # Check 4: Code-content validation
    # -----------------------------------------------------------------------
    result["checks_run"].append("code_content")
    code_res = _check_code_content(repo_data.file_tree, owner, repo)
    breakdown["code_content"] = code_res["score"]
    result["evidence"].extend(code_res["evidence"])
    result["warnings"].extend(code_res["warnings"])
    code_samples = code_res.get("code_samples", {})

    # -----------------------------------------------------------------------
    # Check 5: Paper-to-code concept alignment
    # -----------------------------------------------------------------------
    result["checks_run"].append("concept_alignment")
    align_res = _check_concept_alignment(paper_concepts, repo_data.file_tree, code_samples)
    breakdown["concept_alignment"] = align_res["score"]
    result["evidence"].extend(align_res["evidence"])
    result["warnings"].extend(align_res["warnings"])

    # -----------------------------------------------------------------------
    # Check 6: Completeness
    # -----------------------------------------------------------------------
    result["checks_run"].append("completeness")
    comp_res = _check_completeness(repo_data.file_tree, code_samples)
    breakdown["completeness"] = comp_res["score"]
    result["completeness_signals"] = comp_res["signals"]
    result["evidence"].extend(comp_res["evidence"])
    result["warnings"].extend(comp_res["warnings"])

    # -----------------------------------------------------------------------
    # Check 7: Relevance ratio by file inspection
    # -----------------------------------------------------------------------
    result["checks_run"].append("relevance_ratio")
    ratio_res = _check_relevance_ratio(repo_data.file_tree, paper_concepts)
    breakdown["relevance_ratio"] = ratio_res["score"]
    result["file_stats"].update(ratio_res["file_stats"])
    result["evidence"].extend(ratio_res["evidence"])

    # -----------------------------------------------------------------------
    # Check 8: Execution sanity (static always; dynamic opt-in)
    # -----------------------------------------------------------------------
    result["checks_run"].append("execution_sanity")
    exec_res = _check_execution_static(repo_data.file_tree, code_samples)
    if execution_check:
        dyn = _check_execution_dynamic(owner, repo)
        exec_res["score"] = min(
            exec_res["score"] + dyn["score_delta"], MAX_EXECUTION_SANITY
        )
        exec_res["evidence"].extend(dyn["evidence"])
        exec_res["warnings"].extend(dyn["warnings"])
    breakdown["execution_sanity"] = exec_res["score"]
    result["evidence"].extend(exec_res["evidence"])
    result["warnings"].extend(exec_res["warnings"])

    # -----------------------------------------------------------------------
    # Check 9: Official vs unofficial classifier
    # -----------------------------------------------------------------------
    result["checks_run"].append("official_classifier")
    official_res = _classify_official(
        owner, repo,
        repo_data.readme, repo_data.description,
        paper_json, repo_data.contributors,
    )
    breakdown["official_classifier"] = official_res["score"]
    result["official_classification"] = official_res["classification"]
    result["evidence"].extend(official_res["evidence"])
    result["warnings"].extend(official_res["warnings"])

    # -----------------------------------------------------------------------
    # Phase F: Compute final normalised score + derived labels
    # -----------------------------------------------------------------------
    raw_score = sum(breakdown.values())
    if not existence["has_code"]:
        raw_score = max(0, raw_score - 20)   # Penalty: no real code

    normalised = min(100, int(round(raw_score * 100 / RAW_TOTAL)))
    result["relevance_score"] = normalised
    result["overall_confidence_score"] = normalised
    result["verdict"] = _score_to_verdict(normalised)

    # Summary labels derived from sub-scores
    paper_match_raw = (
        breakdown["title_similarity"]
        + breakdown["readme_semantics"]
        + breakdown["concept_alignment"]
    )
    paper_match_max = MAX_TITLE_SIMILARITY + MAX_README_SEMANTICS + MAX_CONCEPT_ALIGNMENT
    result["paper_match"] = _score_to_label(paper_match_raw, paper_match_max)
    result["implementation_completeness"] = _score_to_label(
        breakdown["completeness"], MAX_COMPLETENESS
    )
    result["official_likelihood"] = official_res["classification"].get(
        "label_confidence", "low"
    )

    logger.info(
        "Repo relevance for %s/%s: %d/100 → %s",
        owner, repo, normalised, result["verdict"],
    )
    return result


# ---------------------------------------------------------------------------
# Phase 0: Data fetch (Coral SQL → GitHub API fallback)
# ---------------------------------------------------------------------------


def _fetch_repo_data(owner: str, repo: str) -> RepoData:
    """Fetch all repository data in one pass."""
    data = RepoData(owner=owner, repo=repo)
    try:
        from agents.coral_utils import get_coral_client
        coral = get_coral_client()
        if coral.available:
            _coral_fetch(coral, data)
            if data.readme or data.file_tree:
                return data
    except Exception as exc:
        logger.debug("Coral fetch failed: %s", exc)
        data.fetch_errors.append(f"Coral unavailable: {exc}")

    _github_api_fetch(data)
    return data


def _coral_fetch(coral, data: RepoData) -> None:
    """Populate RepoData using Coral SQL queries."""
    # Query 1: Repo metadata
    res = coral.sql(f"""
        SELECT description, default_branch, topics
        FROM github.repos_get
        WHERE owner = '{data.owner}' AND repo = '{data.repo}'
    """)
    data.coral_queries_used += 1
    if res and "results" in res and res["results"]:
        row = res["results"][0]
        data.description = row.get("description", "") or ""
        data.default_branch = row.get("default_branch", "main") or "main"
        topics = row.get("topics", [])
        if isinstance(topics, list):
            data.topics = topics
        elif isinstance(topics, str):
            data.topics = [t.strip() for t in topics.split(",") if t.strip()]

    # Query 2: README (try common names)
    for readme_name in ["README.md", "readme.md", "README.rst", "README"]:
        res = coral.sql(f"""
            SELECT content_text AS content
            FROM github.contents
            WHERE owner = '{data.owner}'
              AND repo  = '{data.repo}'
              AND path  = '{readme_name}'
        """)
        data.coral_queries_used += 1
        if res and "results" in res and res["results"]:
            content = res["results"][0].get("content", "")
            if content:
                data.readme = content
                break

    # Query 3: Recursive file tree
    res = coral.sql(f"""
        SELECT path, type, size
        FROM github.contents
        WHERE owner     = '{data.owner}'
          AND repo      = '{data.repo}'
          AND recursive = true
    """)
    data.coral_queries_used += 1
    if res and "results" in res:
        data.file_tree = res["results"]

    # Query 4: Contributors
    res = coral.sql(f"""
        SELECT login
        FROM github.contributors
        WHERE owner = '{data.owner}' AND repo = '{data.repo}'
        LIMIT 20
    """)
    data.coral_queries_used += 1
    if res and "results" in res:
        data.contributors = [
            r.get("login", "") for r in res["results"] if r.get("login")
        ]


def _github_api_fetch(data: RepoData) -> None:
    """Fallback: populate RepoData using the GitHub REST API."""
    try:
        import base64
        import requests

        headers: dict = {}
        token = os.environ.get("GITHUB_TOKEN", "")
        if token and "dummy" not in token.lower() and "your_" not in token.lower():
            headers["Authorization"] = f"token {token}"

        # Repo metadata
        resp = requests.get(
            f"https://api.github.com/repos/{data.owner}/{data.repo}",
            headers=headers, timeout=10,
        )
        if resp.status_code == 200:
            d = resp.json()
            data.description = d.get("description", "") or ""
            data.default_branch = d.get("default_branch", "main")
            data.topics = d.get("topics", [])

        # README
        resp = requests.get(
            f"https://api.github.com/repos/{data.owner}/{data.repo}/readme",
            headers=headers, timeout=10,
        )
        if resp.status_code == 200:
            d = resp.json()
            if d.get("encoding") == "base64":
                data.readme = base64.b64decode(d["content"]).decode("utf-8", errors="replace")
            elif d.get("download_url"):
                dl = requests.get(d["download_url"], timeout=10)
                if dl.status_code == 200:
                    data.readme = dl.text

        # Full file tree via Git Trees API (recursive=1)
        resp = requests.get(
            f"https://api.github.com/repos/{data.owner}/{data.repo}"
            f"/git/trees/{data.default_branch}?recursive=1",
            headers=headers, timeout=15,
        )
        if resp.status_code == 200:
            d = resp.json()
            data.file_tree = [
                {"path": item["path"], "type": item["type"], "size": item.get("size", 0)}
                for item in d.get("tree", [])
            ]

        # Contributors
        resp = requests.get(
            f"https://api.github.com/repos/{data.owner}/{data.repo}/contributors",
            headers=headers, timeout=10, params={"per_page": 20},
        )
        if resp.status_code == 200:
            data.contributors = [c.get("login", "") for c in resp.json() if c.get("login")]

    except Exception as exc:
        logger.debug("GitHub API fallback failed: %s", exc)
        data.fetch_errors.append(f"GitHub API failed: {exc}")


# ---------------------------------------------------------------------------
# Check 1: Repository Existence
# ---------------------------------------------------------------------------


def _check_repo_existence(file_tree: list) -> dict:
    """
    Confirm the repo is not empty or a trivial placeholder.
    Returns gate signals — no score value, but can trigger early exit.
    """
    all_paths = [
        item.get("path", "").lower()
        for item in file_tree
        if item.get("type") == "blob"
    ]
    code_files = [p for p in all_paths if _ext(p) in CODE_EXTENSIONS]
    substantial = [p for p in code_files if _file_size(p, file_tree) > 1024]

    non_trivial = [p for p in all_paths if os.path.basename(p) not in TRIVIAL_FILES]
    folders = {p.split("/")[0] for p in all_paths if "/" in p}
    meaningful_folders = folders & {
        "src", "source", "models", "model", "train", "training",
        "inference", "eval", "experiments", "scripts", "lib",
    }

    has_code = bool(code_files)
    is_trivially_empty = len(non_trivial) == 0 or (
        not code_files and len(all_paths) < 5
    )

    evidence = []
    if has_code:
        evidence.append(f"Repository contains {len(code_files)} code file(s)")
    if meaningful_folders:
        evidence.append(
            f"Meaningful folder structure detected: {', '.join(sorted(meaningful_folders))}"
        )

    return {
        "has_code": has_code,
        "is_trivially_empty": is_trivially_empty,
        "evidence": evidence,
        "file_stats": {
            "total_files": len(all_paths),
            "code_files": len(code_files),
            "substantial_code_files": len(substantial),
            "meaningful_folders": sorted(meaningful_folders),
        },
    }


# ---------------------------------------------------------------------------
# Check 2: Paper-title / Repo-title Similarity (enhanced)
# ---------------------------------------------------------------------------


def _check_title_similarity(
    title: str,
    repo_name: str,
    readme: str,
    topics: list,
    description: str,
) -> dict:
    """
    Compare paper title against repo name, README H1 heading, and topics/tags.
    Uses rapidfuzz + optional sentence-transformers embeddings.
    Returns score (0–30) and evidence.
    """
    score = 0
    evidence: list = []
    warnings: list = []

    if not title:
        warnings.append("No paper title provided for title similarity check")
        return {"score": 0, "evidence": evidence, "warnings": warnings}

    title_clean = _normalize_text(title).lower()
    abbreviation = _extract_abbreviation(title)

    # Sub-check A: title vs repo name (max 10 pts)
    repo_clean = repo_name.lower().replace("-", " ").replace("_", " ")
    ratio_a = _fuzzy_ratio(title_clean, repo_clean)
    if abbreviation and abbreviation.lower() in repo_clean:
        score += 10
        evidence.append(
            f"Paper abbreviation '{abbreviation}' found in repo name '{repo_name}' (+10)"
        )
    elif ratio_a >= 0.7:
        pts = int(10 * ratio_a)
        score += pts
        evidence.append(f"Paper title matches repo name (similarity: {ratio_a:.0%}) (+{pts})")
    elif ratio_a >= 0.4:
        score += 4
        evidence.append(f"Partial title match in repo name (similarity: {ratio_a:.0%}) (+4)")

    # Sub-check B: title vs README H1 headings (max 15 pts)
    h1_lines = [
        line.lstrip("#").strip()
        for line in readme.split("\n")
        if line.startswith("#") and not line.startswith("##")
    ]
    best_h1_ratio = max(
        (_fuzzy_ratio(title_clean, h1.lower()) for h1 in h1_lines),
        default=0.0,
    )
    # Embedding similarity (graceful fallback if library absent)
    emb_sim = _embedding_similarity(title, "\n".join(h1_lines)) if h1_lines else 0.0
    combined_h1 = max(best_h1_ratio, emb_sim)

    if combined_h1 >= 0.75:
        score += 15
        evidence.append(
            f"Paper title strongly matches README heading "
            f"(similarity: {combined_h1:.0%}) (+15)"
        )
    elif combined_h1 >= 0.5:
        score += 8
        evidence.append(
            f"Partial title match in README heading (similarity: {combined_h1:.0%}) (+8)"
        )
    elif not h1_lines:
        warnings.append("README has no H1 heading for title comparison")

    # Sub-check C: title vs topics / tags (max 5 pts)
    if topics:
        topic_text = " ".join(topics).lower().replace("-", " ")
        title_tokens = set(title_clean.split())
        topic_tokens = set(topic_text.split())
        overlap = len(title_tokens & topic_tokens) / max(len(title_tokens), 1)
        if abbreviation and abbreviation.lower() in topic_text:
            score += 5
            evidence.append(
                f"Paper abbreviation '{abbreviation}' found in repo topics (+5)"
            )
        elif overlap >= 0.3:
            pts = min(5, int(5 * overlap * 2))
            score += pts
            evidence.append(
                f"Paper title tokens overlap with repo topics ({overlap:.0%}) (+{pts})"
            )

    score = min(score, MAX_TITLE_SIMILARITY)
    return {"score": score, "evidence": evidence, "warnings": warnings}


# ---------------------------------------------------------------------------
# Check 3: README Semantic Validation (enhanced)
# ---------------------------------------------------------------------------


def _check_readme_semantics(
    readme: str, paper_json: dict, concepts: list
) -> dict:
    """
    Check whether the README genuinely references the paper.
    Returns score (0–15) and evidence.
    """
    score = 0
    evidence: list = []
    warnings: list = []

    if not readme:
        warnings.append("No README found — cannot perform semantic validation")
        return {"score": 0, "evidence": evidence, "warnings": warnings}

    readme_lower = readme.lower()
    title    = paper_json.get("title", "")
    arxiv_id = paper_json.get("arxiv_id", "")
    authors  = paper_json.get("authors", [])

    # Signal 1: arXiv ID present (max 5 pts)
    if arxiv_id:
        patterns = [
            re.escape(arxiv_id),
            rf"arxiv[:\s]*{re.escape(arxiv_id)}",
            rf"arxiv\.org/abs/{re.escape(arxiv_id)}",
        ]
        for pat in patterns:
            if re.search(pat, readme, re.IGNORECASE):
                score += 5
                evidence.append(f"arXiv ID '{arxiv_id}' found in README (+5)")
                break
        else:
            warnings.append(f"arXiv ID '{arxiv_id}' not found in README")

    # Signal 2: Title or abbreviation (max 3 pts)
    if title:
        abbrev = _extract_abbreviation(title)
        if _normalize_text(title).lower() in readme_lower:
            score += 3
            evidence.append("Full paper title found in README (+3)")
        elif abbrev and len(abbrev) >= 3 and abbrev.lower() in readme_lower:
            score += 2
            evidence.append(f"Paper abbreviation '{abbrev}' found in README (+2)")

    # Signal 3: Author names (max 3 pts)
    matched_authors = []
    for author in authors:
        if isinstance(author, dict):
            last = (author.get("last", "") or "").strip()
        elif isinstance(author, str):
            parts = author.strip().split()
            last = parts[-1] if parts else ""
        else:
            continue
        if last and len(last) >= 2 and last.lower() in readme_lower:
            matched_authors.append(last)
    if len(matched_authors) >= 2:
        score += 3
        evidence.append(
            f"Author names found in README: {', '.join(matched_authors)} (+3)"
        )
    elif len(matched_authors) == 1:
        score += 1
        evidence.append(f"Author name found in README: {matched_authors[0]} (+1)")

    # Signal 4: "Official implementation" claim phrases (max 2 pts)
    for phrase in OFFICIAL_PHRASES:
        if phrase in readme_lower:
            score += 2
            evidence.append(
                f"Implementation claim phrase detected in README: '{phrase}' (+2)"
            )
            break

    # Signal 5: Key paper concepts mentioned in README (max 2 pts)
    concept_hits = sum(1 for c in concepts if c.lower() in readme_lower)
    if concept_hits >= 3:
        score += 2
        evidence.append(f"README mentions {concept_hits} key paper concepts (+2)")
    elif concept_hits >= 1:
        score += 1
        evidence.append(f"README mentions {concept_hits} key paper concept(s) (+1)")

    score = min(score, MAX_README_SEMANTICS)
    return {"score": score, "evidence": evidence, "warnings": warnings}


# ---------------------------------------------------------------------------
# Check 4: Code-Content Validation
# ---------------------------------------------------------------------------


def _check_code_content(file_tree: list, owner: str, repo: str) -> dict:
    """
    Check whether the repo has real, executable source code.
    Returns score (0–40), evidence, and sampled code content dict.
    """
    score = 0
    evidence: list = []
    warnings: list = []
    code_samples: dict = {}

    all_paths = [item.get("path", "") for item in file_tree if item.get("type") == "blob"]
    basenames  = {os.path.basename(p).lower(): p for p in all_paths}
    code_paths = [p for p in all_paths if _ext(p.lower()) in CODE_EXTENSIONS]

    # Signal A: Substantial code files > 1 KB (max 10 pts)
    substantial = [p for p in code_paths if _file_size(p, file_tree) > 1024]
    if len(substantial) >= 5:
        score += 10
        evidence.append(f"{len(substantial)} substantial code files found (>1 KB) (+10)")
    elif len(substantial) >= 1:
        pts = max(4, len(substantial) * 2)
        score += pts
        evidence.append(f"{len(substantial)} substantial code file(s) found (+{pts})")
    elif code_paths:
        score += 2
        evidence.append(f"{len(code_paths)} code file(s) found (small/stub) (+2)")
    else:
        warnings.append("No code files found in repository")
        return {"score": 0, "evidence": evidence, "warnings": warnings, "code_samples": {}}

    # Signal B: Training entrypoint (max 8 pts)
    train_matches = [p for b, p in basenames.items() if b in ENTRYPOINT_NAMES]
    if train_matches:
        score += 8
        evidence.append(f"Training entrypoint found: {', '.join(train_matches[:2])} (+8)")
        code_samples.update(_fetch_code_samples(train_matches[:1], owner, repo))

    # Signal C: Inference / eval script (max 5 pts)
    infer_matches = [p for b, p in basenames.items() if b in INFERENCE_NAMES]
    if infer_matches:
        score += 5
        evidence.append(f"Inference/eval script found: {', '.join(infer_matches[:2])} (+5)")
        code_samples.update(_fetch_code_samples(infer_matches[:1], owner, repo))

    # Signal D: Model definition (max 8 pts)
    model_matches = [p for b, p in basenames.items() if b in MODEL_NAMES]
    model_dirs    = [
        p for p in all_paths
        if p.lower().startswith(("models/", "model/", "src/model"))
        and _ext(p.lower()) in CODE_EXTENSIONS
    ]
    if model_matches or model_dirs:
        score += 8
        combined = (model_matches + model_dirs)[:2]
        evidence.append(f"Model definition found: {', '.join(combined)} (+8)")
        code_samples.update(_fetch_code_samples((model_matches + model_dirs)[:1], owner, repo))

    # Signal E: Dataset loader (max 5 pts)
    data_base_patterns = {"dataset.py", "dataloader.py", "data_loader.py", "data_utils.py"}
    data_matches = [p for b, p in basenames.items() if b in data_base_patterns]
    data_dir_files = [
        p for p in all_paths
        if p.lower().startswith("data/") and _ext(p.lower()) in CODE_EXTENSIONS
    ]
    if data_matches or data_dir_files:
        score += 5
        evidence.append("Dataset loader / data directory found (+5)")

    # Signal F: Config file (max 4 pts)
    config_matches = [p for b, p in basenames.items() if b in CONFIG_NAMES]
    if config_matches:
        score += 4
        evidence.append(f"Config file found: {config_matches[0]} (+4)")

    score = min(score, MAX_CODE_CONTENT)
    return {
        "score": score,
        "evidence": evidence,
        "warnings": warnings,
        "code_samples": code_samples,
    }


def _fetch_code_samples(paths: list, owner: str, repo: str) -> dict:
    """Fetch raw content of a few key files for deeper analysis (capped at 8 KB each)."""
    samples: dict = {}
    try:
        from agents.coral_utils import get_coral_client
        coral = get_coral_client()
        if not coral.available:
            raise RuntimeError("Coral not available")
        for path in paths[:3]:
            res = coral.sql(f"""
                SELECT content_text AS content
                FROM github.contents
                WHERE owner = '{owner}' AND repo = '{repo}' AND path = '{path}'
            """)
            if res and "results" in res and res["results"]:
                content = res["results"][0].get("content", "")
                if content:
                    samples[path] = content[:8000]
    except Exception as exc:
        logger.debug("Code sample fetch failed: %s", exc)
    return samples


# ---------------------------------------------------------------------------
# Check 5: Paper-to-Code Concept Alignment
# ---------------------------------------------------------------------------


def _check_concept_alignment(
    concepts: list, file_tree: list, code_samples: dict
) -> dict:
    """
    Check whether the paper's unique technical vocabulary appears in the codebase
    (filenames, function/class names, comments).
    Returns score (0–20) and evidence.
    """
    score = 0
    evidence: list = []
    warnings: list = []

    if not concepts:
        warnings.append("No paper concepts extracted for alignment check")
        return {"score": 0, "evidence": evidence, "warnings": warnings}

    # Build a searchable corpus from filenames + code symbols + comments
    filenames_blob = " ".join(
        os.path.basename(item.get("path", "")).lower().replace("_", " ").replace("-", " ")
        for item in file_tree
    )

    symbol_parts: list = []
    for content in code_samples.values():
        symbol_parts.extend(re.findall(r"\bdef\s+(\w+)", content))
        symbol_parts.extend(re.findall(r"\bclass\s+(\w+)", content))
        symbol_parts.extend(re.findall(r"#\s*(.+)", content))
    symbols_blob = " ".join(symbol_parts).lower()

    corpus = filenames_blob + " " + symbols_blob

    matched: list = []
    for concept in concepts:
        concept_lower = concept.lower()
        if concept_lower in corpus:
            matched.append(concept)
        else:
            # Try individual meaningful words in the concept
            words = [w for w in concept_lower.split() if len(w) >= 4]
            if words and any(w in corpus for w in words):
                matched.append(concept)

    hit_ratio = len(matched) / max(len(concepts), 1)

    if hit_ratio >= 0.6:
        score = 20
        evidence.append(
            f"High concept alignment: {len(matched)}/{len(concepts)} paper concepts "
            f"found in codebase ({hit_ratio:.0%}) (+20)"
        )
    elif hit_ratio >= 0.4:
        score = 12
        evidence.append(
            f"Moderate concept alignment: {len(matched)}/{len(concepts)} "
            f"concepts found ({hit_ratio:.0%}) (+12)"
        )
    elif hit_ratio >= 0.2:
        score = 6
        evidence.append(
            f"Low concept alignment: {len(matched)}/{len(concepts)} "
            f"concepts found ({hit_ratio:.0%}) (+6)"
        )
    else:
        warnings.append(
            f"Weak concept alignment: only {len(matched)}/{len(concepts)} "
            "paper concepts found in codebase"
        )

    return {"score": score, "evidence": evidence, "warnings": warnings}


# ---------------------------------------------------------------------------
# Check 6: Completeness
# ---------------------------------------------------------------------------


def _check_completeness(file_tree: list, code_samples: dict) -> dict:
    """
    Assess whether the implementation appears complete and not stubbed out.
    Returns score (0–20), signals dict, and evidence.
    """
    score = 0
    evidence: list = []
    warnings: list = []
    signals: dict = {}

    all_paths = [item.get("path", "") for item in file_tree if item.get("type") == "blob"]
    basenames  = {os.path.basename(p).lower() for p in all_paths}
    folder_set = {p.split("/")[0].lower() for p in all_paths if "/" in p}

    # A: Training entrypoint (4 pts)
    signals["has_training_entrypoint"] = bool(basenames & ENTRYPOINT_NAMES)
    if signals["has_training_entrypoint"]:
        score += 4
        evidence.append("Training entrypoint present (+4)")

    # B: Inference entrypoint (4 pts)
    signals["has_inference_entrypoint"] = bool(basenames & INFERENCE_NAMES)
    if signals["has_inference_entrypoint"]:
        score += 4
        evidence.append("Inference / eval entrypoint present (+4)")

    # C: Dependency declaration files (3 pts)
    signals["has_dependency_files"] = bool(basenames & DEPENDENCY_FILES)
    if signals["has_dependency_files"]:
        score += 3
        evidence.append("Dependency file(s) present (requirements.txt / setup.py / etc.) (+3)")

    # D: Config files (2 pts)
    signals["has_config_files"] = bool(basenames & CONFIG_NAMES)
    if signals["has_config_files"]:
        score += 2
        evidence.append("Config file(s) present (+2)")

    # E: Results / experiments / checkpoints folder (2 pts)
    results_dirs = {"results", "experiments", "outputs", "logs", "checkpoints", "runs"}
    signals["has_results_folder"] = bool(folder_set & results_dirs)
    if signals["has_results_folder"]:
        score += 2
        evidence.append("Results / experiments folder present (+2)")

    # F: No TODO / stub / placeholder patterns (3 pts; penalised if found)
    stub_count = sum(
        len(re.findall(pat, content, re.IGNORECASE | re.MULTILINE))
        for content in code_samples.values()
        for pat in STUB_PATTERNS
    )
    signals["stub_count"] = stub_count
    if stub_count == 0:
        score += 3
        evidence.append("No TODO / stub / placeholder patterns detected in sampled code (+3)")
    elif stub_count <= 3:
        score += 1
        warnings.append(f"Minor stubs detected ({stub_count} occurrence(s))")
    else:
        warnings.append(
            f"Significant stub / TODO count ({stub_count}) — implementation may be incomplete"
        )

    # G: Average code file size (2 pts)
    code_items = [
        item for item in file_tree
        if _ext(item.get("path", "").lower()) in CODE_EXTENSIONS
    ]
    avg_size = (
        sum(item.get("size", 0) or 0 for item in code_items) / len(code_items)
        if code_items else 0
    )
    signals["avg_code_file_size_bytes"] = int(avg_size)
    if avg_size > 5000:
        score += 2
        evidence.append(f"Code files are substantial (avg {int(avg_size/1024)} KB) (+2)")
    elif avg_size > 1000:
        score += 1
        evidence.append("Code files are moderately sized (+1)")

    score = min(score, MAX_COMPLETENESS)
    return {"score": score, "evidence": evidence, "warnings": warnings, "signals": signals}


# ---------------------------------------------------------------------------
# Check 7: Relevance Ratio by File Inspection
# ---------------------------------------------------------------------------


def _check_relevance_ratio(file_tree: list, concepts: list) -> dict:
    """
    Classify code files (core / utility / junk / third-party) and compute a
    relevance ratio.  Returns score (0–10) and evidence.
    """
    all_paths  = [item.get("path", "") for item in file_tree if item.get("type") == "blob"]
    code_paths = [p for p in all_paths if _ext(p.lower()) in CODE_EXTENSIONS]

    _CORE = re.compile(
        r"\b(model|train|loss|network|arch|module|layer|encoder|decoder|head|"
        r"attention|transformer|conv|rnn|lstm|gnn|diffus|vae|gan|embed)\b",
        re.I,
    )
    _UTIL = re.compile(
        r"\b(util|helper|io|logger|metric|eval|hook|callback|viz|plot|visual)\b",
        re.I,
    )
    _JUNK = re.compile(
        r"(test_|__pycache__|\.egg|node_modules|\.pyc|setup\.cfg|\.github)",
        re.I,
    )
    _THIRD = re.compile(
        r"(vendor/|third.party/|extern/|external/|thirdparty/)",
        re.I,
    )

    concept_tokens = {
        t for c in concepts for t in c.lower().split() if len(t) >= 4
    }

    core, utility, junk, third_party = [], [], [], []

    for path in code_paths:
        bname = os.path.basename(path).lower()
        if _THIRD.search(path):
            third_party.append(path)
        elif _JUNK.search(path):
            junk.append(path)
        elif _CORE.search(bname) or any(t in bname for t in concept_tokens):
            core.append(path)
        else:
            utility.append(path)

    usable = len(code_paths) - len(junk) - len(third_party)
    ratio  = len(core) / max(usable, 1)
    score  = min(MAX_RELEVANCE_RATIO, int(ratio * MAX_RELEVANCE_RATIO * 1.5))

    evidence: list = []
    if ratio >= 0.4:
        evidence.append(
            f"Good relevance ratio: {len(core)}/{usable} code files are core "
            f"implementation ({ratio:.0%}) (+{score})"
        )
    elif ratio >= 0.2:
        evidence.append(f"Moderate relevance ratio ({ratio:.0%}) (+{score})")

    return {
        "score": score,
        "evidence": evidence,
        "file_stats": {
            "core_files":        len(core),
            "utility_files":     len(utility),
            "junk_files":        len(junk),
            "third_party_files": len(third_party),
            "relevance_ratio":   round(ratio, 2),
        },
    }


# ---------------------------------------------------------------------------
# Check 8: Execution Sanity
# ---------------------------------------------------------------------------


def _check_execution_static(file_tree: list, code_samples: dict) -> dict:
    """
    Static analysis only (no cloning).  Always runs.
    Returns score (0–5) and evidence.
    """
    score = 0
    evidence: list = []
    warnings: list = []

    # A: Successfully parse at least one code file with the ast module (2 pts)
    for path, content in code_samples.items():
        if not content:
            continue
        try:
            tree = ast.parse(content)
            imports = {
                node.names[0].name.split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.Import)
            }
            from_imports = {
                (node.module or "").split(".")[0]
                for node in ast.walk(tree)
                if isinstance(node, ast.ImportFrom)
            }
            meaningful = (imports | from_imports) - {
                "", "os", "sys", "re", "math", "json", "time", "logging", "typing",
            }
            if meaningful:
                score += 2
                evidence.append(
                    f"'{os.path.basename(path)}' has valid imports: "
                    f"{', '.join(list(meaningful)[:5])} (+2)"
                )
                break
        except SyntaxError as exc:
            warnings.append(
                f"Syntax error detected in '{os.path.basename(path)}': {exc}"
            )

    # B: Runnable __main__ block present (3 pts)
    for path, content in code_samples.items():
        if "if __name__" in content or "__main__" in content:
            score += 3
            evidence.append(
                f"Runnable __main__ block found in '{os.path.basename(path)}' (+3)"
            )
            break

    score = min(score, 5)   # Static tier: max 5 pts
    return {"score": score, "evidence": evidence, "warnings": warnings}


def _check_execution_dynamic(owner: str, repo: str) -> dict:
    """
    Dynamic execution check (opt-in, disabled by default).
    Shallow-clones the repo to a temp dir and attempts to parse the main Python
    file.  Returns score_delta (–5 to +5) and evidence.
    """
    evidence: list = []
    warnings: list = []
    score_delta = 0

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            clone = subprocess.run(
                ["git", "clone", "--depth=1",
                 f"https://github.com/{owner}/{repo}.git", tmpdir],
                capture_output=True, text=True, timeout=60,
            )
            if clone.returncode != 0:
                warnings.append(
                    f"Could not clone repo for dynamic check: "
                    f"{clone.stderr[:200]}"
                )
                return {"score_delta": 0, "evidence": evidence, "warnings": warnings}

            evidence.append("Repo cloned successfully for dynamic check")

            py_files = [f for f in os.listdir(tmpdir) if f.endswith(".py")]
            if py_files:
                target = os.path.join(tmpdir, py_files[0])
                parse_run = subprocess.run(
                    ["python", "-c",
                     f"import ast; ast.parse(open(r'{target}').read())"],
                    capture_output=True, text=True, timeout=30,
                )
                if parse_run.returncode == 0:
                    score_delta = 5
                    evidence.append(
                        f"'{py_files[0]}' parses cleanly (dynamic check) (+5)"
                    )
                else:
                    score_delta = -5
                    warnings.append(
                        f"Dynamic check: '{py_files[0]}' has parse errors"
                    )

    except subprocess.TimeoutExpired:
        warnings.append("Dynamic execution check timed out (60 s)")
    except Exception as exc:
        warnings.append(f"Dynamic execution check failed: {exc}")

    return {"score_delta": score_delta, "evidence": evidence, "warnings": warnings}


# ---------------------------------------------------------------------------
# Check 9: Official vs Unofficial Classifier
# ---------------------------------------------------------------------------


def _classify_official(
    owner: str,
    repo: str,
    readme: str,
    description: str,
    paper_json: dict,
    contributors: list,
) -> dict:
    """
    Classify the repo as official / unofficial / unclear.
    Returns score (0–10), classification dict, evidence, and warnings.
    """
    score = 0
    evidence: list = []
    warnings: list = []
    signals_for: list = []
    signals_against: list = []

    readme_lower = (readme + " " + description).lower()
    authors = paper_json.get("authors", [])

    # Official: "official implementation" phrases (max 4 pts)
    for phrase in OFFICIAL_PHRASES:
        if phrase in readme_lower:
            signals_for.append(f"README contains: '{phrase}'")
            score += 4
            evidence.append(f"Official implementation phrase detected: '{phrase}' (+4)")
            break

    # Official: arXiv link in README (max 2 pts)
    arxiv_id = paper_json.get("arxiv_id", "")
    if arxiv_id and arxiv_id in readme_lower:
        signals_for.append(f"arXiv ID '{arxiv_id}' present in README")
        score += 2
        evidence.append("Paper arXiv link present in README (+2)")

    # Official: owner name matches author / affiliation (max 3 pts)
    owner_lower = owner.lower()
    author_tokens: list = []
    for author in authors:
        if isinstance(author, dict):
            author_tokens.append((author.get("last", "") or "").lower())
            author_tokens.append((author.get("affiliation", "") or "").lower())
        elif isinstance(author, str):
            parts = author.strip().split()
            author_tokens.append(parts[-1].lower() if parts else "")
    if any(t and t in owner_lower for t in author_tokens):
        signals_for.append(f"Repo owner '{owner}' matches an author / affiliation")
        score += 3
        evidence.append("Repo owner matches paper author / affiliation (+3)")

    # Official: known research lab / org (max 1 pt)
    known_labs = {
        "google", "deepmind", "openai", "meta", "facebook", "microsoft",
        "huggingface", "stanford", "mit", "berkeley", "cmu", "oxford",
        "cambridge", "alibaba", "baidu", "nvidia", "aws", "apple",
    }
    if owner_lower in known_labs or any(lab in owner_lower for lab in known_labs):
        signals_for.append(f"Owner '{owner}' is a known research lab / org")
        score += 1
        evidence.append("Repo owned by a known research organisation (+1)")

    # Unofficial: wording signals (penalty –2)
    for phrase in UNOFFICIAL_PHRASES:
        if phrase in readme_lower:
            signals_against.append(f"README contains: '{phrase}'")
            score = max(0, score - 2)
            warnings.append(f"Unofficial signal detected in README: '{phrase}'")
            break

    # Derive label
    if score >= 6:
        label, conf = "official", "high"
    elif score >= 3:
        label, conf = "unclear", "medium"
    else:
        label, conf = "unofficial", "low"

    score = min(score, MAX_OFFICIAL)

    classification = {
        "label":            label,
        "confidence":       conf,
        "label_confidence": conf,    # used for official_likelihood top-level field
        "signals_for":      signals_for,
        "signals_against":  signals_against,
    }

    return {
        "score":          score,
        "classification": classification,
        "evidence":       evidence,
        "warnings":       warnings,
    }


# ---------------------------------------------------------------------------
# Concept extraction helpers
# ---------------------------------------------------------------------------


def _extract_paper_concepts(paper_json: dict) -> list:
    """
    Extract key technical concepts from paper JSON.
    Returns a deduplicated list (max 30 items).
    """
    concepts: set = set()

    title    = paper_json.get("title", "")
    abstract = paper_json.get("abstract", "")
    keywords = paper_json.get("keywords", [])

    # From explicit keywords field
    for kw in keywords:
        if isinstance(kw, str) and len(kw) >= 3:
            concepts.add(kw.strip().lower())

    # Model / acronym names from title (CamelCase or ALL_CAPS)
    for word in re.findall(r"\b[A-Z][A-Za-z0-9]{1,12}\b", title):
        if len(word) >= 3:
            concepts.add(word.lower())

    # From abstract: multi-word noun phrases + frequent content words
    if abstract:
        # Capitalised 2–3 word phrases (likely proper names / model names)
        for np in re.findall(r"\b(?:[A-Z][a-z]+\s+){1,2}[A-Z][a-z]+\b", abstract):
            if len(np) >= 6:
                concepts.add(np.lower())

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
        words = [re.sub(r"[^a-z0-9]", "", w) for w in abstract.lower().split()]
        words = [w for w in words if w not in stop_words and len(w) > 4]
        freq: dict = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        for w, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)[:10]:
            concepts.add(w)

    # Known benchmark / dataset names
    text_lower = (title + " " + abstract).lower()
    for ds in KNOWN_DATASETS:
        if ds in text_lower:
            concepts.add(ds)

    return list(concepts)[:30]


def _extract_abbreviation(title: str) -> str:
    """
    Extract the primary model abbreviation from a paper title.

    Examples:
        "BERT: Pre-training..."   → "BERT"
        "Attention Is All You Need" → "AIAYN"
    """
    # Leading named acronym (e.g. "BERT:", "ViT:", "GPT-4:")
    m = re.match(r"^([A-Z][A-Za-z0-9]{1,12})\b", title)
    if m:
        candidate = m.group(1)
        if len(candidate) >= 2:
            return candidate

    # All-caps words anywhere in title
    caps_words = re.findall(r"\b[A-Z]{2,10}\b", title)
    if caps_words:
        return caps_words[0]

    # Initials from meaningful title words
    skip = {"the", "a", "an", "of", "in", "on", "for", "and", "with", "via", "is"}
    words = [w for w in title.split() if len(w) > 1 and w.lower() not in skip]
    if len(words) >= 3:
        return "".join(w[0] for w in words[:8]).upper()

    return ""


# ---------------------------------------------------------------------------
# Fuzzy & embedding helpers
# ---------------------------------------------------------------------------


def _fuzzy_ratio(a: str, b: str) -> float:
    """Compute fuzzy similarity (0.0–1.0). Uses rapidfuzz if available."""
    try:
        from rapidfuzz import fuzz
        return max(
            fuzz.partial_ratio(a, b) / 100.0,
            fuzz.token_sort_ratio(a, b) / 100.0,
        )
    except ImportError:
        words_a = set(a.split())
        words_b = set(b.split())
        if not words_a:
            return 0.0
        return len(words_a & words_b) / len(words_a)


# Module-level cache for the sentence-transformer model
_sentence_model = None


def _embedding_similarity(text_a: str, text_b: str) -> float:
    """
    Compute cosine similarity between two texts using sentence-transformers.
    Uses 'all-MiniLM-L6-v2' (small, fast, ~80 MB).
    Returns 0.0 gracefully if the library is not installed.
    """
    if not text_a or not text_b:
        return 0.0
    try:
        import numpy as np
        from sentence_transformers import SentenceTransformer

        global _sentence_model
        if _sentence_model is None:
            _sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

        embs = _sentence_model.encode([text_a, text_b], normalize_embeddings=True)
        return float(np.dot(embs[0], embs[1]))
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Scoring / label helpers
# ---------------------------------------------------------------------------


def _score_to_verdict(score: int) -> str:
    if score >= BAND_HIGHLY_CONFIDENT:
        return "highly_confident"
    if score >= BAND_STRONG_MATCH:
        return "strong_match"
    if score >= BAND_PLAUSIBLE:
        return "plausible"
    if score >= BAND_WEAK_MATCH:
        return "weak_match"
    return "unrelated"


def _score_to_label(raw: int, max_raw: int) -> str:
    """Map a raw/max pair to 'low' / 'medium' / 'high'."""
    ratio = raw / max(max_raw, 1)
    if ratio >= 0.65:
        return "high"
    if ratio >= 0.35:
        return "medium"
    return "low"


def _empty_result() -> dict:
    return {
        "relevance_score":             0,
        "verdict":                     "unrelated",
        "contains_code":               False,
        "paper_match":                 "weak",
        "implementation_completeness": "low",
        "official_likelihood":         "low",
        "overall_confidence_score":    0,
        "score_breakdown": {
            "title_similarity":    0,
            "readme_semantics":    0,
            "code_content":        0,
            "concept_alignment":   0,
            "completeness":        0,
            "relevance_ratio":     0,
            "execution_sanity":    0,
            "official_classifier": 0,
        },
        "evidence":               [],
        "warnings":               [],
        "checks_run":             [],
        "file_stats":             {},
        "official_classification":{},
        "completeness_signals":   {},
        "coral_queries_used":     0,
    }


# ---------------------------------------------------------------------------
# Misc helpers (unchanged from v1)
# ---------------------------------------------------------------------------


def _normalize_text(text: str) -> str:
    """Collapse whitespace and strip."""
    return re.sub(r"\s+", " ", text).strip()


def _parse_github_url(url: str):
    """Extract (owner, repo) from a GitHub URL."""
    if not url:
        return None, None
    for pattern in [
        r"github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$",
        r"github\.com/([^/]+)/([^/]+?)(?:/.*)?$",
    ]:
        m = re.search(pattern, url)
        if m:
            return m.group(1), m.group(2)
    return None, None


def _ext(path: str) -> str:
    """Return the lowercased file extension."""
    return os.path.splitext(path)[1].lower()


def _file_size(path: str, file_tree: list) -> int:
    """Look up a file's size from the cached file tree."""
    path_lower = path.lower()
    for item in file_tree:
        if item.get("path", "").lower() == path_lower:
            return item.get("size", 0) or 0
    return 0


# ---------------------------------------------------------------------------
# Formatting helper (updated verdict labels)
# ---------------------------------------------------------------------------


def format_relevance_badge(result: dict) -> str:
    """Return a human-readable relevance badge string."""
    score   = result.get("relevance_score", 0)
    verdict = result.get("verdict", "unrelated")
    badges  = {
        "highly_confident": f"🎯 Highly Confident ({score}/100)",
        "strong_match":     f"✅ Strong Match ({score}/100)",
        "plausible":        f"🟡 Plausible Match ({score}/100)",
        "weak_match":       f"⚠️  Weak Match ({score}/100)",
        "unrelated":        f"❌ Unrelated ({score}/100)",
    }
    return badges.get(verdict, f"❓ Unknown ({score}/100)")


# ---------------------------------------------------------------------------
# CLI / standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)

    test_paper = {
        "title": "Attention Is All You Need",
        "arxiv_id": "1706.03762",
        "authors": [
            {"first": "Ashish", "last": "Vaswani"},
            {"first": "Noam",   "last": "Shazeer"},
        ],
        "keywords": ["transformer", "attention", "self-attention", "encoder", "decoder"],
        "abstract": (
            "The dominant sequence transduction models are based on complex recurrent "
            "or convolutional neural networks that include an encoder and a decoder. "
            "We propose a new simple network architecture, the Transformer, based solely "
            "on attention mechanisms, dispensing with recurrence and convolutions entirely."
        ),
    }

    result = validate_repo_relevance(
        "https://github.com/tensorflow/tensor2tensor",
        test_paper,
    )
    print(json.dumps(result, indent=2))
    print(f"\n{format_relevance_badge(result)}")
