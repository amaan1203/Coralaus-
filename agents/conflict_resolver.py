"""
Component 5 — Conflict Resolution & Dockerfile Generation

Resolves dependency conflicts and generates a working Dockerfile.
Uses Groq (Llama 3.3 70B) for speed.

Pipeline:
  5a. HyDE Draft Generation    — hypothetical Dockerfile for keyword extraction
  5b. HyDE Keyword Extraction  — extract search terms from draft
  5c. Coral RAG Search         — fetch 3-4 real Dockerfiles from GitHub
  5d. FLARE Active Retrieval   — targeted search for conflict-specific snippets
  5e. Base Image Selection     — dedicated LLM call to pick & justify the base image
  5f. Dockerfile Generation    — full Dockerfile with selected base injected
  5g. Self-Healing Validation  — fix any structural issues found by validator

Input:  Original requirements + conflict list from Component 4
Output: Dockerfile and/or environment.yaml
"""

import os
import json
import logging
import re
from typing import Optional
from agents.coral_utils import get_coral_client

logger = logging.getLogger(__name__)


def resolve_conflicts(requirements_content: str, conflicts: list, paper_title: str = "",
                      entrypoint: str = None, readme_content: str = "",
                      dep_files: dict = None, repo_year: str = None) -> dict:
    """
    Resolve dependency conflicts and generate a working Dockerfile.

    Args:
        requirements_content: Original requirements.txt content
        conflicts: List of conflict descriptions from Component 4
        paper_title: Paper title for Dockerfile comments
        entrypoint: Detected entrypoint script (e.g. 'main.py', 'train.py').
                    If None, the Dockerfile will use a generic COPY without a CMD.
        readme_content: README.md content for build instructions
        dep_files: All dependency files found {filename: content}
        repo_year: Year the repository was created/last active (for temporal pinning)

    Returns:
        Dict with:
            - dockerfile (str): Generated Dockerfile content
            - resolved_requirements (str): Fixed requirements.txt
            - environment_yaml (str): Optional conda environment
            - resolution_notes (list): What was changed and why
            - selected_base_image (str): Base image chosen by agent
    """
    if dep_files is None:
        dep_files = {}

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set, generating basic Dockerfile")
        return _basic_dockerfile(requirements_content, paper_title, entrypoint)

    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        # 5a. HyDE Draft Generation
        hyde_draft = _generate_hyde_draft(client, requirements_content, conflicts)

        # 5b. HyDE Keyword Extraction
        keywords = _extract_hyde_keywords(client, hyde_draft)
        logger.info(f"Extracted HyDE keywords: {keywords}")

        # 5c. Coral RAG Search — fetch real Dockerfiles from GitHub
        rag_dockerfiles = _search_github_dockerfiles(keywords, dep_files=dep_files)

        # 5d. FLARE Active Retrieval — targeted conflict-specific snippets
        flare_contexts = _flare_active_retrieval(client, conflicts)

        # 5e. Dedicated base image selection (NEW — separated from full generation)
        selected = _select_base_image(
            client,
            dep_files=dep_files,
            conflicts=conflicts,
            repo_year=repo_year,
            rag_dockerfiles=rag_dockerfiles,
        )
        selected_base_image = selected.get("base_image", "")
        logger.info(f"Agent selected base image: {selected_base_image} — {selected.get('reason', '')}")

        # 5f. Generate Dockerfile with selected base + full context
        dockerfile = _generate_dockerfile(
            client, requirements_content, conflicts, paper_title, entrypoint,
            rag_dockerfiles=rag_dockerfiles, flare_contexts=flare_contexts,
            readme_content=readme_content, dep_files=dep_files, repo_year=repo_year,
            selected_base_image=selected_base_image,
        )

        # 5g. Self-healing validation — pass selected_base_image so validator respects agent's choice
        from agents.dockerfile_validator import validate_dockerfile
        validation = validate_dockerfile(dockerfile, requirements_content, suggested_base_image=selected_base_image)
        if not validation["valid"]:
            logger.warning(f"Dockerfile validation found {len(validation['issues'])} issues: {validation['issues']}")
            dockerfile = _self_heal_dockerfile(
                client, dockerfile, validation["issues"],
                requirements_content, dep_files, readme_content,
                selected_base_image=selected_base_image,
            )

        # Generate resolved requirements
        resolved_reqs = _resolve_requirements(
            client, requirements_content, conflicts,
            rag_dockerfiles=rag_dockerfiles, flare_contexts=flare_contexts,
            repo_year=repo_year
        )

        return {
            "dockerfile": dockerfile,
            "resolved_requirements": resolved_reqs.get("requirements", requirements_content),
            "environment_yaml": None,
            "resolution_notes": resolved_reqs.get("notes", []),
            "llm_model": "llama-3.3-70b-versatile",
            "selected_base_image": selected_base_image,
            "base_image_reason": selected.get("reason", ""),
        }

    except ImportError:
        logger.warning("groq package not installed, generating basic Dockerfile")
        return _basic_dockerfile(requirements_content, paper_title, entrypoint)
    except Exception as e:
        logger.error(f"Conflict resolution failed: {e}")
        return _basic_dockerfile(requirements_content, paper_title, entrypoint)


# ---------------------------------------------------------------------------
# 5a. HyDE Draft Generation
# ---------------------------------------------------------------------------

def _generate_hyde_draft(client, requirements: str, conflicts: list) -> str:
    """Generate a hypothetical draft Dockerfile for HyDE keyword extraction."""
    prompt = f"""You are a DevOps assistant. Generate a short hypothetical Dockerfile based on these requirements and conflicts.
This is for keyword extraction (HyDE) — it does not need to be 100% correct, but must contain relevant base images, system packages, and framework installation steps.

Requirements:
{requirements[:2000]}

Conflicts:
{json.dumps(conflicts[:10])}

Return only the raw Dockerfile content."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a DevOps expert. Return only raw Dockerfile content, no markdown fences or extra text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
            max_tokens=1024,
        )
        draft = response.choices[0].message.content.strip()
        if draft.startswith("```"):
            lines = draft.split("\n")
            draft = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return draft
    except Exception as e:
        logger.error(f"HyDE draft generation failed: {e}")
        return ""


# ---------------------------------------------------------------------------
# 5b. HyDE Keyword Extraction
# ---------------------------------------------------------------------------

def _extract_hyde_keywords(client, hyde_draft: str) -> list:
    """Extract 3-5 search keywords/tags from the draft Dockerfile."""
    if not hyde_draft:
        return []

    prompt = f"""Analyze the following draft Dockerfile and extract 3 to 5 core keywords that represent:
1. The base image or framework tag (e.g. 'nvidia/cuda', 'pytorch', 'tensorflow')
2. Key system libraries or python packages being installed (e.g. 'libgl1', 'torchvision')
3. Specific version tags if crucial (e.g. '1.7.1', 'cu101')

Draft Dockerfile:
{hyde_draft[:2000]}

Return only a JSON object with a 'keywords' array. Example: {{"keywords": ["cuda11.8", "torch", "libgl1"]}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an assistant. Return only a valid JSON object with a 'keywords' array of strings."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=256,
        )
        content = response.choices[0].message.content.strip()
        data = json.loads(content)
        if isinstance(data, dict):
            # Accept either {"keywords": [...]} or {"key": [...]}
            for val in data.values():
                if isinstance(val, list):
                    return val
        return []
    except Exception as e:
        logger.error(f"HyDE keyword extraction failed: {e}")
        return []


# ---------------------------------------------------------------------------
# 5c. Coral RAG Search
# ---------------------------------------------------------------------------

def _filter_search_keywords(keywords: list) -> list:
    """
    Filter and prioritize keyword tokens for code search.
    Keeps ML-relevant tokens, drops generic DevOps noise.
    """
    tokens = []
    for kw in keywords:
        parts = re.findall(r'[a-zA-Z0-9\.]+', kw)
        tokens.extend(p.strip() for p in parts if p.strip())

    # Words that pollute search results with generic hits
    noise_words = {
        'pip', 'python3', 'python', 'install', 'apt', 'git', 'wget', 'curl',
        'requirements', 'txt', 'github', 'libpq', 'build', 'essential',
        'devel', 'cudnn', 'ubuntu', 'slim', 'alpine', 'base', 'latest',
    }

    prioritized = []
    others = []

    for token in tokens:
        tl = token.lower()
        # Skip noise
        if any(w in tl for w in noise_words) or tl in noise_words:
            continue
        # Skip very short or very long tokens
        if len(token) < 3 or len(token) > 20:
            continue

        # ML-specific tokens get priority
        if any(f in tl for f in [
            'torch', 'cu', 'cuda', 'nvidia', 'tensorflow', 'jax',
            'transformers', 'diffusers', 'triton', 'xformers',
        ]) or re.match(r'^\d+\.\d+(\.\d+)?$', token):
            prioritized.append(token)
        else:
            others.append(token)

    result = prioritized + others
    seen = set()
    deduped = []
    for r in result:
        rl = r.lower()
        if rl not in seen:
            seen.add(rl)
            deduped.append(r)

    return deduped[:5]


def _extract_from_lines(dockerfiles: list) -> list:
    """Extract FROM lines from a list of Dockerfile dicts for use as base-image hints."""
    from_lines = []
    for df in dockerfiles:
        content = df.get("content", "")
        for line in content.splitlines():
            if line.strip().upper().startswith("FROM "):
                from_lines.append(f"{line.strip()}  # (from {df.get('repo', 'unknown')})")
                break  # Only first FROM per Dockerfile
    return from_lines


def _search_github_dockerfiles(keywords: list, dep_files: dict = None) -> list:
    """
    Use github.search_code and github.contents to fetch up to 4 real-world Dockerfiles.

    Improvements over the old version:
    - Adds 'stars:>50' filter for higher-quality results
    - Fetches up to 4 (was 3) reference Dockerfiles
    - Falls back to a broader query if the specific one returns 0 results
    """
    if not keywords:
        return []

    coral = get_coral_client()
    if not coral.available:
        logger.warning("Coral CLI is not available, skipping RAG search")
        return []

    cleaned_keywords = _filter_search_keywords(keywords)
    if not cleaned_keywords:
        return []

    dockerfiles = []

    # Primary query: specific keywords + quality filter
    primary_query = " ".join(cleaned_keywords) + " filename:Dockerfile stars:>50"
    dockerfiles = _run_dockerfile_search(coral, primary_query, limit=4)
    logger.info(f"RAG primary query '{primary_query}': {len(dockerfiles)} results")

    # Fallback: relax to just the top 2 keywords without star filter
    if len(dockerfiles) < 2 and len(cleaned_keywords) >= 2:
        fallback_query = " ".join(cleaned_keywords[:2]) + " filename:Dockerfile"
        dockerfiles = _run_dockerfile_search(coral, fallback_query, limit=4)
        logger.info(f"RAG fallback query '{fallback_query}': {len(dockerfiles)} results")

    return dockerfiles


def _run_dockerfile_search(coral, query_str: str, limit: int = 4) -> list:
    """Execute a single Coral search_code query and fetch matching Dockerfiles."""
    sql = f"""
        SELECT repository_full_name, path
        FROM github.search_code(q => '{query_str}')
        LIMIT {limit}
    """
    results = coral.sql(sql)
    if not results or "results" not in results or not results["results"]:
        return []

    dockerfiles = []
    for item in results["results"]:
        repo_full = item.get("repository_full_name")
        path = item.get("path")
        if not repo_full or not path:
            continue

        parts = repo_full.split("/")
        if len(parts) != 2:
            continue
        owner, repo = parts[0], parts[1]

        content_sql = f"""
            SELECT content_text as content
            FROM github.contents
            WHERE owner = '{owner}'
              AND repo = '{repo}'
              AND path = '{path}'
        """
        content_res = coral.sql(content_sql)
        if content_res and "results" in content_res and content_res["results"]:
            content = (
                content_res["results"][0].get("content")
                or content_res["results"][0].get("content_text")
            )
            if content:
                dockerfiles.append({"repo": repo_full, "path": path, "content": content})
                logger.info(f"Retrieved Dockerfile from {repo_full}:{path}")

    return dockerfiles


# ---------------------------------------------------------------------------
# 5d. FLARE Active Retrieval
# ---------------------------------------------------------------------------

def _flare_active_retrieval(client, conflicts: list) -> list:
    """
    Targeted search for specific legacy/conflict packages
    (e.g. torch==1.7.1) to find exact install instructions.
    Expanded to also detect jax, tensorflow, diffusers, transformers patterns.
    """
    coral = get_coral_client()
    if not coral.available or not conflicts:
        return []

    retrieved_contexts = []
    target_terms = []

    for conflict in conflicts:
        # ML framework patterns (expanded)
        matches = re.findall(
            r'(torch[a-z]*|cuda|cu\d+|python\s*\d+\.\d+|tensorflow|jax|'
            r'transformers|diffusers|xformers|triton)',
            conflict.lower()
        )
        target_terms.extend(matches)
        # Version numbers
        versions = re.findall(r'(\d+\.\d+\.\d+)', conflict)
        target_terms.extend(versions)

    target_terms = list(dict.fromkeys(target_terms))[:5]
    if not target_terms:
        return []

    # Code search for Dockerfiles containing these terms
    target_query = " ".join(target_terms) + " filename:Dockerfile"
    sql = f"""
        SELECT repository_full_name, path
        FROM github.search_code(q => '{target_query}')
        LIMIT 2
    """
    logger.info(f"FLARE code search: {target_query}")
    results = coral.sql(sql)

    if results and "results" in results and results["results"]:
        for item in results["results"]:
            repo_full = item.get("repository_full_name")
            path = item.get("path")
            if not repo_full or not path:
                continue
            parts = repo_full.split("/")
            if len(parts) != 2:
                continue
            owner, repo = parts[0], parts[1]

            content_sql = f"""
                SELECT content_text as content
                FROM github.contents
                WHERE owner = '{owner}'
                  AND repo = '{repo}'
                  AND path = '{path}'
            """
            content_res = coral.sql(content_sql)
            if content_res and "results" in content_res and content_res["results"]:
                content = (
                    content_res["results"][0].get("content")
                    or content_res["results"][0].get("content_text")
                )
                if content:
                    lines = content.split("\n")
                    matching_lines = [
                        line.strip() for line in lines
                        if any(term in line.lower() for term in target_terms)
                    ]
                    if matching_lines:
                        context_str = (
                            f"From {repo_full}/{path}:\n"
                            + "\n".join(matching_lines[:8])
                        )
                        retrieved_contexts.append(context_str)
                        logger.info(f"FLARE matched {len(matching_lines)} lines from {repo_full}:{path}")

    # Also search GitHub issues for known install problems
    issue_query = " ".join(target_terms) + " install error"
    issue_sql = f"""
        SELECT title, repository_url, number
        FROM github.search_issues(q => '{issue_query}')
        LIMIT 2
    """
    logger.info(f"FLARE issue search: {issue_query}")
    issue_results = coral.sql(issue_sql)
    if issue_results and "results" in issue_results and issue_results["results"]:
        for item in issue_results["results"]:
            title = item.get("title")
            repo_url = item.get("repository_url")
            number = item.get("number")
            if not title or not repo_url or number is None:
                continue
            repo_match = re.search(r'repos/([^/]+)/([^/]+)', repo_url)
            if not repo_match:
                continue
            owner, repo = repo_match.group(1), repo_match.group(2)
            retrieved_contexts.append(
                f"GitHub Issue {owner}/{repo} #{number}: '{title}'"
            )

    return retrieved_contexts


# ---------------------------------------------------------------------------
# 5e. Dedicated Base Image Selection (NEW)
# ---------------------------------------------------------------------------

def _select_base_image(
    client,
    dep_files: dict,
    conflicts: list,
    repo_year: str = None,
    rag_dockerfiles: list = None,
) -> dict:
    """
    Dedicated LLM call to pick the optimal base image BEFORE writing the
    full Dockerfile. This single-task prompt produces a cleaner, more
    reasoned decision than burying base-image selection inside a large prompt.

    The agent is free to suggest ANY image — there is no hardcoded allowed list.
    Guidance is provided via FROM lines observed in RAG Dockerfiles and
    analysis of the requirements.

    Returns:
        {"base_image": "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04", "reason": "..."}
    """
    # Collect requirements content
    all_deps = ""
    for name, content in dep_files.items():
        all_deps += f"\n--- {name} ---\n{content[:1500]}\n"
    if not all_deps:
        all_deps = "(no dependency files found)"

    # Collect FROM lines from RAG Dockerfiles as evidence
    from_hints = ""
    if rag_dockerfiles:
        from_lines = _extract_from_lines(rag_dockerfiles)
        if from_lines:
            from_hints = (
                "\n## Base images used in similar real-world repositories "
                "(from GitHub Coral search):\n"
                + "\n".join(from_lines)
            )

    # Temporal context
    temporal_hint = ""
    if repo_year:
        temporal_hint = f"\nThis repository is from {repo_year}. Choose a base image that was mature and available in {repo_year}.\n"

    conflicts_str = json.dumps(conflicts[:10], indent=2) if conflicts else "None"

    prompt = f"""You are a senior ML DevOps engineer selecting the optimal Docker base image for a research repository.

## Dependency files:
{all_deps}

## Dependency conflicts:
{conflicts_str}
{from_hints}
{temporal_hint}

## Your task:
Analyze the dependencies and select ONE specific, pinned Docker base image.

Decision rules:
- If requirements contain torch, torchvision, jax, tensorflow-gpu, or CUDA-specific packages → use nvidia/cuda (e.g. nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04) or pytorch/pytorch
- If requirements use only CPU frameworks (sklearn, pandas, numpy, huggingface without GPU) → use python:3.X-slim (pick Python version to match any pinned version in requirements)
- If requirements contain conda channels or environment.yml → use continuumio/miniconda3
- If requirements mention rocm → use rocm/pytorch
- Match the CUDA version to the torch version: torch 2.x → CUDA 12.1 or 11.8; torch 1.x → CUDA 11.x or 10.x

Return ONLY a JSON object with exactly two keys:
{{
  "base_image": "<full image:tag>",
  "reason": "<1-2 sentence justification>"
}}"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a DevOps expert. Return ONLY a valid JSON object "
                        "with 'base_image' and 'reason' keys. No markdown, no extra text."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=256,
        )
        result = json.loads(response.choices[0].message.content)
        base_image = result.get("base_image", "").strip()
        reason = result.get("reason", "")
        if base_image:
            return {"base_image": base_image, "reason": reason}
    except Exception as e:
        logger.error(f"Base image selection failed: {e}")

    # Fallback: infer from requirements
    return _infer_base_image_fallback(dep_files, repo_year)


def _infer_base_image_fallback(dep_files: dict, repo_year: str = None) -> dict:
    """
    Heuristic fallback when the LLM call fails.
    Inspects requirements for GPU packages and picks accordingly.
    """
    all_reqs = " ".join(dep_files.values()).lower()

    gpu_packages = ["torch", "torchvision", "tensorflow-gpu", "jax[cuda]", "cupy"]
    needs_gpu = any(pkg in all_reqs for pkg in gpu_packages)

    if needs_gpu:
        # Try to match torch version to a CUDA version
        torch_match = re.search(r'torch[=<>!]+(\d+)\.(\d+)', all_reqs)
        if torch_match:
            major, minor = int(torch_match.group(1)), int(torch_match.group(2))
            if major >= 2:
                cuda_tag = "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04"
            elif major == 1 and minor >= 10:
                cuda_tag = "nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04"
            else:
                cuda_tag = "nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04"
        else:
            cuda_tag = "nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04"
        return {"base_image": cuda_tag, "reason": "GPU packages detected in requirements (fallback heuristic)"}

    # CPU-only — pick Python version from requirements if pinned
    py_match = re.search(r'python[=<>!]+(\d+\.\d+)', all_reqs)
    py_version = py_match.group(1) if py_match else "3.11"
    return {
        "base_image": f"python:{py_version}-slim",
        "reason": f"CPU-only packages detected (fallback heuristic), Python {py_version}",
    }


# ---------------------------------------------------------------------------
# 5f. Dockerfile Generation (with selected base image injected)
# ---------------------------------------------------------------------------

def _generate_dockerfile(
    client, requirements: str, conflicts: list, paper_title: str,
    entrypoint: str = None, rag_dockerfiles: list = None,
    flare_contexts: list = None, readme_content: str = "",
    dep_files: dict = None, repo_year: str = None,
    selected_base_image: str = "",
) -> str:
    """
    Generate a working Dockerfile using Groq, enriched with:
    - The pre-selected base image (injected as a hard constraint)
    - RAG reference Dockerfiles from similar repositories
    - FLARE snippets for conflict-specific install patterns
    - FROM-line hints extracted from RAG results
    """
    if dep_files is None:
        dep_files = {}

    cmd_instruction = (
        f"7. CMD: `python {entrypoint}`" if entrypoint
        else "7. CMD: Do NOT include a CMD instruction since the repo entrypoint is unknown"
    )

    # Build reference context from RAG
    reference_context = ""
    from_lines = _extract_from_lines(rag_dockerfiles or [])
    if from_lines:
        reference_context += "\n## Base images used in similar repositories (evidence for your choice):\n"
        reference_context += "\n".join(from_lines) + "\n"

    if rag_dockerfiles:
        reference_context += "\n## Reference Dockerfiles from similar repositories:\n"
        for i, df in enumerate(rag_dockerfiles):
            reference_context += f"--- Reference {i+1} ({df['repo']}:{df['path']}) ---\n"
            reference_context += df['content'][:1200] + "\n"

    if flare_contexts:
        reference_context += "\n## Real-world installation snippets from GitHub:\n"
        for ctx in flare_contexts:
            reference_context += f"- {ctx}\n"

    # Dependency files context
    dep_context = ""
    if dep_files:
        dep_context = "\n## All dependency files found in the repository:\n"
        for name, content in dep_files.items():
            dep_context += f"\n--- {name} ---\n{content[:2500]}\n"
    elif requirements:
        dep_context = f"\n## requirements.txt:\n{requirements[:3000]}\n"

    # README context
    readme_section = ""
    if readme_content:
        readme_section = f"\n## README.md (may contain build/install instructions):\n{readme_content[:2500]}\n"
        readme_section += (
            "\nCRITICAL: If the README mentions special installation steps "
            "(e.g. 'pip install open_spiel', 'build from source', custom CUDA instructions, "
            "conda commands), you MUST include those steps in the Dockerfile.\n"
        )

    # Temporal pinning
    temporal_rule = ""
    if repo_year:
        temporal_rule = (
            f"\nTEMPORAL RULE: This repository is from {repo_year}.\n"
            f"- Pin ALL packages to versions that were current in {repo_year}\n"
            f"- Do NOT upgrade to latest — they may have breaking API changes\n"
            f"- Example: A 2019 repo should use torch~=1.2, NOT torch>=2.0\n"
        )

    # ─── Mandatory base image constraint ────────────────────────────────────
    base_constraint = ""
    if selected_base_image:
        base_constraint = f"""
╔══════════════════════════════════════════════════════════╗
║  MANDATORY BASE IMAGE — DO NOT CHANGE THIS              ║
║  FROM {selected_base_image:<48} ║
║  (Selected by dedicated analysis of requirements)       ║
╚══════════════════════════════════════════════════════════╝
"""
    # ────────────────────────────────────────────────────────────────────────

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a DevOps expert specializing in ML research reproducibility. "
                    "Return only raw Dockerfile content — no markdown fences, no extra text."
                ),
            },
            {
                "role": "user",
                "content": f"""Generate a working Dockerfile for this ML research paper implementation.

Paper: {paper_title}
{base_constraint}
{temporal_rule}
{dep_context}

Known conflicts to fix:
{json.dumps(conflicts[:10], indent=2)}
{reference_context}
{readme_section}

=== DOCKERFILE GENERATION CHECKLIST — follow these steps IN ORDER ===

1. BASE IMAGE: Use EXACTLY the mandatory base image shown above. Do not substitute or change it.
2. SYSTEM DEPS: Always install: build-essential, git, wget. Check README for extra system packages.
3. PYTHON SETUP: If using nvidia/cuda or ubuntu base, MUST install python3, python3-pip, and create symlink `ln -sf /usr/bin/python3 /usr/bin/python`.
4. COPY DEPENDENCY FILES FIRST (for Docker layer caching):
   - If requirements.txt exists: `COPY requirements.txt .` then `RUN pip install --no-cache-dir -r requirements.txt`
   - If setup.py exists: `COPY setup.py .` then `RUN pip install --no-cache-dir -e .`
   - If pyproject.toml exists: `COPY pyproject.toml .` then `RUN pip install --no-cache-dir .`
5. COPY ALL SOURCE: `COPY . /workspace/`
6. WORKDIR: Set to `/workspace`
{cmd_instruction}

=== ANTI-PATTERNS — NEVER DO THESE ===
❌ Do NOT change the mandatory base image
❌ Don't guess package names — use only the requirements listed above
❌ Don't hardcode `pip install torch==X.Y.Z` if torch is already in requirements.txt
❌ Don't skip installing python3 when using a bare OS or CUDA base image
❌ Don't forget `rm -rf /var/lib/apt/lists/*` after apt-get install
""",
            },
        ],
        temperature=0.15,
        max_tokens=2048,
    )

    dockerfile = response.choices[0].message.content.strip()

    if dockerfile.startswith("```"):
        lines = dockerfile.split("\n")
        dockerfile = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    return dockerfile


# ---------------------------------------------------------------------------
# 5g. Self-Healing
# ---------------------------------------------------------------------------

def _self_heal_dockerfile(
    client, dockerfile: str, issues: list,
    requirements: str, dep_files: dict, readme_content: str,
    selected_base_image: str = "",
) -> str:
    """Re-prompt the LLM to fix validation issues found in the generated Dockerfile."""
    dep_file_names = list(dep_files.keys()) if dep_files else []

    base_constraint = ""
    if selected_base_image:
        base_constraint = f"\nMANDATORY: Keep the base image as '{selected_base_image}' — do not change it.\n"

    correction_prompt = f"""The Dockerfile you generated has the following validation issues:

{chr(10).join(f'- {issue}' for issue in issues)}

Here is the current Dockerfile:
{dockerfile}

Dependency files available: {dep_file_names}
{base_constraint}

Fix ALL of the above issues. Remember:
- If requirements.txt exists, COPY it and run `pip install --no-cache-dir -r requirements.txt`
- If setup.py exists, COPY it and run `pip install --no-cache-dir -e .`
- Install python3 and pip when using bare OS or CUDA base images
- Add `rm -rf /var/lib/apt/lists/*` after apt-get install
- Set WORKDIR

Return ONLY the corrected Dockerfile content, no markdown fences or extra text."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are a DevOps expert. Return only raw Dockerfile content, no markdown fences or extra text."},
                {"role": "user", "content": correction_prompt}
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        corrected = response.choices[0].message.content.strip()
        if corrected.startswith("```"):
            lines = corrected.split("\n")
            corrected = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        logger.info("Self-healing: Dockerfile corrected successfully")
        return corrected
    except Exception as e:
        logger.error(f"Self-healing correction failed: {e}")
        return dockerfile


# ---------------------------------------------------------------------------
# Requirements resolver
# ---------------------------------------------------------------------------

def _resolve_requirements(
    client, requirements: str, conflicts: list,
    rag_dockerfiles: list = None, flare_contexts: list = None,
    repo_year: str = None,
) -> dict:
    """Resolve version conflicts in requirements using Groq and reference RAG context."""

    if not conflicts:
        return {"requirements": requirements, "notes": ["No conflicts to resolve"]}

    reference_context = ""
    if rag_dockerfiles:
        reference_context += "\nHere are reference Dockerfiles from similar repositories:\n"
        for i, df in enumerate(rag_dockerfiles):
            reference_context += f"--- Reference {i+1} ({df['repo']}:{df['path']}) ---\n"
            reference_context += df['content'][:1200] + "\n"

    if flare_contexts:
        reference_context += "\nAdditional real-world installation snippets from GitHub:\n"
        for ctx in flare_contexts:
            reference_context += f"- {ctx}\n"

    temporal_rule = ""
    if repo_year:
        temporal_rule = (
            f"CRITICAL: This code is from {repo_year}. "
            "Pin packages to versions standard in that year. Do NOT upgrade to modern versions."
        )

    system_msg = (
        "You are a Python dependency expert. Return ONLY valid JSON — no markdown, no code fences, "
        "no extra text. CRITICAL: The 'requirements' field MUST be a single JSON string. "
        "Separate packages using the literal escape sequence \\n inside the string. "
        "NEVER split the string across multiple lines using Python-style implicit concatenation."
    )

    user_msg = f"""Fix these Python dependency conflicts. {temporal_rule}

Original requirements.txt:
{requirements[:3000]}

Conflicts:
{json.dumps(conflicts[:10], indent=2)}
{reference_context}

Return a JSON object with exactly these two keys:
{{
  "requirements": "all packages as a SINGLE string, separated by \\n escape sequences",
  "notes": ["explanation of each change made"]
}}

Example: {{"requirements": "numpy==1.24.2\\nopencv-python==4.7.0\\ntorch==2.0.1", "notes": ["pinned versions"]}}
DO NOT write packages on separate lines inside the JSON string."""

    def _try_parse(raw: str):
        raw = raw.strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        repaired = re.sub(r'"\s*\n\s*"', '', raw)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
        m = re.search(r'\{.*\}', repaired, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                pass
        return None

    # First attempt
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=2048,
        )
        result = _try_parse(response.choices[0].message.content)
        if result is not None:
            return result
        logger.warning("_resolve_requirements: first response unparseable, retrying...")
    except Exception as e:
        logger.warning(f"_resolve_requirements: first call failed ({e}), retrying...")

    # Retry with stricter prompt
    retry_user_msg = (
        f"Return ONLY a JSON object. The 'requirements' value must be a SINGLE string "
        f"with packages joined by the escape sequence \\n.\n\n"
        f"Correct example:\n"
        f'  {{"requirements": "numpy==1.24.2\\nopencv-python==4.7.0\\ntorch==2.0.1", "notes": ["pinned versions"]}}\n\n'
        f"Fix the following:\nRequirements:\n{requirements[:2000]}\n\nConflicts:\n{json.dumps(conflicts[:5])}"
    )
    try:
        retry_response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": retry_user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1024,
        )
        result = _try_parse(retry_response.choices[0].message.content)
        if result is not None:
            return result
        logger.error("_resolve_requirements retry also unparseable")
    except Exception as e:
        logger.error(f"_resolve_requirements retry also failed: {e}")

    return {"requirements": requirements, "notes": ["Failed to parse LLM response — using original requirements"]}


# ---------------------------------------------------------------------------
# Fallback: no LLM available
# ---------------------------------------------------------------------------

def _basic_dockerfile(requirements_content: str, paper_title: str = "", entrypoint: str = None) -> dict:
    """Generate a basic Dockerfile without LLM (fallback)."""
    cmd_line = (
        f'CMD ["python", "{entrypoint}"]' if entrypoint
        else '# CMD ["python", "<entrypoint>.py"]  # TODO: set the correct entrypoint'
    )

    dockerfile = f"""# Dockerfile for: {paper_title or 'Research Paper Implementation'}
# Generated by Coralaus

FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \\
    build-essential \\
    git \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy implementation
COPY . .

# Run
{cmd_line}
"""

    return {
        "dockerfile": dockerfile,
        "resolved_requirements": requirements_content,
        "environment_yaml": None,
        "resolution_notes": ["Basic Dockerfile generated (no LLM available for conflict resolution)"],
        "llm_model": None,
        "selected_base_image": "python:3.11-slim",
        "base_image_reason": "Fallback: no LLM available",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_reqs = "torch==2.0.1\ntorchvision==0.15.2\nnumpy>=1.20\n"
    test_conflicts = ["torch 2.0.1 requires CUDA 11.7 or 11.8 but base image has CUDA 10.2"]
    result = resolve_conflicts(test_reqs, test_conflicts, "Test Paper")
    print("=== Selected Base Image ===")
    print(f"  {result.get('selected_base_image')} — {result.get('base_image_reason')}")
    print("=== Dockerfile ===")
    print(result["dockerfile"])
    print("\n=== Notes ===")
    for note in result["resolution_notes"]:
        print(f"  - {note}")
