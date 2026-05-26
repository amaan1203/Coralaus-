"""
Component 5 — Conflict Resolution & Dockerfile Generation

Resolves dependency conflicts and generates a working Dockerfile.
Uses Groq (Llama 3.3 70B) for speed.

Input:  Original requirements + conflict list from Component 4
Output: Dockerfile and/or environment.yaml
"""

import os
import json
import logging
from typing import Optional
from agents.coral_utils import get_coral_client

logger = logging.getLogger(__name__)


def resolve_conflicts(requirements_content: str, conflicts: list, paper_title: str = "", entrypoint: str = None) -> dict:
    """
    Resolve dependency conflicts and generate a working Dockerfile.

    Args:
        requirements_content: Original requirements.txt content
        conflicts: List of conflict descriptions from Component 4
        paper_title: Paper title for Dockerfile comments
        entrypoint: Detected entrypoint script (e.g. 'main.py', 'train.py').
                    If None, the Dockerfile will use a generic COPY without a CMD.

    Returns:
        Dict with:
            - dockerfile (str): Generated Dockerfile content
            - resolved_requirements (str): Fixed requirements.txt
            - environment_yaml (str): Optional conda environment
            - resolution_notes (list): What was changed and why
    """
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set, generating basic Dockerfile")
        return _basic_dockerfile(requirements_content, paper_title, entrypoint)

    try:
        from groq import Groq

        client = Groq(api_key=api_key)

        # 1. HyDE Draft Generation
        hyde_draft = _generate_hyde_draft(client, requirements_content, conflicts)
        
        # 2. HyDE Keyword Extraction
        keywords = _extract_hyde_keywords(client, hyde_draft)
        logger.info(f"Extracted HyDE keywords: {keywords}")
        
        # 3. Coral Reconnaissance (Search and Fetch reference Dockerfiles)
        rag_dockerfiles = _search_github_dockerfiles(keywords)
        
        # 4. FLARE Active Retrieval
        flare_contexts = _flare_active_retrieval(client, conflicts)

        # Generate Dockerfile
        dockerfile = _generate_dockerfile(
            client, requirements_content, conflicts, paper_title, entrypoint,
            rag_dockerfiles=rag_dockerfiles, flare_contexts=flare_contexts
        )

        # Generate resolved requirements
        resolved_reqs = _resolve_requirements(
            client, requirements_content, conflicts,
            rag_dockerfiles=rag_dockerfiles, flare_contexts=flare_contexts
        )

        return {
            "dockerfile": dockerfile,
            "resolved_requirements": resolved_reqs.get("requirements", requirements_content),
            "environment_yaml": None,
            "resolution_notes": resolved_reqs.get("notes", []),
            "llm_model": "llama-3.3-70b-versatile",
        }

    except ImportError:
        logger.warning("groq package not installed, generating basic Dockerfile")
        return _basic_dockerfile(requirements_content, paper_title, entrypoint)
    except Exception as e:
        logger.error(f"Conflict resolution failed: {e}")
        return _basic_dockerfile(requirements_content, paper_title, entrypoint)


def _generate_hyde_draft(client, requirements: str, conflicts: list) -> str:
    """Generate a hypothetical, potentially flawed draft Dockerfile based on requirements/conflicts."""
    prompt = f"""You are a DevOps assistant. Generate a hypothetical, draft Dockerfile based on the following requirements and conflicts.
This draft is for search embedding purposes (HyDE) and doesn't need to be 100% syntactically perfect, but it must contain relevant base images, system packages, and framework installation steps.

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


def _extract_hyde_keywords(client, hyde_draft: str) -> list:
    """Extract 3-5 search keywords/tags from the draft Dockerfile to query github.search_code."""
    if not hyde_draft:
        return []

    prompt = f"""Analyze the following draft Dockerfile and extract 3 to 5 core keywords or search terms that represent:
1. The base image or framework tag (e.g. 'nvidia/cuda', 'pytorch', 'tensorflow')
2. Key system libraries or python packages being installed (e.g. 'libgl1', 'gstreamer', 'torchvision', etc.)
3. The conflict version or specific version tags if crucial (e.g. '1.7.1', 'cu101')

Draft Dockerfile:
{hyde_draft[:2000]}

Return only a JSON array of string keywords. Example: ["cuda10.1", "torch==1.7.1", "libgl1"]"""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": "You are an assistant. Return only a valid JSON array of strings, no explanation or markdown fences."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=256,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            lines = content.split("\n")
            content = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        data = json.loads(content)
        if isinstance(data, list):
            return data
        elif isinstance(data, dict):
            for val in data.values():
                if isinstance(val, list):
                    return val
        return []
    except Exception as e:
        logger.error(f"HyDE keyword extraction failed: {e}")
        return []


def _filter_search_keywords(keywords: list) -> list:
    """Filter and prioritize individual keyword tokens for code search to avoid empty results."""
    tokens = []
    import re
    for kw in keywords:
        # Extract simple clean tokens (alphanumeric and dots)
        parts = re.findall(r'[a-zA-Z0-9\.]+', kw)
        for part in parts:
            part = part.strip()
            if part:
                tokens.append(part)
                
    prioritized = []
    others = []
    ignore_words = {'pip', 'python3', 'python', 'install', 'apt', 'git', 'requirements', 'txt', 'github', 'libpq-dev', 'build-essential', 'devel', 'cudnn', 'ubuntu', 'slim', 'alpine'}
    
    for token in tokens:
        token_lower = token.lower()
        if any(w in token_lower for w in ignore_words) or token_lower in ignore_words:
            continue
        if len(token) > 15:
            continue
        
        # Prioritize core packages/versions
        if any(f in token_lower for f in ['torch', 'cu', 'cuda', 'nvidia', 'tensorflow', 'jax', 'transformers']) or re.match(r'^\d+\.\d+(\.\d+)?', token):
            prioritized.append(token)
        else:
            others.append(token)
            
    # Combine and return top 4 distinct tokens
    result = prioritized + others
    seen = set()
    deduped = []
    for r in result:
        r_lower = r.lower()
        if r_lower not in seen:
            seen.add(r_lower)
            deduped.append(r)
            
    return deduped[:4]


def _search_github_dockerfiles(keywords: list) -> list:
    """Use github.search_code and github.contents to fetch up to 3 real-world Dockerfiles."""
    if not keywords:
        return []

    coral = get_coral_client()
    if not coral.available:
        logger.warning("Coral CLI is not available, skipping RAG search")
        return []

    cleaned_keywords = _filter_search_keywords(keywords)
    if not cleaned_keywords:
        return []

    query_str = " ".join(cleaned_keywords) + " filename:Dockerfile"
    sql = f"""
        SELECT repository_full_name, path 
        FROM github.search_code(q => '{query_str}') 
        LIMIT 3
    """
    logger.info(f"RAG search query: {query_str}")
    results = coral.sql(sql)
    if not results or "results" not in results or not results["results"]:
        logger.info("RAG search returned no results")
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
            content = content_res["results"][0].get("content")
            if content:
                dockerfiles.append({
                    "repo": repo_full,
                    "path": path,
                    "content": content
                })
                logger.info(f"Successfully retrieved Dockerfile from {repo_full}:{path}")

    return dockerfiles


def _flare_active_retrieval(client, conflicts: list) -> list:
    """Targeted search for specific legacy or conflict packages (e.g. torch==1.7.1) to find exact wheel URLs or extra install instructions."""
    coral = get_coral_client()
    if not coral.available or not conflicts:
        return []

    retrieved_contexts = []
    import re
    target_terms = []
    for conflict in conflicts:
        matches = re.findall(r'(torch[a-z]*|cuda|cu\d+|python\s*\d+\.\d+)', conflict.lower())
        target_terms.extend(matches)
        versions = re.findall(r'(\d+\.\d+\.\d+)', conflict)
        target_terms.extend(versions)

    target_terms = list(dict.fromkeys(target_terms))[:4]
    if not target_terms:
        return []

    target_query = " ".join(target_terms) + " filename:Dockerfile"
    sql = f"""
        SELECT repository_full_name, path 
        FROM github.search_code(q => '{target_query}') 
        LIMIT 2
    """
    logger.info(f"FLARE active retrieval code search query: {target_query}")
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
                content = content_res["results"][0].get("content")
                if content:
                    lines = content.split("\n")
                    matching_lines = []
                    for line in lines:
                        if any(term in line.lower() for term in target_terms):
                            matching_lines.append(line.strip())
                    if matching_lines:
                        context_str = f"From Dockerfile in {repo_full} (path: {path}):\n" + "\n".join(matching_lines[:5])
                        retrieved_contexts.append(context_str)
                        logger.info(f"FLARE retrieved matches from {repo_full}:{path}")

    # Search issues
    issue_query = " ".join(target_terms) + " install"
    issue_sql = f"""
        SELECT title, repository_url, number
        FROM github.search_issues(q => '{issue_query}')
        LIMIT 2
    """
    logger.info(f"FLARE active retrieval issue search query: {issue_query}")
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
            context_str = f"GitHub Issue from {owner}/{repo} #{number}: '{title}'"
            retrieved_contexts.append(context_str)

    return retrieved_contexts


def _generate_dockerfile(client, requirements: str, conflicts: list, paper_title: str, entrypoint: str = None, rag_dockerfiles: list = None, flare_contexts: list = None) -> str:
    """Generate a working Dockerfile using Groq, enriched with RAG context."""
    if entrypoint:
        cmd_instruction = f"- The final CMD should be: python {entrypoint}"
    else:
        cmd_instruction = "- Do NOT include a CMD instruction since the repo entrypoint is unknown"

    reference_context = ""
    if rag_dockerfiles:
        reference_context += "\nHere are reference Dockerfiles from other repositories that solve similar installations:\n"
        for i, df in enumerate(rag_dockerfiles):
            reference_context += f"--- Reference Dockerfile {i+1} (from {df['repo']}:{df['path']}) ---\n"
            reference_context += df['content'][:1500] + "\n"

    if flare_contexts:
        reference_context += "\nHere are additional real-world installation snippets and references found on GitHub:\n"
        for ctx in flare_contexts:
            reference_context += f"- {ctx}\n"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a DevOps expert. Return only raw Dockerfile content, no markdown fences or extra text.",
            },
            {
                "role": "user",
                "content": f"""Generate a working Dockerfile for this ML research paper implementation.

Paper: {paper_title}

requirements.txt:
{requirements[:3000]}

Known conflicts to fix:
{json.dumps(conflicts[:10], indent=2)}
{reference_context}

Rules:
- CRITICAL BASE IMAGE RULE: You must analyze the provided 'Reference Dockerfiles' to determine the correct base image. If the requirements contain heavy ML frameworks (torch, tensorflow, jax) with CUDA dependencies, you MUST use an official nvidia/cuda base image that matches the references. Do NOT use python:slim images for GPU-bound ML frameworks.
CRITICAL SYSTEM RULE: If you select a bare-metal OS image like nvidia/cuda or ubuntu, you MUST explicitly install python3, python3-pip, and create a symlink so python points to python3 BEFORE attempting to run any pip install commands
- Pin all dependency versions
- Add a comment above each conflict fix explaining what was changed
- Install system dependencies commonly needed for ML (build-essential, git, etc.)
- Set up a clean working directory and copy all project files into it
- USE the above Reference Dockerfiles and Installation snippets to find correct pip install arguments (especially for torch/cuda binaries, e.g. using --find-links stable wheels if required)
{cmd_instruction}
""",
            },
        ],
        temperature=0.2,
        max_tokens=2048,
    )

    dockerfile = response.choices[0].message.content.strip()

    if dockerfile.startswith("```"):
        lines = dockerfile.split("\n")
        dockerfile = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    return dockerfile


def _resolve_requirements(client, requirements: str, conflicts: list, rag_dockerfiles: list = None, flare_contexts: list = None) -> dict:
    """Resolve version conflicts in requirements using Groq and reference RAG context."""
    if not conflicts:
        return {"requirements": requirements, "notes": ["No conflicts to resolve"]}

    reference_context = ""
    if rag_dockerfiles:
        reference_context += "\nHere are reference Dockerfiles from other repositories that solve similar installations:\n"
        for i, df in enumerate(rag_dockerfiles):
            reference_context += f"--- Reference Dockerfile {i+1} (from {df['repo']}:{df['path']}) ---\n"
            reference_context += df['content'][:1500] + "\n"

    if flare_contexts:
        reference_context += "\nHere are additional real-world installation snippets and references found on GitHub:\n"
        for ctx in flare_contexts:
            reference_context += f"- {ctx}\n"

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {
                "role": "system",
                "content": "You are a Python dependency expert. Return only valid JSON, no markdown.",
            },
            {
                "role": "user",
                "content": f"""Fix these Python dependency conflicts.

Original requirements.txt:
{requirements[:3000]}

Conflicts:
{json.dumps(conflicts[:10], indent=2)}
{reference_context}

Return JSON:
{{
  "requirements": "fixed requirements.txt content with pinned versions, including any required -f / --find-links flags if necessary based on reference contexts",
  "notes": ["explanation of each change made"]
}}""",
            },
        ],
        response_format={"type": "json_object"},
        temperature=0.1,
        max_tokens=2048,
    )

    try:
        return json.loads(response.choices[0].message.content)
    except json.JSONDecodeError:
        return {"requirements": requirements, "notes": ["Failed to parse LLM response"]}


def _basic_dockerfile(requirements_content: str, paper_title: str = "", entrypoint: str = None) -> dict:
    """Generate a basic Dockerfile without LLM (fallback)."""
    if entrypoint:
        cmd_line = f'CMD ["python", "{entrypoint}"]'
    else:
        cmd_line = '# CMD ["python", "<entrypoint>.py"]  # TODO: set the correct entrypoint for this repo'

    dockerfile = f"""# Dockerfile for: {paper_title or 'Research Paper Implementation'}
# Generated by PaperDock

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
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_reqs = "torch==1.9.0\ntorchvision==0.14.0\nnumpy>=1.20\n"
    test_conflicts = ["torch 1.9.0 and torchvision 0.14.0 have version mismatch"]
    result = resolve_conflicts(test_reqs, test_conflicts, "Test Paper")
    print("=== Dockerfile ===")
    print(result["dockerfile"])
    print("\n=== Notes ===")
    for note in result["resolution_notes"]:
        print(f"  - {note}")
