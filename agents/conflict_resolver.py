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

        # Generate Dockerfile
        dockerfile = _generate_dockerfile(client, requirements_content, conflicts, paper_title, entrypoint)

        # Generate resolved requirements
        resolved_reqs = _resolve_requirements(client, requirements_content, conflicts)

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


def _generate_dockerfile(client, requirements: str, conflicts: list, paper_title: str, entrypoint: str = None) -> str:
    """Generate a working Dockerfile using Groq."""
    if entrypoint:
        cmd_instruction = f"- The final CMD should be: python {entrypoint}"
    else:
        cmd_instruction = "- Do NOT include a CMD instruction since the repo entrypoint is unknown"

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

Rules:
- Use Python 3.11-slim as base image
- Pin all dependency versions
- Add a comment above each conflict fix explaining what was changed
- Install system dependencies commonly needed for ML (build-essential, git, etc.)
- Set up a clean working directory and copy all project files into it
{cmd_instruction}
""",
            },
        ],
        temperature=0.2,
        max_tokens=2048,
    )

    dockerfile = response.choices[0].message.content.strip()

    # Strip markdown fences if present
    if dockerfile.startswith("```"):
        lines = dockerfile.split("\n")
        dockerfile = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    return dockerfile


def _resolve_requirements(client, requirements: str, conflicts: list) -> dict:
    """Resolve version conflicts in requirements using Groq."""
    if not conflicts:
        return {"requirements": requirements, "notes": ["No conflicts to resolve"]}

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

Return JSON:
{{
  "requirements": "fixed requirements.txt content with pinned versions",
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
