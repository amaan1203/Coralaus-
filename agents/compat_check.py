"""
Component 4 — Dependency Compatibility Check

Fetches requirements.txt / environment.yml from the repo using Coral,
then runs pip dry-run to detect conflicts. Falls back to Groq LLM
for complex conflict analysis.

Input:  GitHub repo URL (owner/repo)
Output: Conflict report dict
"""

import os
import json
import logging
import subprocess
import tempfile
from typing import Optional
from agents.coral_utils import get_coral_client

logger = logging.getLogger(__name__)


def check_compatibility(repo_url: str, owner: str = None, repo: str = None) -> dict:
    """
    Check dependency compatibility for a GitHub repository.

    Args:
        repo_url: Full GitHub URL
        owner: Repo owner (parsed from URL if not provided)
        repo: Repo name (parsed from URL if not provided)

    Returns:
        Dict with:
            - requirements_found (bool)
            - requirements_content (str)
            - conflicts (list)
            - warnings (list)
            - clean (bool)
            - analysis_method (str): 'pip_dry_run' or 'groq_llm'
    """
    import re

    if not owner or not repo:
        match = re.search(r"github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", repo_url)
        if match:
            owner, repo = match.group(1), match.group(2)
        else:
            return _empty_result("Could not parse GitHub URL")

    # Step 1: Fetch dependency files via Coral
    dep_files = _fetch_dep_files_coral(owner, repo)

    # Fallback: use GitHub API if Coral is unavailable
    if not dep_files:
        dep_files = _fetch_dep_files_github(owner, repo)

    if not dep_files:
        return _empty_result("No dependency files found in repository")

    result = {
        "requirements_found": True,
        "dep_files": {name: content[:500] for name, content in dep_files.items()},
        "conflicts": [],
        "warnings": [],
        "clean": True,
        "analysis_method": None,
    }

    # Step 2: Try automated pip dry-run (no LLM)
    requirements_content = dep_files.get("requirements.txt", "")
    if requirements_content:
        pip_result = _pip_dry_run(requirements_content)
        if pip_result["has_errors"]:
            result["conflicts"] = pip_result["errors"]
            result["warnings"] = pip_result["warnings"]
            result["clean"] = False
            result["analysis_method"] = "pip_dry_run"

            # Step 3: If conflicts found, get detailed analysis from Groq
            groq_analysis = _groq_analyze(requirements_content, pip_result["raw_output"])
            if groq_analysis:
                result["conflicts"] = groq_analysis.get("conflicts", result["conflicts"])
                result["warnings"] = groq_analysis.get("warnings", result["warnings"])
                result["clean"] = groq_analysis.get("clean", False)
                result["analysis_method"] = "groq_llm"
        else:
            result["analysis_method"] = "pip_dry_run"
            result["clean"] = True

    # Check setup.py / pyproject.toml as supplementary info
    for extra_file in ["setup.py", "pyproject.toml", "environment.yml"]:
        if extra_file in dep_files:
            result["warnings"].append(f"Also found {extra_file} — manual review recommended")

    logger.info(f"Compat check: clean={result['clean']}, {len(result['conflicts'])} conflicts")
    return result


def _fetch_dep_files_coral(owner: str, repo: str) -> dict:
    """Fetch dependency files using Coral GitHub connector."""
    coral = get_coral_client()
    if not coral.available:
        return {}

    dep_files = {}
    for filename in ['requirements.txt', 'environment.yml', 'setup.py', 'pyproject.toml']:
        result = coral.sql(f"""
            SELECT content_text as content
            FROM github.contents
            WHERE owner = '{owner}'
              AND repo = '{repo}'
              AND path = '{filename}'
        """)
        if result and "results" in result and result["results"]:
            content = result["results"][0].get("content")
            if content:
                dep_files[filename] = content

    return dep_files


def _fetch_dep_files_github(owner: str, repo: str) -> dict:
    """Fallback: fetch dependency files from GitHub API."""
    import requests

    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token and "dummy" not in token.lower() and "your_" not in token.lower():
        headers["Authorization"] = f"token {token}"

    dep_files = {}
    targets = ["requirements.txt", "environment.yml", "setup.py", "pyproject.toml"]

    for filename in targets:
        try:
            resp = requests.get(
                f"https://api.github.com/repos/{owner}/{repo}/contents/{filename}",
                headers=headers, timeout=10
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("encoding") == "base64":
                    import base64
                    content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                    dep_files[filename] = content
                elif data.get("download_url"):
                    dl_resp = requests.get(data["download_url"], timeout=10)
                    if dl_resp.status_code == 200:
                        dep_files[filename] = dl_resp.text
        except Exception as e:
            logger.debug(f"Could not fetch {filename}: {e}")

    return dep_files


def _pip_dry_run(requirements_content: str) -> dict:
    """Run pip install --dry-run to detect conflicts."""
    result = {
        "has_errors": False,
        "errors": [],
        "warnings": [],
        "raw_output": "",
    }

    try:
        # Write requirements to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write(requirements_content)
            tmp_path = tmp.name

        # Run pip dry-run
        proc = subprocess.run(
            ["pip", "install", "--dry-run", "-r", tmp_path],
            capture_output=True, text=True, timeout=60
        )

        result["raw_output"] = proc.stderr + proc.stdout

        # Check for conflict/error indicators
        error_keywords = ["conflict", "error", "incompatible", "could not find",
                          "no matching distribution", "failed"]
        for line in result["raw_output"].lower().split("\n"):
            for keyword in error_keywords:
                if keyword in line:
                    result["has_errors"] = True
                    result["errors"].append(line.strip())
                    break

        # Check for warnings
        for line in result["raw_output"].lower().split("\n"):
            if "warning" in line or "deprecated" in line:
                result["warnings"].append(line.strip())

        # Clean up
        os.unlink(tmp_path)

    except subprocess.TimeoutExpired:
        result["has_errors"] = True
        result["errors"].append("pip dry-run timed out — complex dependency tree")
    except FileNotFoundError:
        result["warnings"].append("pip not found in PATH — skipping dry-run check")
    except Exception as e:
        result["warnings"].append(f"pip dry-run failed: {str(e)}")

    return result


def _groq_analyze(requirements_content: str, pip_output: str) -> Optional[dict]:
    """Use Groq LLM for detailed conflict analysis."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("GROQ_API_KEY not set, skipping LLM analysis")
        return None

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are a Python dependency expert. Return only valid JSON, no markdown fences or extra text.",
                },
                {
                    "role": "user",
                    "content": f"""Analyze these Python dependencies for conflicts:

requirements.txt:
{requirements_content[:3000]}

pip dry-run errors:
{pip_output[:2000]}

Return JSON: {{"conflicts": ["description of each conflict"], "warnings": ["non-critical warnings"], "clean": false}}
If no real conflicts exist, return: {{"conflicts": [], "warnings": [], "clean": true}}""",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1024,
        )

        return json.loads(response.choices[0].message.content)

    except ImportError:
        logger.warning("groq package not installed")
        return None
    except Exception as e:
        logger.error(f"Groq analysis failed: {e}")
        return None


def _empty_result(reason: str) -> dict:
    return {
        "requirements_found": False,
        "dep_files": {},
        "conflicts": [],
        "warnings": [reason],
        "clean": True,
        "analysis_method": None,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = check_compatibility("https://github.com/tensorflow/tensor2tensor")
    print(json.dumps(result, indent=2))
