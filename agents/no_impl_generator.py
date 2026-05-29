"""
Component 6 — No-Implementation Path: Generate from Scratch

When PapersWithCode returns no implementation, this component:
  6a. Extracts keyterms from abstract (Groq — fast, structured)
  6b. Searches GitHub for related repos via Coral SQL
  6c. Cross-repo JOIN — fetches requirements + logic from top 3 repos (Coral money shot)
  6d. Generates implementation using Gemini 2.0 Flash (needs full paper body)

Input:  Paper JSON (full structured paper from Component 1)
Output: Generated implementation.py + Dockerfile
"""

import os
import json
import logging
from typing import Optional
from agents.coral_utils import get_coral_client

logger = logging.getLogger(__name__)


def generate_from_scratch(paper_json: dict) -> dict:
    """
    Generate a reference implementation from scratch when no existing code is found.

    Args:
        paper_json: Full paper JSON from Component 1

    Returns:
        Dict with:
            - implementation_py (str): Generated Python implementation
            - dockerfile (str): Generated Dockerfile
            - keyterms (list): Extracted technical keywords
            - related_repos (list): Related repos found via Coral
            - coral_queries_used (int)
            - llm_calls (dict): Count by provider
    """
    coral = get_coral_client()
    llm_calls = {"gemini_flash": 0, "groq_llama": 0}
    coral_queries = 0

    # 6a. Extract keyterms from abstract
    abstract = paper_json.get("abstract", "")
    keyterms = _extract_keyterms(abstract)
    if keyterms:
        llm_calls["groq_llama"] += 1
    logger.info(f"Extracted keyterms: {keyterms}")

    # 6b. Search GitHub for related repos via Coral
    related_repos = []
    logic_files = []
    common_deps = []

    if coral.available and keyterms:
        search_query = " ".join(keyterms[:3]) + " language:python stars:>100 pushed:>2023-01-01"
        repo_result = coral.sql(f"""
            SELECT full_name, html_url, stargazers_count, description
            FROM github.search_repositories(q => '{search_query}')
            ORDER BY stargazers_count DESC
            LIMIT 3
        """)
        coral_queries += 1

        if repo_result and "results" in repo_result:
            related_repos = repo_result["results"]
            repo_names = [r.get("full_name", "") for r in related_repos if r.get("full_name")]

            # 6c. Cross-repo JOIN — THE CORAL MONEY SHOT
            if repo_names:
                for repo_name in repo_names:
                    parts = repo_name.split("/")
                    if len(parts) != 2:
                        continue
                    owner, repo = parts[0], parts[1]

                    tree_result = coral.sql(f"""
                        SELECT path
                        FROM github.trees
                        WHERE owner = '{owner}'
                          AND repo = '{repo}'
                          AND tree_sha = 'HEAD'
                          AND recursive = 'true'
                          AND (path LIKE '%requirements%'
                            OR path LIKE '%model%py'
                            OR path LIKE '%train%py'
                            OR path LIKE '%environment%'
                            OR path LIKE '%Dockerfile%')
                        LIMIT 7
                    """)
                    coral_queries += 1

                    if tree_result and "results" in tree_result:
                        for row in tree_result["results"]:
                            path = row.get("path", "")
                            if not path:
                                continue
                            content_result = coral.sql(f"""
                                SELECT content_text as content
                                FROM github.contents
                                WHERE owner = '{owner}'
                                  AND repo = '{repo}'
                                  AND path = '{path}'
                            """)
                            coral_queries += 1

                            if content_result and "results" in content_result and content_result["results"]:
                                content = content_result["results"][0].get("content") or content_result["results"][0].get("content_text")
                                if content:
                                    if "requirements" in path or "environment" in path:
                                        common_deps.append(f"# From {repo_name}/{path}\n{content[:500]}")
                                    elif "Dockerfile" in path:
                                        for line in content.splitlines():
                                            if line.strip().upper().startswith("FROM "):
                                                common_deps.append(f"# Found base image in {repo_name}: {line.strip()}")
                                                break
                                    else:
                                        logic_files.append(f"# From {repo_name}/{path}\n{content[:1000]}")

    # Fallback: search via GitHub API if Coral not available
    if not related_repos and keyterms:
        related_repos = _github_search_fallback(keyterms)

    # 6d. Generate implementation using Gemini 2.0 Flash
    generated = _generate_with_gemini(paper_json, logic_files, common_deps)
    if generated:
        llm_calls["gemini_flash"] += 1

    if not generated:
        # Fallback to Groq for shorter generation
        generated = _generate_with_groq(paper_json, logic_files, common_deps)
        if generated:
            llm_calls["groq_llama"] += 1

    if not generated:
        generated = {
            "implementation_py": _placeholder_implementation(paper_json),
            "dockerfile": _placeholder_dockerfile(paper_json),
        }

    return {
        "implementation_py": generated.get("implementation_py", ""),
        "dockerfile": generated.get("dockerfile", ""),
        "keyterms": keyterms,
        "related_repos": [r.get("full_name", "") for r in related_repos],
        "coral_queries_used": coral_queries,
        "llm_calls": llm_calls,
    }


def _extract_keyterms(abstract: str) -> list:
    """Extract technical keywords from abstract using Groq."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key or not abstract:
        return _basic_keyterm_extraction(abstract)

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": f"Extract 5-7 technical keywords from this abstract for a GitHub code search. Return only a JSON object with a 'keywords' array of strings.\n\nAbstract:\n{abstract[:2000]}",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=256,
        )

        data = json.loads(response.choices[0].message.content)
        return data.get("keywords", [])

    except Exception as e:
        logger.error(f"Groq keyterm extraction failed: {e}")
        return _basic_keyterm_extraction(abstract)


def _basic_keyterm_extraction(abstract: str) -> list:
    """Simple keyword extraction without LLM."""
    if not abstract:
        return []

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

    words = abstract.lower().split()
    words = [w.strip(".,;:!?()[]{}\"'") for w in words]
    words = [w for w in words if w not in stop_words and len(w) > 3]

    # Count frequency
    freq = {}
    for w in words:
        freq[w] = freq.get(w, 0) + 1

    # Return top keywords by frequency
    sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    return [w for w, _ in sorted_words[:7]]


def _generate_with_gemini(paper_json: dict, logic_files: list, common_deps: list) -> Optional[dict]:
    """Generate implementation using Gemini 2.0 Flash (long context)."""
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        logger.warning("GEMINI_API_KEY not set")
        return None

    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")

        # Use full_text directly if available, otherwise reconstruct from sections
        paper_sections = paper_json.get("full_text", "")
        if not paper_sections:
            paper_sections = "\n\n".join(
                f"## {s['heading']}\n{s['text']}"
                for s in paper_json.get("sections", [])
            )

        prompt = f"""You are an expert ML engineer.

Paper: {paper_json.get('title', '')}
Abstract: {paper_json.get('abstract', '')}

Full paper content:
{paper_sections[:30000]}

Related repository logic files:
{chr(10).join(logic_files[:5000])}

Common dependencies across related repos:
{chr(10).join(common_deps[:2000])}

Generate:
1. A minimal, well-commented Python implementation (implementation.py) demonstrating the core algorithm described in this paper.
2. A Dockerfile for Python 3.11 that installs all dependencies and runs it.

Return as JSON: {{"implementation_py": "...", "dockerfile": "..."}}

Important: The implementation should be runnable and demonstrate the key contribution of the paper.
Return ONLY valid JSON, no markdown fences."""

        response = model.generate_content(prompt)
        text = response.text.strip()

        # Strip markdown fences if present
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

        return json.loads(text)

    except ImportError:
        logger.warning("google-generativeai not installed")
        return None
    except Exception as e:
        logger.error(f"Gemini generation failed: {e}")
        return None


def _generate_with_groq(paper_json: dict, logic_files: list, common_deps: list) -> Optional[dict]:
    """Fallback: generate shorter implementation using Groq."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None

    try:
        from groq import Groq

        client = Groq(api_key=api_key)
        paper_content = paper_json.get("full_text", "")
        if not paper_content:
            paper_content = "\n\n".join(
                f"## {s['heading']}\n{s['text']}"
                for s in paper_json.get("sections", [])
            )
        if not paper_content:
            paper_content = paper_json.get("abstract", "")

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "system",
                    "content": "You are an ML engineer. Return only valid JSON with 'implementation_py' and 'dockerfile' keys.",
                },
                {
                    "role": "user",
                    "content": f"""Generate a minimal Python implementation for this paper.

Title: {paper_json.get('title', '')}
Abstract: {paper_json.get('abstract', '')[:1000]}

Full paper content:
{paper_content[:15000]}

Related repository logic files:
{chr(10).join(logic_files[:3000])}

Common dependencies across related repos:
{chr(10).join(common_deps[:1000])}

Return JSON: {{"implementation_py": "python code here", "dockerfile": "dockerfile content here"}}""",
                },
            ],
            response_format={"type": "json_object"},
            temperature=0.3,
            max_tokens=4096,
        )

        return json.loads(response.choices[0].message.content)

    except Exception as e:
        logger.error(f"Groq generation failed: {e}")
        return None


def _github_search_fallback(keyterms: list) -> list:
    """Search GitHub directly when Coral is not available."""
    import requests

    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token and "dummy" not in token.lower() and "your_" not in token.lower():
        headers["Authorization"] = f"token {token}"

    query = " ".join(keyterms[:3]) + " language:python"
    try:
        resp = requests.get(
            "https://api.github.com/search/repositories",
            params={"q": query, "sort": "stars", "per_page": 3},
            headers=headers, timeout=10,
        )
        if resp.status_code == 200:
            return resp.json().get("items", [])
    except Exception as e:
        logger.error(f"GitHub search fallback failed: {e}")

    return []


def _placeholder_implementation(paper_json: dict) -> str:
    return f'''"""
Reference Implementation: {paper_json.get("title", "Research Paper")}
Generated by PaperDock

NOTE: This is a placeholder. The full implementation requires
      setting up GEMINI_API_KEY for AI-powered code generation.
"""

import numpy as np

def main():
    print("Paper: {paper_json.get("title", "")}")
    print("Abstract: {paper_json.get("abstract", "")[:200]}...")
    print()
    print("TODO: Implement core algorithm from the paper")
    print("Keywords: {", ".join(paper_json.get("keywords", []))}")

if __name__ == "__main__":
    main()
'''


def _placeholder_dockerfile(paper_json: dict) -> str:
    return f"""# Dockerfile for: {paper_json.get("title", "Research Paper")}
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["python", "implementation.py"]
"""


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_paper = {
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks...",
        "sections": [{"heading": "Introduction", "text": "Recurrent neural networks..."}],
        "keywords": ["transformer", "attention", "self-attention"],
    }
    result = generate_from_scratch(test_paper)
    print(f"Keyterms: {result['keyterms']}")
    print(f"Related repos: {result['related_repos']}")
    print(f"Implementation length: {len(result['implementation_py'])} chars")
