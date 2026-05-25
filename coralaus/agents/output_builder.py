"""
Component 7 — Final Output Builder.

Assembles the results from various components into a structured JSON payload
that the Streamlit UI consumes.
"""

from typing import Any, Dict, Optional


def build_output(
    paper_json: Dict[str, Any],
    implementation_found: bool = False,
    repo_url: Optional[str] = None,
    repo_health_score: Optional[int] = None,
    health_signals: Optional[Dict[str, Any]] = None,
    conflicts_found: Optional[list[str]] = None,
    conflicts_resolved: bool = False,
    dockerfile: Optional[str] = None,
    implementation_script: Optional[str] = None,
    generated_from_scratch: bool = False,
    coral_queries_used: int = 0,
    llm_calls: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """
    Assemble the final output JSON payload.

    Args:
        paper_json: Output from Component 1 (ingest)
        implementation_found: Whether a repo was found via PapersWithCode
        repo_url: The GitHub repo URL
        repo_health_score: 0-100 score from Component 3
        health_signals: Raw metadata used to compute health
        conflicts_found: List of dependency conflicts from Component 4
        conflicts_resolved: Whether Component 5 successfully resolved them
        dockerfile: The generated Dockerfile (from Component 5 or 6)
        implementation_script: The generated code (from Component 6)
        generated_from_scratch: True if Component 6 was used
        coral_queries_used: Demo counter for Coral SQL usage
        llm_calls: Counter for Gemini and Groq usage

    Returns:
        Structured result dict matching the spec.
    """
    return {
        "paper": {
            "title": paper_json.get("title"),
            "arxiv_id": paper_json.get("arxiv_id"),
            "authors": [f"{a['first']} {a['last']}".strip() for a in paper_json.get("authors", [])],
            "year": paper_json.get("year"),
        },
        "implementation_found": implementation_found,
        "repo_url": repo_url,
        "repo_health_score": repo_health_score,
        "health_signals": health_signals or {},
        "conflicts_found": conflicts_found or [],
        "conflicts_resolved": conflicts_resolved,
        "dockerfile": dockerfile,
        "implementation_script": implementation_script,
        "generated_from_scratch": generated_from_scratch,
        "coral_queries_used": coral_queries_used,
        "llm_calls": llm_calls or {"gemini_flash": 0, "groq_llama": 0},
    }
