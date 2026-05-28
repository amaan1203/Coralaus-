"""
Component 7 — Final Output Assembly

Assembles all results into a structured output JSON for the UI.
Tracks Coral queries, LLM calls, and generates downloadable artifacts.

Input:  Results from all previous components
Output: Complete result JSON
"""

import os
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def build_output(
    paper_json: dict,
    search_result: dict,
    health_result: dict = None,
    compat_result: dict = None,
    resolver_result: dict = None,
    generator_result: dict = None,
    coral_queries_total: int = 0,
    llm_calls: dict = None,
    dockerfile_validation: dict = None,
    repo_relevance: dict = None,
) -> dict:
    """
    Assemble the final structured output from all component results.

    Args:
        paper_json: Parsed paper from Component 1
        search_result: Semantic Scholar + GitHub search result from Component 2
        health_result: Repo health check from Component 3
        compat_result: Compatibility check from Component 4
        resolver_result: Conflict resolution from Component 5
        generator_result: From-scratch generation from Component 6
        coral_queries_total: Total Coral queries used
        llm_calls: Dict of LLM call counts by provider
        dockerfile_validation: Dockerfile validation result from Component 5b
        repo_relevance: Repo relevance validation from Component 2b

    Returns:
        Complete output dict ready for UI display and JSON export
    """
    if llm_calls is None:
        llm_calls = {"gemini_flash": 0, "groq_llama": 0}

    implementation_found = search_result.get("found", False) if search_result else False
    generated_from_scratch = generator_result is not None and not implementation_found

    validation_data = search_result.get("validation") if search_result else None
    if not validation_data:
        validation_data = {
            "confidence_score": 0.0,
            "classification": "UNKNOWN",
            "validator_scores": {}
        }

    # Build the output
    output = {
        # Paper metadata
        "paper": {
            "title": paper_json.get("title", ""),
            "arxiv_id": paper_json.get("arxiv_id", ""),
            "authors": _format_authors(paper_json.get("authors", [])),
            "year": paper_json.get("year"),
            "abstract": paper_json.get("abstract", "")[:500],
            "parsed_by": paper_json.get("parsed_by", "unknown"),
        },

        # Implementation search
        "implementation_found": implementation_found,
        "repo_url": search_result.get("repo_url") if search_result else None,
        "search_method": search_result.get("search_method") if search_result else None,
        "validation": validation_data,

        # Health check
        "repo_health_score": health_result.get("health_score", -1) if health_result else -1,
        "health_signals": health_result.get("signals", {}) if health_result else {},
        "health_label": health_result.get("health_label", "N/A") if health_result else "N/A",
        "health_emoji": health_result.get("health_emoji", "⚪") if health_result else "⚪",

        # Compatibility
        "conflicts_found": compat_result.get("conflicts", []) if compat_result else [],
        "conflicts_resolved": resolver_result is not None and bool(resolver_result.get("dockerfile")),
        "dep_warnings": compat_result.get("warnings", []) if compat_result else [],

        # Generated artifacts
        "dockerfile": _get_dockerfile(resolver_result, generator_result),
        "implementation_script": generator_result.get("implementation_py") if generator_result else None,
        "resolved_requirements": resolver_result.get("resolved_requirements") if resolver_result else None,

        # Generation metadata
        "generated_from_scratch": generated_from_scratch,
        "keyterms": generator_result.get("keyterms", []) if generator_result else [],
        "related_repos": generator_result.get("related_repos", []) if generator_result else [],

        # Dockerfile validation (Component 5b)
        "dockerfile_validation": _safe_validation(dockerfile_validation),

        # Repo relevance (Component 2b)
        "repo_relevance": _safe_relevance(repo_relevance),

        # Stats for demo
        "coral_queries_used": coral_queries_total,
        "llm_calls": llm_calls,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return output


def save_output(output: dict, output_dir: str = "output") -> dict:
    """
    Save output artifacts to disk.

    Returns dict of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    saved = {}

    # Save full result JSON
    result_path = os.path.join(output_dir, "result.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    saved["result_json"] = result_path

    # Save Dockerfile if present
    if output.get("dockerfile"):
        dockerfile_path = os.path.join(output_dir, "Dockerfile")
        with open(dockerfile_path, "w", encoding="utf-8") as f:
            f.write(output["dockerfile"])
        saved["dockerfile"] = dockerfile_path

    # Save implementation if present
    if output.get("implementation_script"):
        impl_path = os.path.join(output_dir, "implementation.py")
        with open(impl_path, "w", encoding="utf-8") as f:
            f.write(output["implementation_script"])
        saved["implementation_py"] = impl_path

    # Save resolved requirements if present
    if output.get("resolved_requirements"):
        reqs_path = os.path.join(output_dir, "requirements_resolved.txt")
        with open(reqs_path, "w", encoding="utf-8") as f:
            f.write(output["resolved_requirements"])
        saved["requirements"] = reqs_path

    logger.info(f"Saved {len(saved)} artifacts to {output_dir}/")
    return saved


def format_summary(output: dict) -> str:
    """Generate a human-readable summary of the analysis."""
    lines = []
    lines.append(f"📄 Paper: {output['paper']['title']}")
    lines.append(f"   Year: {output['paper'].get('year', 'N/A')}")
    lines.append(f"   arXiv: {output['paper'].get('arxiv_id', 'N/A')}")
    lines.append("")

    if output["implementation_found"]:
        lines.append(f"✅ Implementation found: {output['repo_url']}")
        lines.append(f"   Search method: {output.get('search_method', 'N/A')}")
        val = output.get("validation", {})
        if val and val.get("classification") != "UNKNOWN":
            lines.append(f"   Validation: {val.get('classification')} ({val.get('confidence_score', 0.0):.1%})")
        lines.append(f"   Health: {output['health_emoji']} {output['health_label']} ({output['repo_health_score']}/100)")

        signals = output.get("health_signals", {})
        if signals.get("last_commit_days_ago") is not None:
            lines.append(f"   Last commit: {signals['last_commit_days_ago']} days ago")
        if signals.get("stars") is not None:
            lines.append(f"   Stars: {signals['stars']}")

        if output["conflicts_found"]:
            lines.append(f"\n⚠️  {len(output['conflicts_found'])} dependency conflicts found:")
            for c in output["conflicts_found"][:3]:
                lines.append(f"     - {c}")
            if output["conflicts_resolved"]:
                lines.append("   ✅ Conflicts resolved — Dockerfile generated")
        else:
            lines.append("\n✅ No dependency conflicts")

        # Dockerfile validation summary
        df_val = output.get("dockerfile_validation", {})
        if df_val:
            issues = df_val.get('issues', [])
            warnings = df_val.get('warnings', [])
            if df_val.get('valid') and not warnings:
                lines.append("\n🐳 Dockerfile: ✅ Valid")
            elif df_val.get('valid'):
                lines.append(f"\n🐳 Dockerfile: ⚠️  Valid with {len(warnings)} warning(s)")
            else:
                lines.append(f"\n🐳 Dockerfile: ❌ {len(issues)} issue(s)")
            if df_val.get('base_image'):
                fresh = '✅ fresh' if df_val.get('base_image_fresh') else '⚠️  outdated'
                lines.append(f"   Base image: {df_val['base_image']} ({fresh})")

    else:
        lines.append("❌ No existing implementation found")
        if output["generated_from_scratch"]:
            lines.append("🔧 Generated reference implementation from scratch")
            if output.get("keyterms"):
                lines.append(f"   Keywords: {', '.join(output['keyterms'][:5])}")
            if output.get("related_repos"):
                lines.append(f"   Based on: {', '.join(output['related_repos'][:3])}")

    lines.append(f"\n📊 Stats:")
    lines.append(f"   Coral queries: {output['coral_queries_used']}")
    llm = output.get("llm_calls", {})
    lines.append(f"   Gemini calls: {llm.get('gemini_flash', 0)}")
    lines.append(f"   Groq calls: {llm.get('groq_llama', 0)}")

    return "\n".join(lines)


def _format_authors(authors: list) -> list:
    """Format author list for display."""
    result = []
    for a in authors[:5]:
        if isinstance(a, dict):
            name = f"{a.get('first', '')} {a.get('last', '')}".strip()
            result.append(name or "Unknown")
        elif isinstance(a, str):
            result.append(a)
    if len(authors) > 5:
        result.append(f"et al. (+{len(authors) - 5})")
    return result


def _get_dockerfile(resolver_result: dict = None, generator_result: dict = None) -> str:
    """Get Dockerfile from either resolver or generator."""
    if resolver_result and resolver_result.get("dockerfile"):
        return resolver_result["dockerfile"]
    if generator_result and generator_result.get("dockerfile"):
        return generator_result["dockerfile"]
    return None


def _safe_validation(dockerfile_validation: dict = None) -> dict:
    """Return a safe copy of dockerfile validation, or empty dict."""
    if not dockerfile_validation:
        return {}
    return {
        "valid": dockerfile_validation.get("valid", False),
        "issues": dockerfile_validation.get("issues", []),
        "warnings": dockerfile_validation.get("warnings", []),
        "base_image": dockerfile_validation.get("base_image", ""),
        "base_image_fresh": dockerfile_validation.get("base_image_fresh"),
        "base_image_eol_reason": dockerfile_validation.get("base_image_eol_reason", ""),
        "checks_run": dockerfile_validation.get("checks_run", []),
    }


def _safe_relevance(repo_relevance: dict = None) -> dict:
    """Return a safe copy of repo relevance, or empty dict."""
    if not repo_relevance:
        return {}
    return {
        "relevance_score": repo_relevance.get("relevance_score", 0),
        "verdict": repo_relevance.get("verdict", "irrelevant"),
        "evidence": repo_relevance.get("evidence", []),
        "warnings": repo_relevance.get("warnings", []),
        "checks_run": repo_relevance.get("checks_run", []),
    }


if __name__ == "__main__":
    # Demo output
    demo_output = build_output(
        paper_json={"title": "Test Paper", "arxiv_id": "1234.5678", "authors": [{"first": "John", "last": "Doe"}], "year": 2023, "abstract": "A test paper."},
        search_result={"found": True, "repo_url": "https://github.com/test/repo", "search_method": "arxiv_id"},
        health_result={"health_score": 82, "health_label": "Healthy", "health_emoji": "🟢", "signals": {"last_commit_days_ago": 45, "stars": 3400}},
        compat_result={"conflicts": [], "warnings": [], "clean": True},
        coral_queries_total=4,
        llm_calls={"gemini_flash": 0, "groq_llama": 1},
    )
    print(format_summary(demo_output))
