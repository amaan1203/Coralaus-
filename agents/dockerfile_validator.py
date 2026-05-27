"""
Dockerfile Validator — Static validation of generated Dockerfiles.

Checks the generated Dockerfile against known repository files to catch
common errors before shipping. Used as a self-healing gate in the pipeline.
"""

import logging
import re

logger = logging.getLogger(__name__)


def validate_dockerfile(dockerfile: str, dep_files: dict, readme_content: str = "") -> dict:
    """
    Validate a generated Dockerfile against known repository context.

    Args:
        dockerfile: Generated Dockerfile content
        dep_files: All dependency files found {filename: content}
        readme_content: README.md content

    Returns:
        Dict with:
            - valid (bool): True if no issues found
            - issues (list[str]): Descriptions of problems found
    """
    if not dockerfile:
        return {"valid": False, "issues": ["Dockerfile is empty"]}

    issues = []
    dockerfile_lower = dockerfile.lower()

    # Check 1: requirements.txt exists but Dockerfile doesn't install it
    has_requirements = any(
        'requirement' in k.lower() and k.lower().endswith('.txt')
        for k in dep_files
    )
    if has_requirements:
        installs_requirements = (
            'pip install' in dockerfile_lower and 'requirements' in dockerfile_lower
        ) or 'pip install -r' in dockerfile_lower
        if not installs_requirements:
            issues.append(
                "MISSING_REQUIREMENTS_INSTALL: requirements.txt exists in the repo "
                "but Dockerfile never runs `pip install -r requirements.txt`"
            )

    # Check 2: torch/tensorflow/jax in requirements but using python:slim base
    req_content = ""
    for k, v in dep_files.items():
        if 'requirement' in k.lower() and k.endswith('.txt'):
            req_content = v
            break

    gpu_frameworks = ['torch', 'tensorflow', 'jax']
    has_gpu_framework = any(fw in req_content.lower() for fw in gpu_frameworks)
    uses_slim_base = bool(re.search(r'FROM\s+python:\S*slim', dockerfile, re.IGNORECASE))

    if has_gpu_framework and uses_slim_base:
        issues.append(
            "BAD_BASE_IMAGE: GPU framework (torch/tensorflow/jax) found in requirements "
            "but Dockerfile uses python:slim base. Should use nvidia/cuda instead."
        )

    # Check 3: setup.py exists but not installed
    if 'setup.py' in dep_files:
        if 'setup.py' not in dockerfile and 'pip install .' not in dockerfile_lower and 'pip install -e' not in dockerfile_lower:
            issues.append(
                "MISSING_SETUP_INSTALL: setup.py exists in the repo but Dockerfile "
                "never copies or installs it (pip install . / pip install -e .)"
            )

    # Check 4: pyproject.toml exists but not installed
    if 'pyproject.toml' in dep_files:
        if 'pyproject.toml' not in dockerfile and 'pip install .' not in dockerfile_lower:
            issues.append(
                "MISSING_PYPROJECT_INSTALL: pyproject.toml exists in the repo but "
                "Dockerfile never copies or installs it"
            )

    # Check 5: Using bare OS image (nvidia/cuda, ubuntu) without installing python
    uses_bare_os = bool(re.search(r'FROM\s+(nvidia/cuda|ubuntu|debian)', dockerfile, re.IGNORECASE))
    if uses_bare_os:
        installs_python = (
            'python3' in dockerfile_lower and ('apt' in dockerfile_lower or 'install' in dockerfile_lower)
        )
        if not installs_python:
            issues.append(
                "MISSING_PYTHON_INSTALL: Using bare OS base image (nvidia/cuda or ubuntu) "
                "but never installs python3/python3-pip"
            )

    # Check 6: Key packages from requirements.txt are hardcoded instead of using the file
    if has_requirements and 'pip install -r' not in dockerfile_lower:
        # Count how many individual pip install lines there are
        individual_installs = re.findall(r'pip install\s+(?!-r\s)(\S+)', dockerfile)
        if len(individual_installs) > 5:
            issues.append(
                "HARDCODED_DEPS: Dockerfile has {n} individual pip install commands instead of "
                "using `pip install -r requirements.txt`. This is fragile and likely missing packages.".format(
                    n=len(individual_installs)
                )
            )

    # Check 7: No WORKDIR set
    if 'workdir' not in dockerfile_lower:
        issues.append(
            "MISSING_WORKDIR: Dockerfile does not set a WORKDIR. "
            "Should use WORKDIR /workspace or similar."
        )

    # Check 8: No COPY instruction at all
    if 'copy' not in dockerfile_lower:
        issues.append(
            "MISSING_COPY: Dockerfile has no COPY instructions. "
            "Source code will not be available in the container."
        )

    if issues:
        logger.warning(f"Dockerfile validation found {len(issues)} issues")
        for issue in issues:
            logger.warning(f"  - {issue}")
    else:
        logger.info("Dockerfile validation passed — no issues found")

    return {"valid": len(issues) == 0, "issues": issues}
