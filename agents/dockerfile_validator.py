"""
Dockerfile Validator — Component 5b

Validates generated Dockerfiles across four dimensions:
  1. Structural/syntax checks  (regex-based lint, no Docker daemon needed)
  2. Base image freshness      (Docker Hub API — no hardcoded EOL table)
  3. Base dependency conflicts (pattern-based GPU/CPU mismatch detection)
  4. Modern best-practice checks (root user, latest tag, apt cleanup, CUDA alignment)
  5. Real docker build test    (requires Docker daemon; skipped gracefully if absent)

Public API
----------
    validate_dockerfile(dockerfile_str, requirements_str="", suggested_base_image="") -> dict
    build_dockerfile_test(dockerfile_str, requirements_str="", tag="coralaus-test") -> dict
"""

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
import time
from typing import Optional
import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Pattern-based classifiers (replaces hardcoded image sets)
# ---------------------------------------------------------------------------

# Images that provide a CUDA runtime / GPU pass-through
_CUDA_IMAGE_PATTERNS: list[re.Pattern] = [
    re.compile(r"^nvidia/cuda:", re.I),
    re.compile(r"^nvcr\.io/nvidia/", re.I),
    re.compile(r"^pytorch/pytorch:", re.I),
    re.compile(r"^tensorflow/tensorflow:.*gpu", re.I),
    re.compile(r"^rocm/pytorch:", re.I),
    re.compile(r"^rocm/rocm:", re.I),
    re.compile(r"^ghcr\.io/huggingface/", re.I),
]

# Images that are CPU-only (no GPU driver pass-through)
_CPU_ONLY_PATTERNS: list[re.Pattern] = [
    re.compile(r"^python:\d+\.\d+", re.I),
    re.compile(r"^ubuntu:\d+", re.I),
    re.compile(r"^debian:(bullseye|buster|bookworm|stretch|jessie|sid)", re.I),
    re.compile(r"^alpine:\d+", re.I),
    re.compile(r"^continuumio/(mini|ana)conda", re.I),
    re.compile(r"^mambaorg/micromamba", re.I),
    re.compile(r"^slim/python", re.I),
]

# Packages that benefit from / require CUDA at runtime
_CUDA_REQUIRING_PACKAGES = {
    "torch", "torchvision", "torchaudio",
    "tensorflow-gpu", "jax[cuda]", "jaxlib",
    "cupy", "cupy-cuda", "triton", "flash-attn",
    "xformers", "bitsandbytes",
}

# Known CUDA major versions that ship with each PyTorch major.minor
# Source: https://pytorch.org/get-started/previous-versions/
_TORCH_CUDA_COMPAT: dict[str, list[str]] = {
    "2.3": ["12.1", "11.8"],
    "2.2": ["12.1", "11.8"],
    "2.1": ["12.1", "11.8"],
    "2.0": ["11.7", "11.8"],
    "1.13": ["11.6", "11.7"],
    "1.12": ["11.3", "11.6"],
    "1.11": ["11.3", "11.5"],
    "1.10": ["11.1", "11.3"],
    "1.9":  ["11.1", "10.2"],
    "1.8":  ["11.1", "10.2", "10.1"],
    "1.7":  ["11.0", "10.2", "10.1"],
}


def _is_cuda_image(image: str) -> bool:
    return any(p.search(image) for p in _CUDA_IMAGE_PATTERNS)


def _is_cpu_only_image(image: str) -> bool:
    return any(p.search(image) for p in _CPU_ONLY_PATTERNS)


# ---------------------------------------------------------------------------
# Public entry point — structural validation
# ---------------------------------------------------------------------------

def validate_dockerfile(
    dockerfile_str: str,
    requirements_str: str = "",
    suggested_base_image: str = "",
) -> dict:
    """
    Validate a Dockerfile string for structural correctness, base image
    freshness, and CPU/GPU mismatches.

    Args:
        dockerfile_str:       Raw Dockerfile content.
        requirements_str:     Optional requirements.txt content (used for
                              CPU/GPU mismatch and CUDA alignment checks).
        suggested_base_image: The base image recommended by the agent's
                              _select_base_image() step. When provided the
                              validator skips the EOL check for this image
                              (the agent already validated it) and uses it
                              as the authoritative GPU/CPU context.

    Returns:
        {
            "valid":                 bool,         # False if any blocking issue
            "issues":                list[str],    # Blocking errors
            "warnings":              list[str],    # Non-blocking advisories
            "base_image":            str,          # Detected FROM image
            "base_image_fresh":      bool | None,  # None = unknown
            "base_image_eol_reason": str,          # Non-empty if EOL/stale
            "suggested_match":       bool,         # True if FROM == suggested
            "checks_run":            list[str],    # Which checks executed
        }
    """
    result = {
        "valid": True,
        "issues": [],
        "warnings": [],
        "base_image": "",
        "base_image_fresh": None,
        "base_image_eol_reason": "",
        "suggested_match": False,
        "checks_run": [],
    }

    if not dockerfile_str or not dockerfile_str.strip():
        result["valid"] = False
        result["issues"].append("Dockerfile is empty")
        return result

    lines = dockerfile_str.splitlines()

    # --- Check 1: FROM present ---
    result["checks_run"].append("from_present")
    base_image = _extract_base_image(lines)
    if not base_image:
        result["valid"] = False
        result["issues"].append("No FROM instruction found — Dockerfile is invalid")
    else:
        result["base_image"] = base_image

    # --- Check 2: Suggested base image match ---
    result["checks_run"].append("suggested_base_match")
    if suggested_base_image and base_image:
        if base_image.lower().strip() == suggested_base_image.lower().strip():
            result["suggested_match"] = True
            result["warnings"]  # no warning needed — it matched
        else:
            result["suggested_match"] = False
            result["warnings"].append(
                f"Generated base image '{base_image}' differs from agent suggestion "
                f"'{suggested_base_image}'. The self-healing pass may fix this."
            )

    # --- Check 3: :latest tag (non-reproducible) ---
    result["checks_run"].append("latest_tag")
    if base_image:
        _check_latest_tag(base_image, result)

    # --- Check 4: Bare OS + no Python setup ---
    result["checks_run"].append("bare_os_python_setup")
    if base_image:
        _check_bare_os_python(base_image, lines, result)

    # --- Check 5: RUN chain best practices ---
    result["checks_run"].append("run_chain")
    _check_run_chains(lines, result)

    # --- Check 6: WORKDIR set ---
    result["checks_run"].append("workdir")
    _check_workdir(lines, result)

    # --- Check 7: COPY or ADD present ---
    result["checks_run"].append("copy_present")
    _check_copy_present(lines, result)

    # --- Check 8: pip install without --no-cache-dir ---
    result["checks_run"].append("pip_cache")
    _check_pip_no_cache(lines, result)

    # --- Check 9: apt-get cleanup missing ---
    result["checks_run"].append("apt_cleanup")
    _check_apt_cleanup(lines, result)

    # --- Check 10: Root user (security) ---
    result["checks_run"].append("root_user")
    _check_root_user(lines, result)

    # --- Check 11: Base image freshness (Docker Hub API) ---
    result["checks_run"].append("base_image_freshness")
    if base_image:
        # Skip freshness check if agent explicitly chose this image
        if suggested_base_image and base_image.lower() == suggested_base_image.lower():
            result["base_image_fresh"] = True
            result["base_image_eol_reason"] = ""
        else:
            freshness = check_base_image_freshness(base_image)
            result["base_image_fresh"] = freshness["fresh"]
            result["base_image_eol_reason"] = freshness.get("reason", "")
            if not freshness["fresh"]:
                result["warnings"].append(
                    f"Base image '{base_image}' may be outdated or EOL: "
                    f"{freshness.get('reason', 'check Docker Hub for updates')}"
                )

    # --- Check 12: CPU/GPU mismatch ---
    result["checks_run"].append("cpu_gpu_mismatch")
    if base_image and requirements_str:
        _check_cpu_gpu_mismatch(base_image, requirements_str, result)

    # --- Check 13: CUDA / PyTorch version alignment ---
    result["checks_run"].append("cuda_torch_alignment")
    if base_image and requirements_str:
        _check_cuda_torch_alignment(base_image, requirements_str, result)

    return result


# ---------------------------------------------------------------------------
# Base image freshness check (Docker Hub API — no hardcoded EOL table)
# ---------------------------------------------------------------------------

def check_base_image_freshness(image_tag: str) -> dict:
    """
    Check if a Docker image tag is fresh (not EOL / deprecated).

    Strategy:
        1. Query Docker Hub API to verify tag exists and get last push date
        2. Flag images whose last push is > 2 years ago
        3. Fall back to 'fresh=True, reason=""' on any API failure

    Returns:
        {"fresh": bool, "reason": str, "last_pushed": str | None}
    """
    image_tag_lower = image_tag.lower().split(" ")[0]

    # Normalize: remove sha256 digest if present
    if "@sha256:" in image_tag_lower:
        image_tag_lower = image_tag_lower.split("@sha256:")[0]

    # Skip check for private / non-Hub registries
    non_hub_prefixes = ("nvcr.io/", "ghcr.io/", "mcr.microsoft.com/", "gcr.io/", "quay.io/")
    if any(image_tag_lower.startswith(p) for p in non_hub_prefixes):
        return {"fresh": True, "reason": "Non-Docker Hub registry — skipping freshness check", "last_pushed": None}

    # Docker Hub API check
    try:
        name, tag = _split_image_tag(image_tag_lower)
        hub_result = _query_docker_hub(name, tag)
        if hub_result:
            return hub_result
    except Exception as e:
        logger.debug(f"Docker Hub API check failed for {image_tag}: {e}")

    return {"fresh": True, "reason": "", "last_pushed": None}


def _split_image_tag(image_tag: str):
    """Split 'repo/name:tag' into ('repo/name', 'tag')."""
    if ":" in image_tag.split("/")[-1]:
        parts = image_tag.rsplit(":", 1)
        return parts[0], parts[1]
    return image_tag, "latest"


def _query_docker_hub(name: str, tag: str) -> Optional[dict]:
    """
    Query Docker Hub v2 API to check if an image tag exists and its age.
    Returns None if the API request fails.
    """
    # Normalize official images (e.g. 'python' -> 'library/python')
    if "/" not in name:
        name = f"library/{name}"

    url = f"https://hub.docker.com/v2/repositories/{name}/tags/{tag}/"
    try:
        req = urllib.request.Request(
            url,
            headers={"Accept": "application/json",
                     "User-Agent": "Coralaus/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode())

        last_pushed = data.get("tag_last_pushed") or data.get("last_updated") or ""
        tag_status = data.get("tag_status", "")

        # Docker Hub marks retired tags
        if tag_status in ("retired", "inactive"):
            return {
                "fresh": False,
                "reason": f"Docker Hub reports tag '{tag}' as '{tag_status}'",
                "last_pushed": last_pushed,
            }

        # Check age (> 2 years old = advisory warning)
        if last_pushed:
            age_days = _days_since_iso(last_pushed)
            if age_days and age_days > 730:
                return {
                    "fresh": False,
                    "reason": (
                        f"Image tag '{tag}' was last updated {age_days} days ago "
                        f"(> 2 years). Consider updating to a newer tag."
                    ),
                    "last_pushed": last_pushed,
                }

        return {"fresh": True, "reason": "", "last_pushed": last_pushed}

    except urllib.error.HTTPError as e:
        if e.code == 404:
            return {
                "fresh": False,
                "reason": f"Image tag '{tag}' not found on Docker Hub — it may have been removed",
                "last_pushed": None,
            }
        return None
    except Exception:
        return None


def _days_since_iso(iso_str: str) -> Optional[int]:
    """Return days since an ISO 8601 date string, or None on parse failure."""
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        now = datetime.now(timezone.utc)
        return max(0, (now - dt).days)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# docker build test
# ---------------------------------------------------------------------------

def build_dockerfile_test(
    dockerfile_str: str,
    requirements_str: str = "",
    tag: str = "coralaus-build-test",
) -> dict:
    """
    Write the Dockerfile (and optional requirements.txt) to a temp directory
    and run `docker build`. Requires Docker daemon to be running.

    Returns:
        {
            "success":      bool,
            "available":    bool,   # False if Docker not found/running
            "duration_s":   float,
            "log":          str,    # Last 50 lines of build output
            "error":        str,    # Error summary if failed
        }
    """
    result = {
        "success": False,
        "available": False,
        "duration_s": 0.0,
        "log": "",
        "error": "",
    }

    # Check Docker availability
    if not shutil.which("docker"):
        result["error"] = "Docker is not installed or not in PATH"
        logger.info("Docker build test skipped: docker not found in PATH")
        return result

    try:
        probe = subprocess.run(
            ["docker", "info"],
            capture_output=True, text=True, timeout=10
        )
        if probe.returncode != 0:
            result["error"] = "Docker daemon is not running"
            logger.info("Docker build test skipped: daemon not running")
            return result
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        result["error"] = f"Docker unavailable: {e}"
        return result

    result["available"] = True

    # Write build context to temp dir
    tmp_dir = tempfile.mkdtemp(prefix="coralaus_docker_build_")
    try:
        dockerfile_path = os.path.join(tmp_dir, "Dockerfile")
        with open(dockerfile_path, "w", encoding="utf-8") as f:
            f.write(dockerfile_str)

        if requirements_str:
            req_path = os.path.join(tmp_dir, "requirements.txt")
            with open(req_path, "w", encoding="utf-8") as f:
                f.write(requirements_str)
        else:
            req_path = os.path.join(tmp_dir, "requirements.txt")
            with open(req_path, "w", encoding="utf-8") as f:
                f.write("# placeholder\n")

        with open(os.path.join(tmp_dir, "main.py"), "w") as f:
            f.write('print("coralaus build test")\n')

        logger.info(f"Running docker build in {tmp_dir} with tag {tag}")
        import threading
        start = time.monotonic()

        proc = subprocess.Popen(
            ["docker", "build", "--no-cache", "-t", tag, tmp_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        build_log = []

        def reader_thread(stream, log):
            try:
                for line in iter(stream.readline, ""):
                    stripped = line.rstrip()
                    print(f"  [Docker] {stripped}", flush=True)
                    log.append(line)
            except Exception:
                pass
            finally:
                try:
                    stream.close()
                except Exception:
                    pass

        t = threading.Thread(target=reader_thread, args=(proc.stdout, build_log))
        t.daemon = True
        t.start()

        timed_out = False
        try:
            return_code = proc.wait(timeout=600)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            timed_out = True

        t.join(timeout=5)

        duration = time.monotonic() - start
        result["duration_s"] = round(duration, 1)
        result["log"] = "".join(build_log[-50:])

        if timed_out:
            result["error"] = "docker build timed out after 10 minutes"
            logger.error("docker build timed out")
        elif return_code == 0:
            result["success"] = True
            logger.info(f"docker build succeeded in {result['duration_s']}s")
            _cleanup_docker_image(tag)
        else:
            log_lines = [l.strip() for l in build_log if l.strip()]
            error_lines = [
                l for l in log_lines
                if any(kw in l.lower() for kw in ("error", "failed", "exit code", "returned a non-zero"))
            ]
            result["error"] = error_lines[-1] if error_lines else "Build failed (see log)"
            logger.warning(f"docker build failed: {result['error']}")

    except Exception as e:
        result["error"] = f"Build test error: {e}"
        logger.error(f"Docker build test error: {e}")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return result


def _cleanup_docker_image(tag: str) -> None:
    """Remove a locally built test image to avoid disk clutter."""
    try:
        subprocess.run(
            ["docker", "rmi", "-f", tag],
            capture_output=True, timeout=30,
        )
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Structural lint helpers
# ---------------------------------------------------------------------------

def _extract_base_image(lines: list[str]) -> str:
    """Return the first FROM image (ignoring comments and ARG lines)."""
    for line in lines:
        stripped = line.strip()
        if stripped.upper().startswith("FROM "):
            parts = stripped.split()
            if len(parts) >= 2:
                img = parts[1]
                # Skip variable references like ${BASE_IMAGE}
                if img.startswith("${") or img.startswith("$"):
                    continue
                return img
    return ""


def _check_latest_tag(base_image: str, result: dict) -> None:
    """
    Warn when FROM uses :latest — non-reproducible builds.
    This applies whether the image is agent-suggested or not.
    """
    if base_image.endswith(":latest") or (":" not in base_image.split("/")[-1]):
        result["warnings"].append(
            f"Base image '{base_image}' uses 'latest' or has no tag — builds are "
            "non-reproducible. Pin to a specific version tag (e.g. python:3.11-slim)."
        )


def _check_bare_os_python(base_image: str, lines: list[str], result: dict) -> None:
    """
    Warn when a bare OS image (ubuntu/debian/nvidia/cuda etc.) is used but
    the Dockerfile doesn't explicitly install Python 3 and pip.

    Fix vs. old version: CUDA images are not considered an error when Python
    is not installed — we just warn (not block), because the CUDA image itself
    may already contain Python in some variants. Only pure ubuntu/debian/centos
    images that clearly omit Python become blocking errors.
    """
    image_lower = base_image.lower()

    is_cuda = _is_cuda_image(base_image)
    is_bare_os = (
        not is_cuda
        and any(image_lower.startswith(p) for p in (
            "ubuntu", "debian", "centos", "fedora", "rockylinux", "amazonlinux",
        ))
    )

    if not is_bare_os and not is_cuda:
        return

    full_text = "\n".join(lines).lower()
    has_python_install = (
        ("apt-get install" in full_text and "python3" in full_text)
        or ("apk add" in full_text and "python3" in full_text)
        or ("yum install" in full_text and "python3" in full_text)
        or ("dnf install" in full_text and "python3" in full_text)
        or ("conda install" in full_text)
        or ("miniconda" in image_lower)
        or ("anaconda" in image_lower)
    )
    has_pip = (
        "python3-pip" in full_text
        or "pip3 install" in full_text
        or "pip install" in full_text
        or "get-pip.py" in full_text
    )

    if is_bare_os and not has_python_install:
        # Blocking for plain OS images
        result["issues"].append(
            f"Base image '{base_image}' is a bare OS image but Python 3 is not explicitly installed. "
            "Add: RUN apt-get update && apt-get install -y python3 python3-pip"
        )
        result["valid"] = False
    elif is_cuda and not has_python_install:
        # Advisory only for CUDA images — many variants include Python
        result["warnings"].append(
            f"Base image '{base_image}' is a CUDA image. Ensure the chosen variant includes "
            "Python (e.g. use a '-runtime' or '-devel' tag that bundles Python), or add: "
            "RUN apt-get update && apt-get install -y python3 python3-pip"
        )
    elif has_python_install and not has_pip:
        result["warnings"].append(
            f"Base image '{base_image}' installs Python but pip may be missing. "
            "Ensure 'python3-pip' is installed before running pip install."
        )


def _check_run_chains(lines: list[str], result: dict) -> None:
    """
    Warn about consecutive separate RUN apt-get commands that should be chained.
    Threshold lowered to 2 (industry standard practice).
    """
    apt_run_count = sum(
        1 for l in lines
        if re.match(r"^\s*RUN\s+apt-get\s+install", l, re.IGNORECASE)
    )
    if apt_run_count >= 2:
        result["warnings"].append(
            f"Found {apt_run_count} separate 'RUN apt-get install' commands. "
            "Chain them with && to reduce image layers and image size."
        )


def _check_workdir(lines: list[str], result: dict) -> None:
    """Warn if no WORKDIR is set (common mistake)."""
    has_workdir = any(
        l.strip().upper().startswith("WORKDIR") for l in lines
    )
    if not has_workdir:
        result["warnings"].append(
            "No WORKDIR instruction found. "
            "Best practice: set a working directory (e.g. WORKDIR /app)."
        )


def _check_copy_present(lines: list[str], result: dict) -> None:
    """Warn if no COPY/ADD is present (the image would have no application code)."""
    has_copy = any(
        l.strip().upper().startswith("COPY") or l.strip().upper().startswith("ADD ")
        for l in lines
    )
    if not has_copy:
        result["warnings"].append(
            "No COPY or ADD instruction found. The image will not contain any application files."
        )


def _check_pip_no_cache(lines: list[str], result: dict) -> None:
    """Warn when pip install is used without --no-cache-dir (bloats image size)."""
    for line in lines:
        stripped = line.strip()
        if "pip install" in stripped and "--no-cache-dir" not in stripped:
            result["warnings"].append(
                "pip install without --no-cache-dir detected. "
                "Add '--no-cache-dir' to reduce image size."
            )
            break  # Only warn once


def _check_apt_cleanup(lines: list[str], result: dict) -> None:
    """
    Warn if apt-get install is used but apt lists are not cleaned up afterwards.
    Missing cleanup adds ~50-100 MB to the image unnecessarily.
    """
    full_text = "\n".join(lines)
    has_apt_install = bool(re.search(r"apt-get\s+install", full_text, re.IGNORECASE))
    has_cleanup = "rm -rf /var/lib/apt/lists" in full_text

    if has_apt_install and not has_cleanup:
        result["warnings"].append(
            "apt-get install detected but '/var/lib/apt/lists/*' not cleaned up. "
            "Add '&& rm -rf /var/lib/apt/lists/*' after apt-get install to reduce image size."
        )


def _check_root_user(lines: list[str], result: dict) -> None:
    """
    Warn if the final image runs as root (no USER instruction).
    Running as root in containers is a security anti-pattern.
    """
    has_user = any(
        l.strip().upper().startswith("USER ") for l in lines
    )
    if not has_user:
        result["warnings"].append(
            "No USER instruction found — the container will run as root. "
            "Consider adding 'USER 1000' or a named non-root user for security."
        )


def _check_cpu_gpu_mismatch(
    base_image: str,
    requirements_str: str,
    result: dict,
) -> None:
    """
    Flag when a CPU-only base image is combined with GPU-requiring packages.
    Uses regex pattern classifiers instead of hardcoded sets.
    """
    if not _is_cpu_only_image(base_image):
        return  # CUDA or unknown image — skip

    reqs_lower = requirements_str.lower()
    cuda_pkgs_found = [
        pkg for pkg in _CUDA_REQUIRING_PACKAGES
        if pkg in reqs_lower
    ]

    if cuda_pkgs_found:
        result["warnings"].append(
            f"Potential CPU/GPU mismatch: base image '{base_image}' appears to be CPU-only, "
            f"but requirements contain GPU packages: {', '.join(cuda_pkgs_found)}. "
            "Consider using an nvidia/cuda or pytorch/pytorch base image for GPU workloads."
        )


def _check_cuda_torch_alignment(
    base_image: str,
    requirements_str: str,
    result: dict,
) -> None:
    """
    Check that the CUDA version in the base image is compatible with the
    PyTorch version in requirements. Mismatches cause silent runtime failures.
    """
    if not _is_cuda_image(base_image):
        return

    # Extract CUDA version from base image tag (e.g. nvidia/cuda:11.8.0-...)
    cuda_match = re.search(r"cuda[:/]?(\d+\.\d+)", base_image, re.I)
    if not cuda_match:
        return
    cuda_version = cuda_match.group(1)  # e.g. "11.8"

    # Extract torch version from requirements
    torch_match = re.search(r"torch==(\d+\.\d+)", requirements_str, re.I)
    if not torch_match:
        return
    torch_version = torch_match.group(1)  # e.g. "2.0"

    # Find the torch major.minor key (e.g. "2.0")
    torch_key = None
    for key in _TORCH_CUDA_COMPAT:
        if torch_version.startswith(key):
            torch_key = key
            break

    if torch_key is None:
        return  # Unknown torch version — skip

    compatible_cudas = _TORCH_CUDA_COMPAT[torch_key]
    cuda_major_minor = ".".join(cuda_version.split(".")[:2])

    compatible = any(
        cuda_major_minor.startswith(c) or c.startswith(cuda_major_minor)
        for c in compatible_cudas
    )
    if not compatible:
        result["warnings"].append(
            f"CUDA/PyTorch version mismatch: base image uses CUDA {cuda_version} "
            f"but torch=={torch_version} requires CUDA {' or '.join(compatible_cudas)}. "
            "This may cause silent runtime failures. Align the CUDA version."
        )


# ---------------------------------------------------------------------------
# Utility: pretty summary
# ---------------------------------------------------------------------------

def format_validation_summary(result: dict) -> str:
    """Return a human-readable single-line summary of a validation result."""
    if result["valid"] and not result["warnings"]:
        return f"✅ Valid — base image: {result['base_image']}"
    if not result["valid"]:
        return (
            f"❌ Invalid — {len(result['issues'])} issue(s), "
            f"{len(result['warnings'])} warning(s) — base: {result['base_image']}"
        )
    return (
        f"⚠️  Valid with {len(result['warnings'])} warning(s) — base: {result['base_image']}"
    )


# ---------------------------------------------------------------------------
# CLI / standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    sample = """FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

RUN apt-get update && apt-get install -y python3 python3-pip git build-essential

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python3", "main.py"]
"""

    reqs = "torch==2.0.1\ntorchvision==0.15.2\nnumpy>=1.20\n"

    print("=== Dockerfile Validation ===")
    result = validate_dockerfile(sample, reqs, suggested_base_image="nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04")
    print(json.dumps(result, indent=2))

    print("\n=== Summary ===")
    print(format_validation_summary(result))

    if "--build" in sys.argv:
        print("\n=== Docker Build Test ===")
        build_result = build_dockerfile_test(sample, reqs)
        print(json.dumps({k: v for k, v in build_result.items() if k != "log"}, indent=2))
        if build_result.get("log"):
            print("\n--- Build Log (last lines) ---")
            print(build_result["log"][-2000:])
