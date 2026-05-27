"""
Dockerfile Validator — Component 5b

Validates generated Dockerfiles across three dimensions:
  1. Structural/syntax checks  (regex-based lint, no Docker daemon needed)
  2. Base image freshness      (Docker Hub API + hardcoded EOL table)
  3. Base–dependency conflicts (e.g. CPU-only base + CUDA packages)
  4. Real docker build test    (requires Docker daemon; skipped gracefully if absent)

Public API
----------
    validate_dockerfile(dockerfile_str, requirements_str="") -> dict
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
# Known EOL / deprecated base image tags (offline fallback)
# ---------------------------------------------------------------------------

_KNOWN_EOL_TAGS: dict[str, str] = {
    # Python EOL images
    "python:2.7": "Python 2.7 reached EOL on 2020-01-01",
    "python:3.5": "Python 3.5 reached EOL on 2020-09-13",
    "python:3.6": "Python 3.6 reached EOL on 2021-12-23",
    "python:3.7": "Python 3.7 reached EOL on 2023-06-27",
    "python:2.7-slim": "Python 2.7 reached EOL on 2020-01-01",
    "python:3.5-slim": "Python 3.5 reached EOL on 2020-09-13",
    "python:3.6-slim": "Python 3.6 reached EOL on 2021-12-23",
    "python:3.7-slim": "Python 3.7 reached EOL on 2023-06-27",
    # Ubuntu EOL
    "ubuntu:14.04": "Ubuntu 14.04 reached EOL on 2019-04-25",
    "ubuntu:16.04": "Ubuntu 16.04 reached EOL on 2021-04-30",
    "ubuntu:18.04": "Ubuntu 18.04 LTS reached end of standard support 2023-04",
    # CUDA deprecated
    "nvidia/cuda:9.0": "CUDA 9.0 is no longer supported by NVIDIA",
    "nvidia/cuda:9.2": "CUDA 9.2 is no longer supported by NVIDIA",
    "nvidia/cuda:10.0": "CUDA 10.0 is no longer supported by NVIDIA",
    "nvidia/cuda:10.1": "CUDA 10.1 is no longer supported by NVIDIA",
    "nvidia/cuda:10.2": "CUDA 10.2 is no longer supported by NVIDIA",
}

# Base images that are considered CUDA-capable
_CUDA_BASE_IMAGES = {"nvidia/cuda", "tensorflow/tensorflow", "pytorch/pytorch"}

# Packages that require CUDA at runtime
_CUDA_REQUIRING_PACKAGES = {
    "torch", "torchvision", "torchaudio",
    "tensorflow-gpu", "jax[cuda]", "cupy",
    "triton", "flash-attn",
}

# CPU-only base images (no GPU pass-through)
_CPU_ONLY_BASES = {
    "python",           # python:X.Y / python:X.Y-slim
    "ubuntu",
    "debian",
    "alpine",
    "continuumio/miniconda3",
    "continuumio/anaconda3",
}


# ---------------------------------------------------------------------------
# Public entry point — structural validation
# ---------------------------------------------------------------------------

def validate_dockerfile(dockerfile_str: str, requirements_str: str = "") -> dict:
    """
    Validate a Dockerfile string for structural correctness, base image
    freshness, and CPU/GPU mismatches.

    Args:
        dockerfile_str:    Raw Dockerfile content.
        requirements_str:  Optional requirements.txt content (used for
                           CPU/GPU mismatch detection).

    Returns:
        {
            "valid":            bool,          # False if any blocking issue
            "issues":           list[str],     # Blocking errors
            "warnings":         list[str],     # Non-blocking advisories
            "base_image":       str,           # Detected FROM image
            "base_image_fresh": bool | None,   # None = unknown
            "base_image_eol_reason": str,      # Non-empty if EOL
            "checks_run":       list[str],     # Which checks executed
        }
    """
    result = {
        "valid": True,
        "issues": [],
        "warnings": [],
        "base_image": "",
        "base_image_fresh": None,
        "base_image_eol_reason": "",
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

    # --- Check 2: Bare OS + no Python setup ---
    result["checks_run"].append("bare_os_python_setup")
    if base_image:
        _check_bare_os_python(base_image, lines, result)

    # --- Check 3: RUN chain best practices ---
    result["checks_run"].append("run_chain")
    _check_run_chains(lines, result)

    # --- Check 4: WORKDIR set ---
    result["checks_run"].append("workdir")
    _check_workdir(lines, result)

    # --- Check 5: COPY or ADD present ---
    result["checks_run"].append("copy_present")
    _check_copy_present(lines, result)

    # --- Check 6: pip install without --no-cache-dir ---
    result["checks_run"].append("pip_cache")
    _check_pip_no_cache(lines, result)

    # --- Check 7: Base image freshness (Docker Hub API) ---
    result["checks_run"].append("base_image_freshness")
    if base_image:
        freshness = check_base_image_freshness(base_image)
        result["base_image_fresh"] = freshness["fresh"]
        result["base_image_eol_reason"] = freshness.get("reason", "")
        if not freshness["fresh"]:
            result["warnings"].append(
                f"Base image '{base_image}' may be outdated or EOL: "
                f"{freshness.get('reason', 'check Docker Hub for updates')}"
            )

    # --- Check 8: CPU/GPU mismatch ---
    result["checks_run"].append("cpu_gpu_mismatch")
    if base_image and requirements_str:
        _check_cpu_gpu_mismatch(base_image, requirements_str, result)

    return result


# ---------------------------------------------------------------------------
# Base image freshness check
# ---------------------------------------------------------------------------

def check_base_image_freshness(image_tag: str) -> dict:
    """
    Check if a Docker image tag is fresh (not EOL / deprecated).

    Strategy:
        1. Check hardcoded EOL table (fast, offline)
        2. Query Docker Hub API to verify tag exists and get last push date
        3. Flag images whose last push is > 2 years ago

    Returns:
        {"fresh": bool, "reason": str, "last_pushed": str | None}
    """
    image_tag_lower = image_tag.lower().split(" ")[0]  # strip any digest

    # Normalize: remove sha256 digest if present
    if "@sha256:" in image_tag_lower:
        image_tag_lower = image_tag_lower.split("@sha256:")[0]

    # --- EOL table check ---
    for eol_key, eol_reason in _KNOWN_EOL_TAGS.items():
        if image_tag_lower.startswith(eol_key):
            return {"fresh": False, "reason": eol_reason, "last_pushed": None}

    # --- Docker Hub API check ---
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
            "log":          str,    # Last 40 lines of build output
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
            # Create a minimal placeholder so COPY requirements.txt doesn't fail
            req_path = os.path.join(tmp_dir, "requirements.txt")
            with open(req_path, "w", encoding="utf-8") as f:
                f.write("# placeholder\n")

        # Write a minimal dummy app file so COPY . . doesn't break
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
            proc.wait()  # Reap the terminated process
            timed_out = True

        t.join(timeout=5)  # Wait for reader thread to finish

        duration = time.monotonic() - start
        result["duration_s"] = round(duration, 1)
        result["log"] = "".join(build_log[-50:])

        if timed_out:
            result["error"] = "docker build timed out after 10 minutes"
            logger.error("docker build timed out")
        elif return_code == 0:
            result["success"] = True
            logger.info(f"docker build succeeded in {result['duration_s']}s")
            # Clean up the image after successful test
            _cleanup_docker_image(tag)
        else:
            # Extract the most relevant error line
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
            # Handle multi-stage: FROM image AS alias
            parts = stripped.split()
            if len(parts) >= 2:
                img = parts[1]
                # Strip variable references like ${BASE_IMAGE}
                if img.startswith("${") or img.startswith("$"):
                    continue
                return img
    return ""


def _check_bare_os_python(base_image: str, lines: list[str], result: dict) -> None:
    """
    Warn when a bare OS image (ubuntu/debian/nvidia/cuda) is used but
    the Dockerfile doesn't explicitly install Python 3 and pip.
    """
    image_lower = base_image.lower()
    is_bare_os = any(image_lower.startswith(p) for p in (
        "ubuntu", "debian", "nvidia/cuda", "centos", "fedora", "rockylinux",
    ))
    if not is_bare_os:
        return

    full_text = "\n".join(lines).lower()
    has_python_install = (
        "apt-get install" in full_text and "python3" in full_text
        or "apk add" in full_text and "python3" in full_text
        or "yum install" in full_text and "python3" in full_text
    )
    has_pip_install = "python3-pip" in full_text or "pip3 install" in full_text or "pip install" in full_text

    if not has_python_install:
        result["issues"].append(
            f"Base image '{base_image}' is a bare OS image but Python 3 is not explicitly installed. "
            "Add: RUN apt-get update && apt-get install -y python3 python3-pip"
        )
        result["valid"] = False
    elif not has_pip_install:
        result["warnings"].append(
            f"Base image '{base_image}' installs Python but pip may be missing. "
            "Ensure 'python3-pip' is installed before running pip install."
        )


def _check_run_chains(lines: list[str], result: dict) -> None:
    """
    Warn about consecutive separate RUN apt-get commands that should be chained
    to reduce image layers.
    """
    apt_run_count = sum(
        1 for l in lines
        if re.match(r"^\s*RUN\s+apt-get\s+install", l, re.IGNORECASE)
    )
    if apt_run_count >= 3:
        result["warnings"].append(
            f"Found {apt_run_count} separate 'RUN apt-get install' commands. "
            "Consider chaining them with && to reduce image layers."
        )


def _check_workdir(lines: list[str], result: dict) -> None:
    """Warn if no WORKDIR is set (common mistake)."""
    has_workdir = any(
        l.strip().upper().startswith("WORKDIR") for l in lines
    )
    if not has_workdir:
        result["warnings"].append(
            "No WORKDIR instruction found. "
            "It is best practice to set a working directory (e.g. WORKDIR /app)."
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


def _check_cpu_gpu_mismatch(
    base_image: str,
    requirements_str: str,
    result: dict,
) -> None:
    """
    Flag when a CPU-only base image is combined with GPU-requiring packages.
    """
    image_lower = base_image.lower()
    base_family = image_lower.split(":")[0].split("/")[0]

    is_cpu_only = any(image_lower.startswith(p + ":") or image_lower == p
                      for p in _CPU_ONLY_BASES)
    if not is_cpu_only:
        return

    reqs_lower = requirements_str.lower()
    cuda_pkgs_found = [
        pkg for pkg in _CUDA_REQUIRING_PACKAGES
        if pkg in reqs_lower
    ]

    if cuda_pkgs_found:
        result["warnings"].append(
            f"Potential CPU/GPU mismatch: base image '{base_image}' is CPU-only, "
            f"but requirements contain GPU packages: {', '.join(cuda_pkgs_found)}. "
            "Consider using an nvidia/cuda base image for GPU workloads."
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

    sample = """FROM python:3.8-slim

RUN apt-get update && apt-get install -y git build-essential

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "main.py"]
"""

    reqs = "torch==1.9.0\ntorchvision==0.10.0\nnumpy>=1.20\n"

    print("=== Dockerfile Validation ===")
    result = validate_dockerfile(sample, reqs)
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
