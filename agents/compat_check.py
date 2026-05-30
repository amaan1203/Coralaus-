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
import re
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

    # Fallbacks for repositories with no dependency files
    warnings = []
    reconstructed = False
    if not dep_files:
        # Fix A: Try README parsing
        dep_files = _resolve_linked_repo_deps(owner, repo)
        if dep_files:
            warnings.append("No dependency files found at root or subdirectories. Retrieved dependencies from linked repository.")
            reconstructed = True

    if not dep_files:
        # Fix B: Try AST import scanner
        dep_files = _ast_scan_imports(owner, repo)
        if dep_files:
            warnings.append("No dependency files found. Reconstructed requirements via AST import scanner.")
            reconstructed = True

    if not dep_files:
        return _empty_result("No dependency files found in repository")

    # Detect the repo's actual entrypoint script
    entrypoint = _detect_entrypoint(owner, repo)

    # Fetch README for Dockerfile build instructions (e.g. "install open_spiel from source")
    readme_content = _fetch_readme(owner, repo)

    result = {
        "requirements_found": True,
        "dep_files": dict(dep_files),  # Pass full content — LLM needs complete dependency lists
        "conflicts": [],
        "warnings": warnings,
        "clean": not reconstructed,  # If reconstructed, mark as not clean to trigger Dockerfile gen
        "analysis_method": "reconstructed" if reconstructed else None,
        "entrypoint": entrypoint,
        "readme_content": readme_content[:5000] if readme_content else "",
    }

    # Step 2: Try automated pip dry-run on all requirement files
    # Find all requirement files (root or subdirectory)
    requirements_content = ""
    req_file_used = None
    for name, content in dep_files.items():
        if 'requirement' in name.lower() and name.lower().endswith('.txt'):
            requirements_content = content
            req_file_used = name
            break

    if requirements_content:
        logger.info(f"Running pip dry-run on: {req_file_used}")
        pip_result = _pip_dry_run(requirements_content)
        if pip_result["has_errors"] or reconstructed:
            result["conflicts"] = pip_result["errors"] if pip_result["has_errors"] else ["Reconstructed requirements list - Dockerfile generation requested."]
            result["warnings"].extend(pip_result["warnings"])
            result["clean"] = False if reconstructed else not pip_result["has_errors"]
            result["analysis_method"] = "pip_dry_run" if pip_result["has_errors"] else "reconstructed"
            result["warnings"].insert(0, f"Dependency file used: {req_file_used}")

            # Step 3: If conflicts found, get detailed analysis from Groq
            groq_analysis = _groq_analyze(requirements_content, pip_result["raw_output"])
            if groq_analysis:
                result["conflicts"] = groq_analysis.get("conflicts", result["conflicts"])
                result["warnings"] = groq_analysis.get("warnings", result["warnings"])
                result["clean"] = False if reconstructed else groq_analysis.get("clean", False)
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
    # Step 1: Check root-level standard filenames with exact path= queries
    for filename in ['requirements.txt', 'requirement.txt', 'environment.yml', 'setup.py', 'pyproject.toml']:
        result = coral.sql(f"""
            SELECT content_text as content
            FROM github.contents
            WHERE owner = '{owner}'
              AND repo = '{repo}'
              AND path = '{filename}'
        """)
        if result and "results" in result and result["results"]:
            content = result["results"][0].get("content") or result["results"][0].get("content_text")
            if content:
                dep_files[filename] = content

    # Step 2: If no requirements file found at root, search entire tree for any *requirement*.txt
    # This catches subdirectory files and non-standard naming (e.g. requirements_train.txt)
    if not any(k for k in dep_files if 'requirement' in k.lower() and k.endswith('.txt')):
        logger.info(f"No requirements at root for {owner}/{repo}. Searching full tree via github.trees...")
        try:
            tree_result = coral.sql(f"""
                SELECT path
                FROM github.trees
                WHERE owner = '{owner}'
                  AND repo = '{repo}'
                  AND tree_sha = 'HEAD'
                  AND recursive = 'true'
                  AND type = 'blob'
                  AND path LIKE '%requirement%'
            """)
            if tree_result and "results" in tree_result and tree_result["results"]:
                found_paths = [
                    item.get("path") for item in tree_result["results"]
                    if item.get("path", "").endswith(".txt")
                ]
                logger.info(f"github.trees found requirement files: {found_paths}")
                for path in found_paths[:3]:  # limit to avoid rate limits
                    content_result = coral.sql(f"""
                        SELECT content_text as content
                        FROM github.contents
                        WHERE owner = '{owner}'
                          AND repo = '{repo}'
                          AND path = '{path}'
                    """)
                    if content_result and "results" in content_result and content_result["results"]:
                        content = content_result["results"][0].get("content") or content_result["results"][0].get("content_text")
                        if content:
                            dep_files[path] = content
                            logger.info(f"Fetched {path} from tree search ({len(content)} bytes)")
        except Exception as e:
            logger.debug(f"Coral tree search for requirements failed: {e}")

    return dep_files


def _fetch_dep_files_github(owner: str, repo: str) -> dict:
    """Fallback: fetch dependency files from GitHub API."""
    import requests

    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token and "dummy" not in token.lower() and "your_" not in token.lower():
        headers["Authorization"] = f"token {token}"

    dep_files = {}
    # Include 'requirement.txt' (without 's') — common in some ML repos
    targets = ["requirements.txt", "requirement.txt", "environment.yml", "setup.py", "pyproject.toml"]

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

    # If no requirements file found at root, search subdirectories
    if not any(k for k in dep_files if 'requirement' in k.lower()):
        logger.info("No requirements at root. Searching subdirectories...")
        sub_reqs = _search_subdirs_for_requirements(owner, repo, headers)
        dep_files.update(sub_reqs)

    return dep_files


def _search_subdirs_for_requirements(owner: str, repo: str, headers: dict) -> dict:
    """Search subdirectories for requirement files using GitHub API tree endpoint."""
    import requests

    dep_files = {}
    try:
        # Use the Git tree API to search the whole repo at once
        resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1",
            headers=headers, timeout=15
        )
        if resp.status_code != 200:
            return dep_files

        tree = resp.json().get("tree", [])
        req_files = [
            item["path"] for item in tree
            if item["type"] == "blob"
            and any(name in item["path"].lower().split("/")[-1]
                    for name in ["requirements.txt", "requirement.txt",
                                 "requirements_dev.txt", "requirements-dev.txt"])
        ]

        if req_files:
            logger.info(f"Found requirement files in subdirs: {req_files}")
            # Fetch each file's content (limit to first 3 to avoid rate limits)
            for path in req_files[:3]:
                try:
                    file_resp = requests.get(
                        f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
                        headers=headers, timeout=10
                    )
                    if file_resp.status_code == 200:
                        data = file_resp.json()
                        if data.get("encoding") == "base64":
                            import base64
                            content = base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                            # Store with the full path as the key so we know where it came from
                            dep_files[path] = content
                            logger.info(f"  Fetched {path} ({len(content)} bytes)")
                except Exception as e:
                    logger.debug(f"Could not fetch {path}: {e}")

    except Exception as e:
        logger.debug(f"Subdirectory search failed: {e}")

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
            capture_output=True, text=True, encoding="utf-8", timeout=60
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


def _detect_entrypoint(owner: str, repo: str) -> str:
    """
    Detect the main entrypoint script of a repo by checking for common filenames.
    Uses Coral first, falls back to GitHub API. Returns the best-guess entrypoint
    filename (e.g. 'main.py', 'train.py') or None if nothing obvious is found.
    """
    # Ordered by priority — most common ML repo entrypoints first
    candidates = [
        "main.py", "train.py", "run.py", "app.py", "demo.py",
        "run_experiment.py", "train_model.py", "evaluate.py", "test.py",
    ]

    coral = get_coral_client()
    if coral.available:
        # github.contents requires WHERE path = <constant> — IN (...) is NOT supported
        # Query each candidate individually and stop at the first hit
        for candidate in candidates:
            result = coral.sql(f"""
                SELECT path
                FROM github.contents
                WHERE owner = '{owner}'
                  AND repo = '{repo}'
                  AND path = '{candidate}'
            """)
            if result and "results" in result and result["results"]:
                logger.info(f"Detected entrypoint via Coral: {candidate}")
                return candidate

    # Fallback: check via GitHub API
    try:
        import requests
        headers = {}
        token = os.environ.get("GITHUB_TOKEN")
        if token and "dummy" not in token.lower() and "your_" not in token.lower():
            headers["Authorization"] = f"token {token}"

        resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/contents/",
            headers=headers, timeout=10
        )
        if resp.status_code == 200:
            root_files = [item["name"] for item in resp.json() if item["type"] == "file"]
            for candidate in candidates:
                if candidate in root_files:
                    logger.info(f"Detected entrypoint via GitHub API: {candidate}")
                    return candidate
    except Exception as e:
        logger.debug(f"Entrypoint detection via GitHub API failed: {e}")

    logger.info("No standard entrypoint detected — will let Dockerfile generator decide")
    return None


def _fetch_file_content_github(owner: str, repo: str, path: str) -> Optional[str]:
    """Helper to fetch a file's content directly from the GitHub API using requests."""
    import requests
    headers = {}
    token = os.environ.get("GITHUB_TOKEN")
    if token and "dummy" not in token.lower() and "your_" not in token.lower():
        headers["Authorization"] = f"token {token}"
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/contents/{path}",
            headers=headers, timeout=10
        )
        if resp.status_code == 200:
            data = resp.json()
            if isinstance(data, dict):
                if data.get("encoding") == "base64":
                    import base64
                    return base64.b64decode(data["content"]).decode("utf-8", errors="replace")
                elif data.get("download_url"):
                    dl_resp = requests.get(data["download_url"], timeout=10)
                    if dl_resp.status_code == 200:
                        return dl_resp.text
    except Exception as e:
        logger.debug(f"GitHub API fetch of {path} failed: {e}")
    return None


def _resolve_linked_repo_deps(owner: str, repo: str) -> dict:
    """Fix A: Fetch and parse README.md to find linked repositories, then fetch their dependencies via Coral or GitHub API."""
    coral = get_coral_client()
    readme_content = None

    logger.info(f"Fix A: No dependency files found. Querying README.md for {owner}/{repo}...")
    
    if coral.available:
        try:
            sql = f"""
                SELECT content_text as content
                FROM github.contents
                WHERE owner = '{owner}'
                  AND repo = '{repo}'
                  AND path = 'README.md'
            """
            res = coral.sql(sql)
            if res and "results" in res and res["results"]:
                readme_content = res["results"][0].get("content", "")
        except Exception as e:
            logger.debug(f"Coral query for README.md failed: {e}")

    if not readme_content:
        logger.info("README.md not found via Coral. Falling back to GitHub API...")
        readme_content = _fetch_file_content_github(owner, repo, "README.md")

    if not readme_content:
        logger.info("No README.md found.")
        return {}

    # Match patterns like github.com/owner/repo
    urls = re.findall(r'github\.com/([a-zA-Z0-9_\-\.]+)/([a-zA-Z0-9_\-\.]+)', readme_content)
    
    linked_repos = []
    ignore_owners = {owner.lower(), 'settings', 'features', 'marketplace', 'trending', 'issues', 'pulls', 'sponsors', 'site', 'orgs'}
    for linked_owner, linked_repo in urls:
        linked_owner_lower = linked_owner.lower()
        linked_repo_clean = re.sub(r'[\.\#\?\)\"\'].*$', '', linked_repo).strip()
        if linked_owner_lower in ignore_owners or not linked_repo_clean:
            continue
        linked_repos.append((linked_owner, linked_repo_clean))

    linked_repos = list(dict.fromkeys(linked_repos))
    if not linked_repos:
        logger.info("No external GitHub repositories referenced in README.md")
        return {}

    logger.info(f"Found linked repositories in README: {linked_repos}")
    
    for linked_owner, linked_repo in linked_repos[:2]:
        logger.info(f"Attempting to fetch dependencies from linked repository {linked_owner}/{linked_repo}...")
        linked_files = {}
        for path in ['requirements.txt', 'setup.py', 'environment.yml']:
            content = None
            if coral.available:
                try:
                    linked_sql = f"""
                        SELECT content_text as content
                        FROM github.contents
                        WHERE owner = '{linked_owner}'
                          AND repo = '{linked_repo}'
                          AND path = '{path}'
                    """
                    linked_res = coral.sql(linked_sql)
                    if linked_res and "results" in linked_res and linked_res["results"]:
                        content = linked_res["results"][0].get("content")
                except Exception as e:
                    logger.debug(f"Coral query for linked {path} failed: {e}")
            
            if not content:
                content = _fetch_file_content_github(linked_owner, linked_repo, path)

            if content:
                linked_files[path] = content
                logger.info(f"Retrieved {path} from linked repository {linked_owner}/{linked_repo}")
        
        if linked_files:
            return linked_files

    return {}


def _ast_scan_imports(owner: str, repo: str) -> dict:
    """Fix B: Scan all .py files in the repository recursively, extract imports, map to PyPI packages, and reconstruct a requirements.txt."""
    coral = get_coral_client()
    logger.info(f"Fix B: README search yielded no dependencies. Scanning .py files for {owner}/{repo}...")

    py_paths = []
    if coral.available:
        try:
            sql = f"""
                SELECT path
                FROM github.trees
                WHERE owner = '{owner}'
                  AND repo = '{repo}'
                  AND tree_sha = 'HEAD'
                  AND recursive = 'true'
                  AND type = 'blob'
                  AND path LIKE '%.py'
            """
            res = coral.sql(sql)
            if res and "results" in res and res["results"]:
                py_paths = [item.get("path") for item in res["results"] if item.get("path")]
        except Exception as e:
            logger.debug(f"Coral tree query failed: {e}")

    if not py_paths:
        logger.info("Python files tree not found via Coral. Falling back to GitHub API...")
        import requests
        headers = {}
        token = os.environ.get("GITHUB_TOKEN")
        if token and "dummy" not in token.lower() and "your_" not in token.lower():
            headers["Authorization"] = f"token {token}"
        try:
            resp = requests.get(
                f"https://api.github.com/repos/{owner}/{repo}/git/trees/HEAD?recursive=1",
                headers=headers, timeout=15
            )
            if resp.status_code == 200:
                tree = resp.json().get("tree", [])
                py_paths = [
                    item["path"] for item in tree
                    if item["type"] == "blob" and item["path"].endswith(".py")
                ]
        except Exception as e:
            logger.debug(f"GitHub API tree query failed: {e}")

    if not py_paths:
        logger.info("No python files found in repository tree.")
        return {}

    logger.info(f"Found {len(py_paths)} Python files. Selecting files to scan...")

    local_modules = {os.path.basename(path).replace('.py', '').lower() for path in py_paths}
    for path in py_paths:
        parts = path.split('/')
        if len(parts) > 1:
            local_modules.add(parts[0].lower())

    prioritized_paths = []
    other_paths = []
    for path in py_paths:
        path_lower = path.lower()
        if '/' not in path:
            prioritized_paths.append(path)
        elif any(kw in path_lower for kw in ['main', 'train', 'run', 'eval', 'model', 'predict']):
            prioritized_paths.append(path)
        else:
            other_paths.append(path)

    paths_to_scan = (prioritized_paths + other_paths)[:10]
    logger.info(f"Scanning files: {paths_to_scan}")

    imported_modules = set()
    for path in paths_to_scan:
        content = None
        if coral.available:
            try:
                content_sql = f"""
                    SELECT content_text as content
                    FROM github.contents
                    WHERE owner = '{owner}'
                      AND repo = '{repo}'
                      AND path = '{path}'
                """
                content_res = coral.sql(content_sql)
                if content_res and "results" in content_res and content_res["results"]:
                    content = content_res["results"][0].get("content", "")
            except Exception as e:
                logger.debug(f"Coral query for {path} failed: {e}")
        
        if not content:
            content = _fetch_file_content_github(owner, repo, path)

        if not content:
            continue

        import_matches = re.findall(r'^\s*import\s+([a-zA-Z0-9_\., \t]+)', content, re.MULTILINE)
        for match in import_matches:
            for parts in match.split(','):
                pkg = parts.strip().split('.')[0].strip()
                pkg_parts = pkg.split()
                if pkg_parts:
                    pkg = pkg_parts[0]
                if pkg:
                    imported_modules.add(pkg.lower())

        from_matches = re.findall(r'^\s*from\s+([a-zA-Z0-9_\.]+)\s+import', content, re.MULTILINE)
        for match in from_matches:
            pkg = match.strip().split('.')[0].strip()
            if pkg:
                imported_modules.add(pkg.lower())

    std_lib = {
        'os', 'sys', 'json', 'math', 'time', 're', 'collections', 'itertools', 'typing', 'logging',
        'subprocess', 'tempfile', 'argparse', 'shutil', 'urllib', 'hashlib', 'datetime', 'random',
        'pickle', 'copy', 'io', 'functools', 'abc', 'pathlib', 'warnings', 'threading', 'queue',
        'multiprocessing', 'socket', 'struct', 'select', 'csv', 'ctypes', 'inspect', 'pdb', 'traceback',
        'uuid', 'glob', 'fnmatch', 'weakref', 'contextlib'
    }

    third_party = imported_modules - std_lib - local_modules
    if not third_party:
        logger.info("No third-party packages identified in scanned files.")
        return {}

    pypi_mapping = {
        'pil': 'Pillow',
        'cv2': 'opencv-python',
        'sklearn': 'scikit-learn',
        'yaml': 'pyyaml',
        'tensorboard': 'tensorboard',
        'timm': 'timm',
        'torch': 'torch',
        'torchvision': 'torchvision',
        'numpy': 'numpy',
        'pandas': 'pandas',
        'spacy': 'spacy',
        'transformers': 'transformers',
        'tqdm': 'tqdm',
        'scipy': 'scipy',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'gym': 'gym',
        'h5py': 'h5py',
        'nltk': 'nltk',
        'requests': 'requests',
        'jinja2': 'jinja2',
        'click': 'click',
        'plotly': 'plotly',
    }

    reconstructed_packages = []
    for mod in third_party:
        pypi_name = pypi_mapping.get(mod, mod)
        if mod == 'timm':
            reconstructed_packages.append('timm==0.3.2')
        else:
            reconstructed_packages.append(pypi_name)

    logger.info(f"Reconstructed packages: {reconstructed_packages}")
    
    synthetic_content = "\n".join(reconstructed_packages) + "\n"
    return {'requirements.txt': synthetic_content}


def _fetch_readme(owner: str, repo: str) -> str:
    """Fetch README.md content via Coral or GitHub API for build instructions."""
    coral = get_coral_client()
    if coral.available:
        try:
            res = coral.sql(f"""
                SELECT content_text as content
                FROM github.contents
                WHERE owner = '{owner}' AND repo = '{repo}' AND path = 'README.md'
            """)
            if res and "results" in res and res["results"]:
                content = res["results"][0].get("content") or res["results"][0].get("content_text")
                if content:
                    return content
        except Exception as e:
            logger.debug(f"Coral query for README.md failed: {e}")
    return _fetch_file_content_github(owner, repo, "README.md") or ""


def _empty_result(reason: str) -> dict:
    return {
        "requirements_found": False,
        "dep_files": {},
        "conflicts": [],
        "warnings": [reason],
        "clean": True,
        "analysis_method": None,
        "entrypoint": None,
        "readme_content": "",
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = check_compatibility("https://github.com/tensorflow/tensor2tensor")
    print(json.dumps(result, indent=2))
