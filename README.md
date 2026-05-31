# Coralaus

> 🏴‍☠️ Pirates of the Coral-bean Hackathon · Track 2

**Coralaus** is an end-to-end research paper reproducibility engine. Drop in any ML/CS paper as a PDF and it automatically finds, validates, fixes, or generates a working code implementation — all powered by a **Dual MCP + Coral SQL** architecture.

1. Parses the full paper into a structured JSON representation offline using **PyPDF2**.
2. Searches for official code implementations across **Semantic Scholar, Hugging Face Papers, and GitHub** using 7 parallel strategies.
3. Scores the discovered repository's maintenance health (0–100) using **Coral SQL queries against the GitHub connector**.
4. Fetches dependency files from the repo and identifies Python package conflicts.
5. Resolves conflicts and generates a working `Dockerfile` using **Groq (Llama 3.3 70B)**.
6. When no implementation exists, searches related repos via **Coral cross-repo SQL** and generates a reference implementation using **Gemini 2.0 Flash**.

---

## Architecture

Coralaus uses a **Dual MCP + Coral SQL** architecture. [Coral](https://github.com/withcoral/coral) acts as the unified query layer over GitHub — covering repo search, code search, commit history, issue tracking, and file contents — all via plain SQL with a REST API fallback when Coral is unavailable.

```
PDF Input
   │
   ▼
[Agent 1] PyPDF2 Ingest ──────────────► current_paper.json
                                                │
                                                ▼
[Agent 2] Semantic Scholar + HF Papers + GitHub Search ──► repo URL
                    │
                    │   Coral: github.search_repositories
                    │   Coral: github.search_code (MD files)
                    │
                    ▼
[Agent 3] Repo Health Check ──────────────────────────────► score (0–100)
              │
              │   Coral: github.commits
              │   Coral: github.issues
              │   Coral: github.repos_get
              │
              ▼
[Agent 4] Dependency Fetch + pip dry-run ─────────────────► conflict list
              │
              ▼
[Agent 5] Groq ──────────────────────────────────────────► Dockerfile
              │
              │  (if no repo found in Agent 2)
              ▼
[Agent 6] Coral cross-repo search + Gemini ──────────────► implementation.py
              │
              │   Coral: github.search_repositories
              │   Coral: github.search_code
              │
              ▼
[Agent 7] Output Builder ────────────────────────────────► result.json
```

---

##  Coral Usage — Where & How

Coral is used in **two agents** across the pipeline, always with a full GitHub REST API fallback if Coral is unavailable or rate-limited.

---

### 1. GitHub Repository & Code Search (`agents/pwc_search.py`)

Coral handles two of the seven search strategies in the implementation search agent:

**Strategy 5 & 6 — `github.search_repositories`**

When Semantic Scholar and Hugging Face searches don't yield enough candidates, Coral queries GitHub's repository index directly, ordered by stars:

```python
coral.sql(f"""
    SELECT full_name, html_url, stargazers_count, description
    FROM github.search_repositories(q => '{query}')
    ORDER BY stargazers_count DESC
    LIMIT 10
""")
```

Queries are expanded from the paper title using acronym extraction, prefix splitting, and keyword extraction — so a paper titled *"BERT: Pre-training of Deep Bidirectional Transformers"* would generate queries like `BERT`, `Pre-training Deep Bidirectional`, etc.

**Strategy 7 — `github.search_code`**

Coral also runs a code search scoped to Markdown files, looking for the arXiv ID or title keywords inside READMEs. This catches repos that reference the paper without naming it in the repo title or description:

```python
coral.sql(f"""
    SELECT repository_full_name, path
    FROM github.search_code(q => '"{arxiv_id}" extension:md')
    LIMIT 8
""")
```

When a match is found inside a subdirectory (e.g. `papers/2023/attention/README.md`), Coralaus adds both the root repo URL and the specific subdirectory URL as candidates. Real star counts are fetched from the GitHub API for each discovered repo to enable accurate ranking.

> **Fallback:** If `coral.available` is `False`, both strategies fall back to `github.com/search/repositories` and `github.com/search/code` REST endpoints directly.

---

### 2. Repository Health Scoring (`agents/repo_health.py`)

Coral's built-in GitHub connector powers the entire health check with **3 SQL queries** — no LLM involved. Each query targets a different GitHub table:

**Query 1 — Last commit date** (`github.commits`)
```sql
SELECT commit__committer__date AS committed_date
FROM github.commits
WHERE owner = '{owner}' AND repo = '{repo}'
ORDER BY commit__committer__date DESC
LIMIT 1
```

**Query 2 — Open issue count** (`github.issues`)
```sql
SELECT COUNT(*) AS open_issues
FROM github.issues
WHERE owner = '{owner}' AND repo = '{repo}'
  AND state = 'open'
```

**Query 3 — Repo metadata** (`github.repos_get`)
```sql
SELECT stargazers_count, forks_count, archived, created_at
FROM github.repos_get
WHERE owner = '{owner}' AND repo = '{repo}'
```

These three signals feed into a deterministic scoring function that computes a **0–100 health score**:

| Signal | Penalty / Bonus |
|---|---|
| Last commit > 2 years ago | −50 |
| Last commit > 1 year ago | −25 |
| Last commit > 6 months ago | −10 |
| Issue close rate < 30% | −20 |
| Issue close rate < 50% | −10 |
| Stars/year < 10 (low traction) | −10 |
| Forks > 50 | +5 |
| Contributors ≥ 5 | +5 |
| CI/CD workflows present | +5 |
| Archived | → 0 immediately |

**Score ≥ 80** 🟢 Healthy · **≥ 60** 🟡 Fair · **≥ 30** 🟠 Stale · **< 30** 🔴 Dead

> **Fallback:** If Coral returns no data (unavailable or auth failure), the agent falls back to 4 direct GitHub REST API calls: `GET /repos/{owner}/{repo}`, search/issues (closed count), /contributors, and /.github/workflows (CI check).

---

### Coral Source Registration

```bash
# Install Coral
brew install withcoral/tap/coral        # macOS
# or download binary for Linux/Windows from GitHub releases

# Add GitHub built-in connector
coral source add github --token $GITHUB_TOKEN

# Start the MCP server
coral mcp start
```

> **Note:** `sources/paperswithcode.yaml` is present in the repo as an early design artifact from the hackathon planning phase. It is not used at runtime — the Semantic Scholar integration uses a direct HTTP client (`agents/pwc_mcp_client.py`) rather than a Coral custom source.

---

## Pipeline — Step by Step

### Step 1 — PDF Ingestion (`agents/ingest.py`)

Parses the uploaded PDF into structured JSON using **PyPDF2** — entirely offline, no LLM or network call.

**Input:** Raw PDF  
**Output:** `./output/current_paper.json`

```json
{
  "title": "Attention Is All You Need",
  "abstract": "The dominant sequence transduction models...",
  "authors": [{ "first": "Ashish", "last": "Vaswani" }],
  "year": 2017,
  "arxiv_id": "1706.03762",
  "sections": [{ "heading": "Introduction", "text": "..." }],
  "full_text": "...",
  "parsed_by": "pypdf2"
}
```

---

### Step 2 — Implementation Search (`agents/pwc_search.py`)

>  **Coral used:** `github.search_repositories`, `github.search_code`

Runs **7 search strategies in order**, accumulating candidates until it has at least 10 repo URLs, then ranks them by a scoring function that boosts official repos, known ML orgs (Google, HuggingFace, Meta, etc.), and high star counts, while penalising homework/course repos.

| Strategy | Source | Method |
|---|---|---|
| 0 | Hugging Face Papers API | Direct lookup by arXiv ID |
| 1 | Semantic Scholar | arXiv ID lookup → linked repos |
| 2 | Semantic Scholar | Title search → linked repos |
| 3 | Semantic Scholar | Keyword extraction → linked repos |
| 4 | Paper body (PDF text) | Regex extraction of GitHub URLs |
| 5 |  **Coral** / GitHub REST | `search_repositories` by arXiv ID / title |
| 6 |  **Coral** / GitHub REST | `search_repositories` by expanded title queries |
| 7 |  **Coral** / GitHub REST | `search_code` in `.md` files by arXiv ID / title |

The final candidate list is ranked and the best URL is returned. The full `candidates` list is also surfaced in the output JSON for transparency.

---

### Step 3 — Repository Health Check (`agents/repo_health.py`)

>  **Coral used:** `github.commits`, `github.issues`, `github.repos_get`

Runs 3 Coral SQL queries against the GitHub connector to pull live signals (last commit date, open issue count, star count, fork count, archived status, repo age). These feed into a pure Python scoring function — no LLM involved.

The v2 scoring model uses **issue resolution ratio** (closed/total) instead of a raw open count, and **star velocity** (stars/year) instead of an absolute cutoff — so new official repos aren't unfairly penalised.

---

### Step 4 — Dependency Compatibility Check (`agents/compat_check.py`)

Fetches `requirements.txt`, `environment.yml`, `setup.py`, and `pyproject.toml` from the discovered repo (via GitHub REST API) and runs `pip install --dry-run` locally to detect conflicts. Groq is only invoked if the automated check finds actual problems — most repos with no conflicts skip the LLM entirely.

---

### Step 5 — Conflict Resolution & Dockerfile Generation (`agents/conflict_resolver.py`)

> **LLM used:** Groq — Llama 3.3 70B

Takes the original requirements and the conflict list from Step 4, and generates a working `Dockerfile` for Python 3.11 with all conflicts resolved. Each pinned fix includes an inline comment explaining the change.

**Output:** `./output/Dockerfile`

---

### Step 6 — No-Implementation Generator (`agents/no_impl_generator.py`)

>  **Coral used:** `github.search_repositories`, `github.search_code`  
> **LLMs used:** Groq (keyterm extraction) + Gemini 2.0 Flash (implementation generation)

Triggered when Step 2 finds no usable implementation. The flow:

1. **Groq** extracts 5–7 technical keywords from the paper abstract.
2. **Coral** (same strategies as Step 2, Strategies 5–7) searches GitHub for the top related Python repos.
3. Requirements and core `.py` files are fetched from the top repos.
4. **Gemini 2.0 Flash** receives the full paper body + aggregated repo context and generates `implementation.py` + `Dockerfile`.

**Output:** `./output/implementation.py`, `./output/Dockerfile`

---

### Step 7 — Output Assembly (`agents/output_builder.py`)

Assembles all results into a final JSON artifact and surfaces it in the Streamlit UI with a health badge, collapsible conflict diff view, download buttons, and a live count of Coral queries executed.

**Output:** `./output/result.json`

```json
{
  "paper": { "title": "...", "arxiv_id": "...", "year": 2017 },
  "implementation_found": true,
  "repo_url": "https://github.com/...",
  "repo_health_score": 82,
  "health_signals": {
    "last_commit_days_ago": 45,
    "open_issues": 12,
    "closed_issues": 88,
    "stars": 3400,
    "forks": 210,
    "archived": false
  },
  "conflicts_found": ["torch vs torchvision version mismatch"],
  "conflicts_resolved": true,
  "dockerfile": "FROM python:3.11-slim\n...",
  "implementation_script": null,
  "generated_from_scratch": false,
  "coral_queries_used": 3
}
```

---

## LLM Strategy

The rule: **use SQL before LLM, use Groq before Gemini.**

| Step | Tool | Why |
|---|---|---|
| PDF parsing | PyPDF2 — **no LLM** | Deterministic, offline, fast |
| Implementation search (strategies 0–4) | Direct HTTP — **no LLM** | Pure data lookup |
| Implementation search (strategies 5–7) |  **Coral SQL** — **no LLM** | Structured GitHub query |
| Repo health scoring |  **Coral SQL** — **no LLM** | Deterministic arithmetic |
| Conflict detection | pip dry-run — **no LLM** | Automated; Groq only on failure |
| Conflict analysis + Dockerfile | **Groq (Llama 3.3 70B)** | Short input, fast structured JSON |
| Implementation generation | **Gemini 2.0 Flash** | Full paper body = long context (1M tokens) |

---

## Tech Stack

| Layer | Tool |
|---|---|
| PDF parsing | PyPDF2 (offline) |
| Paper search | Semantic Scholar API + Hugging Face Papers API |
| GitHub data |  Coral built-in GitHub connector + REST API fallback |
| LLM — fast structured outputs | Groq — Llama 3.3 70B |
| LLM — long context | Gemini 2.0 Flash |
| UI | Streamlit |
| Containerization | Docker |

---

## File Structure

```
coralaus/
├── agents/
│   ├── ingest.py             # Step 1: PDF → JSON (PyPDF2)
│   ├── pwc_mcp_client.py     # Semantic Scholar + HF Papers HTTP client
│   ├── coral_utils.py        # Coral client wrapper (with availability check)
│   ├── pwc_search.py         # Step 2: Multi-strategy impl search (incl. Coral)
│   ├── repo_health.py        # Step 3: Coral GitHub → health score
│   ├── compat_check.py       # Step 4: dep file fetch + pip dry-run
│   ├── conflict_resolver.py  # Step 5: Groq → Dockerfile
│   ├── no_impl_generator.py  # Step 6: Coral search + Gemini → implementation
│   └── output_builder.py     # Step 7: Final JSON assembly
├── ui/
│   └── app.py                # Streamlit UI
├── scripts/
│   ├── test_full_pipeline.py
│   └── setup_coral.sh        # Coral install + GitHub source registration
├── output/                   # All generated artifacts land here
├── sample_papers/            # Test PDFs
├── scratch/                  # Dev scratch space
├── requirements.txt
├── .env.example
└── CORAL-IMPLEMENTATION.md   # Detailed internal implementation guide
```

---

## Setup & Usage

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Copy `.env.example` to `.env` and fill in:

```env
GEMINI_API_KEY=...
GITHUB_TOKEN=...                   # Personal Access Token — used by Coral and REST fallback
GROQ_API_KEY=...
SEMANTIC_SCHOLAR_API_KEY=...       # Optional — raises rate limit from 1 to 10 req/s
```

### 3. Set up Coral

```bash
bash scripts/setup_coral.sh
# Installs Coral CLI, registers the GitHub source, starts MCP server
```

Or manually:
```bash
brew install withcoral/tap/coral
coral source add github --token $GITHUB_TOKEN
coral mcp start
```

### 4. Run the UI

```bash
streamlit run ui/app.py
```

### 5. Process a paper from the CLI

```bash
# Ingest only
python -m agents.ingest path/to/paper.pdf
# Output → ./output/current_paper.json

# Full pipeline
python scripts/test_full_pipeline.py path/to/paper.pdf
```

All generated files are written to `./output/`.

---

## Build & Run with Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
CMD ["streamlit", "run", "ui/app.py", "--server.headless", "true", "--server.port", "8501"]
```

```bash
docker build -t coralaus .
docker run -p 8501:8501 --env-file .env coralaus
```
