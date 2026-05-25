# PaperDock — Implementation Guide
> Pirates of the Coral-bean Hackathon · Track 1: Enterprise Agent
> For: Antigravity Team

---

## Project Summary

**PaperDock** takes a research paper (PDF) as input and:
1. Parses the full paper into a rich JSON object (no LLM needed for this)
2. Checks if an official implementation exists on GitHub via PapersWithCode
3. Validates whether that implementation is still maintained and dependency-conflict-free
4. If healthy → returns the repo with a verified environment file
5. If broken → resolves conflicts and regenerates a working Dockerfile/conda.yaml
6. If no implementation exists → generates a reference implementation + environment from scratch

**Core Coral usage:** Coral MCP server is the backbone for all GitHub data queries (files, repo health, cross-repo joins). PapersWithCode is added as a custom Coral source.

---

## Tech Stack

| Layer | Tool |
|---|---|
| LLM — long context (PDF body, code gen) | **Gemini 2.0 Flash** (free, 1M token context, 15 RPM) |
| LLM — fast structured outputs (keyterms, conflicts) | **Groq — Llama 3.3 70B** (free, 14,400 req/day) |
| PDF parsing — primary | **GROBID** (free cloud API or Docker) |
| PDF parsing — fallback | **Docling** (IBM, MIT license, CPU-only, local) |
| Data querying | Coral CLI + Coral MCP server |
| Custom source | PapersWithCode API (custom Coral source spec) |
| GitHub data | Coral built-in GitHub connector |
| Backend | Python 3.11+ |
| Frontend/UI | Streamlit |

### Why Gemini + Groq (not Claude)?
- **Gemini 2.0 Flash** free tier: 15 RPM, 1,000,000 tokens/day — enough for full paper bodies
- **Groq** free tier: 14,400 req/day, ~300 tokens/sec — ideal for fast structured JSON outputs
- Rule: use Gemini when context is large (full paper), use Groq when output needs to be fast and structured

---

## Component Breakdown

---

### Component 1 — PDF Ingestion & Full Paper JSON

**What it does:** Parses the uploaded PDF into a complete structured JSON using GROBID — **no LLM call needed here**.

**Input:** Raw PDF file
**Output:** `current_paper.json` (rich, full-paper object)

#### Why GROBID?
GROBID (GeneRation Of BIbliographic Data) is purpose-built for scientific PDFs. It powers Semantic Scholar's entire S2ORC corpus. It extracts:
- Title, abstract, authors, year, DOI, venue
- Full section-by-section body text
- References with structured metadata
- Figures and table captions
- Keywords (if present in paper)

#### Option A — GROBID Cloud (easiest, no setup)
```python
import requests, json

def parse_paper_grobid(pdf_path: str) -> dict:
    # Free public GROBID server (rate limited but fine for hackathon)
    GROBID_URL = "https://kermitt2-grobid.hf.space/api/processFulltextDocument"
    
    with open(pdf_path, "rb") as f:
        response = requests.post(
            GROBID_URL,
            files={"input": f},
            data={"consolidateHeader": 1, "consolidateCitations": 0}
        )
    
    # GROBID returns TEI XML — convert to JSON
    from grobid_client.grobid_client import GrobidClient
    # OR use the xml2json helper below
    return tei_xml_to_json(response.text)
```

#### Option B — GROBID via Docker (local, no rate limits)
```bash
docker run --rm -p 8070:8070 lfoppiano/grobid:0.8.1
```
Then hit `http://localhost:8070/api/processFulltextDocument`

#### Option C — Docling fallback (pure Python, no server)
```python
from docling.document_converter import DocumentConverter

def parse_paper_docling(pdf_path: str) -> dict:
    converter = DocumentConverter()
    result = converter.convert(pdf_path)
    # Docling exports a native DoclingDocument — convert to dict
    return result.document.export_to_dict()
```
Use Docling when GROBID cloud is rate-limited or down.

#### Output JSON schema — `current_paper.json`

```json
{
  "title": "Attention Is All You Need",
  "abstract": "The dominant sequence transduction models...",
  "authors": [
    { "first": "Ashish", "last": "Vaswani", "affiliation": "Google Brain" }
  ],
  "year": 2017,
  "venue": "NeurIPS",
  "doi": "10.48550/arXiv.1706.03762",
  "arxiv_id": "1706.03762",
  "keywords": [],
  "sections": [
    { "heading": "Introduction", "text": "..." },
    { "heading": "Model Architecture", "text": "..." },
    { "heading": "Attention Mechanism", "text": "..." }
  ],
  "references": [
    { "title": "Neural Machine Translation by Jointly...", "authors": [...], "year": 2015 }
  ],
  "figures": [
    { "caption": "Figure 1: The Transformer model architecture." }
  ],
  "parsed_by": "grobid",
  "parsed_at": "2025-05-25T10:00:00Z"
}
```

> **This JSON is the single source of truth for all downstream steps.**
> Register it as a Coral local source immediately after parsing:
> ```bash
> coral source add local_file --path ./current_paper.json --name paper_meta
> ```

---

### Component 2 — PapersWithCode Custom Coral Source

**What it does:** Connects Coral to the PapersWithCode API so we can SQL-query paper implementations.

**Files to create:**
- `sources/paperswithcode.yaml` — Coral source spec

```yaml
name: paperswithcode
base_url: https://paperswithcode.com/api/v1
auth:
  type: none   # PapersWithCode API is public, no auth needed
endpoints:
  - name: papers
    path: /papers/
    method: GET
    params:
      - name: q
        type: string
        description: Paper title or keyword search
    response_map:
      results:
        - id: id
          title: title
          arxiv_id: arxiv_id
          url_pdf: url_pdf
  - name: paper_implementations
    path: /papers/{paper_id}/repositories/
    method: GET
    response_map:
      results:
        - url: url
          stars: stars
          framework: framework
          is_official: is_official
```

**Register:**
```bash
coral source add paperswithcode --spec ./sources/paperswithcode.yaml
```

**Coral query — search by title (from `current_paper.json`):**
```sql
SELECT r.url, r.stars, r.framework, r.is_official
FROM paperswithcode.papers p
JOIN paperswithcode.paper_implementations r ON p.id = r.paper_id
WHERE p.title LIKE '%{paper_title}%'
  AND r.is_official = true
ORDER BY r.stars DESC
LIMIT 1;
```

**Fallback — search by arxiv_id if title search misses:**
```sql
SELECT r.url, r.stars, r.framework, r.is_official
FROM paperswithcode.papers p
JOIN paperswithcode.paper_implementations r ON p.id = r.paper_id
WHERE p.arxiv_id = '{arxiv_id}'
ORDER BY r.stars DESC
LIMIT 3;
```
The `arxiv_id` comes directly from `current_paper.json` — no LLM needed.

---

### Component 3 — Repo Health Check (Coral GitHub Query)

**What it does:** Given a GitHub repo URL, fetches health signals using Coral's built-in GitHub connector. Pure SQL — no LLM.

**Input:** GitHub repo URL
**Output:** `health_score` (0–100) + signals JSON

```sql
-- Last commit date
SELECT committed_date
FROM github.commits
WHERE owner = '{owner}' AND repo = '{repo}'
ORDER BY committed_date DESC
LIMIT 1;

-- Open issues count
SELECT COUNT(*) as open_issues
FROM github.issues
WHERE owner = '{owner}' AND repo = '{repo}'
  AND state = 'open';

-- Repo metadata
SELECT stargazers_count, forks_count, archived
FROM github.repositories
WHERE owner = '{owner}' AND repo = '{repo}';
```

**Health scoring (Python, no LLM):**
```python
from datetime import date

def compute_health_score(last_commit_date, open_issues, stars, archived) -> int:
    if archived:
        return 0
    score = 100
    days_since_commit = (date.today() - last_commit_date).days
    if days_since_commit > 730:   score -= 50
    elif days_since_commit > 365: score -= 25
    elif days_since_commit > 180: score -= 10
    if open_issues > 100: score -= 20
    elif open_issues > 50: score -= 10
    if stars < 10:        score -= 10
    return max(0, score)
```

- `score >= 60` → proceed to compatibility check
- `score < 60` → warn user, still proceed

---

### Component 4 — Dependency Compatibility Check

**What it does:** Fetches `requirements.txt` / `environment.yml` from the repo. Uses **pip-tools** or **pipdeptree** locally first (no LLM). Falls back to Groq only if automated check can't resolve.

**Coral query to fetch env files:**
```sql
SELECT content, path
FROM github.file_contents
WHERE owner = '{owner}'
  AND repo = '{repo}'
  AND path IN ('requirements.txt', 'environment.yml', 'setup.py', 'pyproject.toml');
```

**Automated conflict detection (no LLM needed for basic cases):**
```bash
# Write fetched requirements to temp file, then dry-run pip
pip install --dry-run -r /tmp/requirements.txt 2>&1 | grep -i "conflict\|error\|incompatible"
```

**If automated check finds conflicts → send to Groq (fast, structured):**
```python
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

def analyze_conflicts(requirements_content: str, pip_error_output: str) -> dict:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "system",
            "content": "You are a Python dependency expert. Return only valid JSON, no markdown."
        }, {
            "role": "user",
            "content": f"""
requirements.txt:
{requirements_content}

pip dry-run errors:
{pip_error_output}

Return JSON: {{"conflicts": [...], "warnings": [...], "clean": bool}}
"""
        }],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)
```

---

### Component 5 — Conflict Resolution & Dockerfile Generation

**What it does:** Resolves dependency conflicts and generates a working Dockerfile. Uses **Groq** for speed.

**Input:** Original requirements + conflict list from Component 4
**Output:** `Dockerfile` and/or `environment.yaml`

```python
def generate_dockerfile(requirements: str, conflicts: list) -> str:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "system",
            "content": "You are a DevOps expert. Return only raw Dockerfile content, no markdown fences."
        }, {
            "role": "user",
            "content": f"""
requirements.txt:
{requirements}

Conflicts to fix:
{json.dumps(conflicts, indent=2)}

Generate a working Dockerfile for Python 3.11 that resolves all conflicts.
Pin all versions. Add a comment above each conflict fix explaining the change.
"""
        }]
    )
    return response.choices[0].message.content
```

---

### Component 6 — No-Implementation Path: Generate from Scratch

**What it does:** When PapersWithCode returns nothing, searches GitHub for related repos and generates a reference implementation. Uses **Gemini 2.0 Flash** here (needs full paper body as context).

**6a. Keyterm extraction — Groq (fast, structured):**
```python
import google.generativeai as genai   # or use Groq here — it's short context

def extract_keyterms(abstract: str) -> list[str]:
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{
            "role": "user",
            "content": f"Extract 5-7 technical keywords from this abstract for a GitHub code search. Return only a JSON array of strings.\n\nAbstract:\n{abstract}"
        }],
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)["keywords"]
```
> `abstract` comes directly from `current_paper.json["abstract"]` — already parsed.

**6b. Coral GitHub search across top 3 repos:**
```sql
SELECT r.full_name, r.url, r.stars, r.description
FROM github.search_repositories
WHERE query = '{kw1} {kw2} {kw3} language:python'
ORDER BY r.stars DESC
LIMIT 3;
```

**6c. Cross-repo JOIN — the Coral money shot:**
```sql
-- Fetch requirements + core logic files from all 3 repos in ONE query
SELECT r.full_name, f.content, f.path
FROM github.file_contents f
JOIN github.repositories r ON r.full_name = f.repo_full_name
WHERE r.full_name IN ('{repo1}', '{repo2}', '{repo3}')
  AND (f.path LIKE '%requirements%'
    OR f.path LIKE '%.py'
    OR f.path LIKE '%environment%')
ORDER BY r.full_name, f.path;
```
Three repos, one SQL JOIN — this is what wins judges.

**6d. Generate implementation — Gemini 2.0 Flash (needs full paper body):**
```python
import google.generativeai as genai

genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

def generate_implementation(paper_json: dict, logic_files: list, common_deps: list) -> dict:
    paper_sections = "\n\n".join(
        f"## {s['heading']}\n{s['text']}"
        for s in paper_json["sections"]
    )
    
    prompt = f"""
You are an expert ML engineer.

Paper: {paper_json['title']}
Abstract: {paper_json['abstract']}

Full paper content:
{paper_sections}

Related repository logic files:
{chr(10).join(logic_files)}

Common dependencies across related repos:
{common_deps}

Generate:
1. A minimal, well-commented Python implementation (implementation.py) demonstrating the core algorithm.
2. A Dockerfile for Python 3.11 that runs it.

Return as JSON: {{"implementation_py": "...", "dockerfile": "..."}}
"""
    response = model.generate_content(prompt)
    return json.loads(response.text)
```

---

### Component 7 — Final Output

**What it does:** Assembles everything into a structured result for the UI.

**Full output JSON:**
```json
{
  "paper": {
    "title": "Attention Is All You Need",
    "arxiv_id": "1706.03762",
    "authors": ["Vaswani et al."],
    "year": 2017
  },
  "implementation_found": true,
  "repo_url": "https://github.com/user/repo",
  "repo_health_score": 82,
  "health_signals": {
    "last_commit_days_ago": 45,
    "open_issues": 12,
    "stars": 3400,
    "archived": false
  },
  "conflicts_found": ["torch vs torchvision version mismatch"],
  "conflicts_resolved": true,
  "dockerfile": "FROM python:3.11-slim\n...",
  "implementation_script": null,
  "generated_from_scratch": false,
  "coral_queries_used": 4,
  "llm_calls": {
    "gemini_flash": 1,
    "groq_llama": 2
  }
}
```

**Streamlit UI:**
- Health score badge (🟢 / 🟡 / 🔴)
- Collapsible conflict diff view
- Download buttons: `Dockerfile`, `implementation.py`, `current_paper.json`
- "Coral queries used" counter (great for demo)

---

## LLM Usage Summary

| Step | Tool | Why |
|---|---|---|
| Keyterm extraction | Groq (Llama 3.3 70B) | Short input, needs fast JSON |
| Conflict analysis | Groq (Llama 3.3 70B) | Short input, needs fast JSON |
| Dockerfile generation | Groq (Llama 3.3 70B) | Code gen, short context |
| Full implementation generation | Gemini 2.0 Flash | Needs full paper body (long context) |
| PDF parsing | **GROBID / Docling — NO LLM** | Deterministic, structured, faster |

---

## Coral Source Registration Sequence

```bash
# 1. Install Coral
brew install withcoral/tap/coral

# 2. Add GitHub connector (built-in)
coral source add github --token $GITHUB_TOKEN

# 3. Add PapersWithCode custom source
coral source add paperswithcode --spec ./sources/paperswithcode.yaml

# 4. After paper parsed, add full JSON as Coral local source
coral source add local_file --path ./current_paper.json --name paper_meta

# 5. Start MCP server for agent tool calls
coral mcp start
```

---

## File Structure

```
paperdock/
├── sources/
│   └── paperswithcode.yaml        # Custom Coral source spec
├── agents/
│   ├── ingest.py                  # Component 1: PDF → JSON (GROBID/Docling)
│   ├── pwc_search.py              # Component 2: PapersWithCode Coral query
│   ├── repo_health.py             # Component 3: Health score (pure SQL)
│   ├── compat_check.py            # Component 4: pip dry-run + Groq fallback
│   ├── conflict_resolver.py       # Component 5: Groq → Dockerfile
│   ├── no_impl_generator.py       # Component 6: Gemini → implementation
│   └── output_builder.py          # Component 7: Final JSON assembly
├── ui/
│   └── app.py                     # Streamlit UI
├── current_paper.json             # Generated at runtime by GROBID
├── requirements.txt
└── README.md
```

---

## Demo Flow (for judges)

1. Upload a real paper PDF — show GROBID parsing it instantly into rich JSON (no LLM!)
2. Show the Coral SQL query hitting PapersWithCode live
3. Show the GitHub health score query (3 SQL calls → one numeric score)
4. Show conflict detection firing on the requirements.txt (pip dry-run first, Groq only if needed)
5. Show the Dockerfile being generated
6. For the second demo: use a paper with NO implementation → show the 3-repo cross-JOIN → show Gemini generating the implementation

**Best paper to demo with:** Any 2020–2022 ML paper where the GitHub repo hasn't been updated in 1+ years and has dependency rot. Good candidates: older BERT variants, early diffusion model repos.

---

## Judging Criteria Alignment

| Criteria | How PaperDock scores |
|---|---|
| Potential Impact | Real pain point for every ML researcher trying to reproduce results |
| Creativity | Deterministic parsing + SQL health scoring + LLM only where needed |
| Best Use of Coral | Custom source spec + GitHub connector + cross-repo JOIN in one query |
| Technical Implementation | GROBID → structured JSON → SQL → targeted LLM calls (minimal, purposeful) |
| Aesthetics & UX | Health badge, conflict diff, downloadable artifacts, Coral query counter |
