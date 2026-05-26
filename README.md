# Coralaus

> Pirates of the Coral-bean Hackathon · Track 2

**Coralaus** takes a research paper (PDF) as input and:
1. Parses the full paper into a structured JSON representation offline using PyPDF2.
2. Checks for official code implementations on GitHub using PapersWithCode MCP / direct search.
3. Automatically scores the repository's maintenance health (0‑100) using Coral SQL queries.
4. Identifies and resolves Python package dependency conflicts, outputting a valid `Dockerfile` for execution.
5. Generates reference code implementations from scratch when no official codebase exists.

---

## Architecture

Coralaus utilizes a **Dual MCP** and **Coral SQL** architecture:
- **PapersWithCode MCP Server:** Integrated via subprocess StdIO transport.
- **Coral CLI Engine:** Powers GitHub schema metadata queries (commits recency, repository stats, issues workload, contents extraction).
- **Offline Fallbacks:** Gracefully falls back to direct unauthenticated GitHub REST API calls when rate‑limited or using dummy keys.
- **Lightweight Parser:** Uses local PyPDF2 offline extraction (Grobid removed).

##  File Structure

```
coralaus/            # Python package
├── agents/          # Core pipeline components
│   ├── ingest.py    # PDF → JSON parsing (PyPDF2 only)
│   ├── pwc_search.py
│   ├── repo_health.py
│   ├── compat_check.py
│   ├── conflict_resolver.py
│   ├── no_impl_generator.py
│   └── output_builder.py
├── ui/              # Streamlit UI
├── scripts/         # Test scripts & setup helper
│   └── setup_coral.sh
├── output/          # Generated artifacts (always ./output)
├── requirements.txt
├── .env.example
├── README.md
└── coralaus/        # Package init files
```

##  Setup & Usage

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **setup the gemini key, github PAT and groq api key**

3. **Run the UI**
   ```bash
   streamlit run ui/app.py 
   ```
4. **Process a paper**
   ```bash
   python -m agents.ingest path/to/paper.pdf
   # Output written to ./output/current_paper.json and includes a `full_text` field
   ```
5. **Run the full pipeline**
   ```bash
   python scripts/test_full_pipeline.py [path_to_pdf]
   ```

All generated files are stored under the `./output` directory relative to the repository root.
---

