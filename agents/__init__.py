# PaperDock Agents
# Each module corresponds to a component in the pipeline:
#   ingest.py          — C1: PDF → JSON (GROBID/Docling)
#   pwc_search.py      — C2: PapersWithCode search
#   repo_health.py     — C3: GitHub health scoring via Coral
#   compat_check.py    — C4: Dependency compatibility check
#   conflict_resolver.py — C5: Conflict resolution + Dockerfile gen
#   no_impl_generator.py — C6: Generate implementation from scratch
#   output_builder.py  — C7: Final output assembly
