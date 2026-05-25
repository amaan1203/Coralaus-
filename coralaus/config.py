import os
from pathlib import Path

# Repository root (one level up from this file's directory)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Output directory – always './output' relative to the repository root
OUTPUT_DIR = PROJECT_ROOT / "output"
