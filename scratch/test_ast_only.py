import logging
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)

from agents.compat_check import _ast_scan_imports

res = _ast_scan_imports("facebookresearch", "mae")
print("RESULT:")
print(json.dumps(res, indent=2))
