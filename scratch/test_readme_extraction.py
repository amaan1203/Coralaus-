import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(level=logging.INFO)

from agents.compat_check import _extract_readme_dependencies, check_compatibility

def test_readme_deps_extraction():
    print("\n--- Test 1: README Dependency Extraction via Groq ---")
    mock_readme = """
    # My Awesome ML Project

    This project implements some super cool neural nets.

    ## Requirements
    - Python 3.8
    - PyTorch >= 1.7.0
    - torchvision == 0.8.1
    - numpy 1.19.2
    - opencv-python
    - some-custom-lib>=1.0
    
    ## Installation
    You can install it using:
    ```bash
    pip install torch==1.7.1 torchvision==0.8.2 numpy==1.19.5
    ```
    And make sure to install cmake via apt: `apt-get install cmake`.
    """

    print("Extracting dependencies from mock README...")
    deps = _extract_readme_dependencies(mock_readme)
    print("Extracted requirements:")
    print("-" * 40)
    print(deps)
    print("-" * 40)

    assert "torch" in deps.lower(), "Should extract torch"
    assert "numpy" in deps.lower(), "Should extract numpy"
    print("SUCCESS: Extraction test passed!")

if __name__ == "__main__":
    test_readme_deps_extraction()
