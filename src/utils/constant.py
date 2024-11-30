from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PREFIX = "Solve the following Leetcode problem in Python: \n"
DATA_ROOT = PROJECT_ROOT / "data"

# Model map
MODEL_MAP = {
    "llama38": "meta-llama/Meta-Llama-3-8B",
    "gemma": "google/gemma-7b",
     "salesforce": "Salesforce/codegen-350M-mono"
}