import argparse
from datasets import load_dataset
import json
from typing import List, Union, Dict
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import determine_device


MODEL_MAP = {
    "llama38": "meta-llama/Meta-Llama-3-8B",
    "gemma": "google/gemma-7b"
}


def prepare_messages(file_path: str, model_type: str) -> List[Union[str, Dict[str, str]]]:
    

def main():
    ds = load_dataset("RayBernard/leetcode")