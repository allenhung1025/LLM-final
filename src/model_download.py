import argparse
import json
from typing import List, Union, Dict
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.determine_device import determine_device


MODEL_MAP = {
    "llama38": "meta-llama/Meta-Llama-3-8B",
    "gemma": "google/gemma-7b"
}


# download the pretrained model from huggingface 
@torch.inference_mode()
def download_model(model_type: str):
    model_name = MODEL_MAP[model_type]
    print(f"Start downloading {model_name} ...")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print("Download completed! \n huggingface caceh directory: ", tokenizer.cache_dir)
    return model, tokenizer



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        choices=["llama38", "gemma"],
        help="The type of model to download"
    )

    args = parser.parse_args()
    
    device = determine_device()
    model, tokenizer = download_model(args.model)
    print("Done!")
    
if __name__ == "__main__":
    main()
    