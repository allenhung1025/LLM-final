from .constant import PREFIX, DATA_ROOT
from typing import List, Dict, Union
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import json

def prepare_training_data() -> None:
    # outdir = "train.jsonl"
    # if outdir not exist, create one
    ds = load_dataset("allenhung1025/leetcode")
    train = ds["train"]
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    file = DATA_ROOT / "train.jsonl"
    
    with open(file, "w") as f:
        for i in tqdm(range(len(train)), desc="Writing data to jsonl"):
            content = train[i]["content"]
            solution = train[i]["python"]
            # print(content)
            message = "".join([PREFIX, content])
            # Create a JSON object
            json_object = {
                "input": message,
                "output": solution
            }
            
            # Write the JSON object as a line in the file
            f.write(json.dumps(json_object) + "\n")


def load_training_data(filename: str) -> List[Dict[str, Union[str, str]]]:
    file = DATA_ROOT / filename
    print(file)
    with open(file, "r") as f:
        return [json.loads(line) for line in f.readlines()]

def tokenize_data(data: List[Dict[str, Union[str, str]]], tokenizer: AutoTokenizer, max_len: int = 512) -> List[Dict[str, Union[str, str]]]:
    # File path to save the tokenized data
    file = DATA_ROOT / "tokenized_train.jsonl"
    
    # Open the file for writing
    with open(file, "w") as f:
        for i in tqdm(range(len(data)), desc="Tokenizing data"):
            # Tokenize the input and output with max_len and padding
            input = tokenizer(data[i]["input"], 
                              return_tensors='pt', 
                              padding='max_length', 
                              truncation=True, 
                              max_length=max_len,
                              return_token_type_ids=False)
            
            output = tokenizer(data[i]["output"], 
                               return_tensors='pt', 
                               padding='max_length', 
                               truncation=True, 
                               max_length=max_len, 
                               return_token_type_ids=False)

            # Prepare the tokenized data dictionary
            tokenized = {
                "input_ids": input["input_ids"].squeeze(0).tolist(),  # Convert tensor to list (squeeze to remove batch dimension)
                "attention_mask": input["attention_mask"].squeeze(0).tolist(),  # Convert tensor to list
                "labels": output["input_ids"].squeeze(0).tolist()  # Labels are the tokenized output
            }

            # Write the tokenized data as a JSON object to the file
            f.write(json.dumps(tokenized) + "\n")

def load_tokenzied_data(model_type: str) -> List[Dict[str, Union[str, str]]]:
    file_name = model_type + "_tokenized_train.jsonl"
    file = DATA_ROOT / file_name
    with open(file, "r") as f:
        return [json.loads(line) for line in f.readlines()]



if __name__ == "__main__":
    # from transformers import AutoTokenizer 
    # data = load_training_data("train.jsonl")
    # tokenize_data(data, tokenizer=AutoTokenizer.from_pretrained("google/gemma-7b"))
    data = load_tokenzied_data("gemma")