import argparse
import json
from typing import List, Union, Dict
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, random_split
from transformers import AutoModelForCausalLM, AdamW, AutoTokenizer
from datasets import load_dataset
from utils.constant import MODEL_MAP
from utils.determine_device import determine_device
from utils.data_prepare import load_training_data, load_tokenzied_data, tokenize_data
from trainer import Trainer
import wandb




device = determine_device()

def collate_fn(batch):
    return {
        "input_ids": torch.stack([torch.tensor(example["input_ids"]) for example in batch]),
        "attention_mask": torch.stack([torch.tensor(example["attention_mask"]) for example in batch]),
        "labels": torch.stack([torch.tensor(example["labels"]) for example in batch])
    }
    
def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--run_name", 
        type=str,
        required=True, 
        help="Name of the run"
    )
    parser.add_argument(
        "--model", 
        type=str,
        required=True, 
        choices=MODEL_MAP.keys(),
        help="The type of the model: "
    )
    parser.add_argument(
        "--epochs",
        type=int,
        required=True,
        help="Training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        required=True,
        help="Batch size"
    )
    
    parser.add_argument(
        "--seqlen",
        type=int,
        required=True,
        help="Max sequence length"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        required=True,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="model checkpoint output path"
    )
    
    return parser.parse_args()


def main():
    with open("wandb_api.json") as json_file:
        credentials = json.load(json_file)
    
    args = parse_args()
    wandb_api = credentials['wandb_api_key']
    wandb.login(key=wandb_api)
    
    wandb.init(project="gemma_ft", config=args, name=args.run_name)
    
    

    # tokenizing
    data = load_training_data(args.model + "_train.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[args.model])
    tokenize_data(data, tokenizer=tokenizer, max_len=args.seqlen)
    dataset = load_tokenzied_data(args.model)
    
    # Create a DataLoader with a batch size of 8
    batch_size = args.batch_size
    train_size = int(0.8 * len(dataset))  # 80% for training
    val_size = len(dataset) - train_size  # 20% for validation

    # Split the dataset into training and validation sets
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # Create DataLoaders for training and validation sets
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=False)
    
    
    
    # model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[args.model], 
        torch_dtype="float16"  # we need half-precision to fit into our machine
    ).to(device)
    
    # Define optimizer (AdamW is commonly used for transformer models)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    
    # Training
    Trainer(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        save_path=args.output,
        args=args
    ).train()
    


if __name__ == "__main__":
    main()