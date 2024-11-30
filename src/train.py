from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    HfArgumentParser
)
import argparse
import json
import os
import torch
import math
from torch.utils.data import random_split
import logging
from dataclasses import dataclass, field
from utils.constant import MODEL_MAP, PROJECT_ROOT
from utils.determine_device import determine_device
from utils.data_prepare import load_training_data, TextDataset
# from trainer import Trainer
import wandb

torch.cuda.empty_cache()
device = determine_device()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



@dataclass
class ModelConfig:
    seq_len: int = field(default=512)
    attention_type: str = field(default="flash_attention_2")

def parse_args():
    parser = argparse.ArgumentParser()
    
    # parser.add_argument(
    #     "--run_name", 
    #     type=str,
    #     required=True, 
    #     help="Name of the run"
    # )
    parser.add_argument(
        "--model", 
        type=str,
        required=True, 
        choices=MODEL_MAP.keys(),
        help="The model to train, also the name of the model and config files "
    )
    
    parser.add_argument(
        "--configs",
        type=str,
        required=True,
        help="config filename for the model training"
    )
    
    return parser.parse_args()


def main():
    with open("wandb_api.json") as json_file:
        credentials = json.load(json_file)
    
    args = parse_args()
    wandb_api = credentials['wandb_api_key']
    parser = HfArgumentParser((ModelConfig, TrainingArguments))
    filename = args.configs + ".json"
    model_config, training_args = parser.parse_json_file(json_file=PROJECT_ROOT / "configs" / filename)
    
    wandb.login(key=wandb_api)
    
    
    data = load_training_data(args.model + "_train.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[args.model])
    tokenizer.pad_token = tokenizer.eos_token
    dataset = TextDataset(data, tokenizer, model_config.seq_len)
    
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Use random_split to split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    

    # data prepare for causal language model 
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # model + flash attention
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[args.model], 
        torch_dtype=torch.bfloat16,  # we need half-precision to fit into our machine
        attn_implementation= model_config.attention_type,
        trust_remote_code=True
    ).to(device)
    
    # Trainer initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir)
    
    
    ### add by me to save log of perplexity
    log_file_path = os.path.join(training_args.output_dir, "perplexity_log.txt")

    def log_perplexity(log_file_path, seq_len, perplexity):
        with open(log_file_path, "a") as f:
            f.write(f"Perplexity ({seq_len}): {perplexity:.2f}\n")
        logger.info(f"Perplexity ({seq_len}): {perplexity:.2f}")
        
    
    # Elvaluate
    results = trainer.evaluate(eval_dataset=val_dataset)
    print(f"Perplexity (seq_len = {training_args.seq_len}): {math.exp(results['eval_loss']):.2f}")
    
    perplexity = math.exp(results['eval_loss'])
    log_perplexity(log_file_path, training_args.seq_len, perplexity)


if __name__ == "__main__":
    main()