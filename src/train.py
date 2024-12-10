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
from utils.data_prepare import load_training_data, TextDataset, prepare_training_data
from utils.print_model_params import print_trainable_parameters
# from trainer import Trainer
import wandb
import os
from peft import LoraConfig, get_peft_model, PrefixTuningConfig




torch.manual_seed(123)
torch.cuda.empty_cache()
device = determine_device()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# uncomment this line to download the data
prepare_training_data()

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
    
    parser.add_argument(
    "--lora_configs",
    type=str,
    default=None,  # Default value is None
    required=False,  # Not mandatory
    help="Config filename for the model training (file path)"
    )
    
    parser.add_argument(
    "--prefix_configs",
    type=str,
    default=None,  # Default value is None
    required=False,  # Not mandatory
    help="Config filename for prefix tuning (file path)"
    )
    
    return parser.parse_args()


class CausalDataCollator:
    def __init__(self, tokenizer=None, mlm=False):
        self.tokenizer = tokenizer
        self.mlm = mlm

    def __call__(self, features):
        # Pad the features
        batch = self.tokenizer.pad(
            features, 
            padding=True, 
            return_tensors='pt'
        )

        # Shift labels by one token to the right
        labels = batch['input_ids'].clone()
        labels[:, :-1] = batch['input_ids'][:, 1:]
        labels[:, -1] = self.tokenizer.pad_token_id

        # Replace padding with -100 to ignore in loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        batch['labels'] = labels
        return batch


def main():
    args = parse_args()
    
    os.environ["WANDB_PROJECT"] = args.model
    
    with open("wandb_api.json") as json_file:
        credentials = json.load(json_file)
    
    wandb_api = credentials['wandb_api_key']
    parser = HfArgumentParser((ModelConfig, TrainingArguments))
    filename = args.configs + ".json"
    model_config, training_args = parser.parse_json_file(json_file=PROJECT_ROOT / "configs" / filename)
    
    
    wandb.login(key=wandb_api)
    
    data = load_training_data("../data/train.jsonl")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[args.model])
    tokenizer.pad_token = tokenizer.eos_token
    dataset = TextDataset(data, tokenizer, model_config.seq_len)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size

    # Use random_split to split the dataset
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    # import pdb; pdb.set_trace()
    # data prepare for causal language model 
    # data_collator = CausalDataCollator(tokenizer, mlm=False)
    
    # original 
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    
    # model + flash attention
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[args.model], 
        torch_dtype=torch.bfloat16,  # we need half-precision to fit into our machine
        attn_implementation= model_config.attention_type,
        trust_remote_code=True
    ).to(device)

    
    # LORA
    if args.lora_configs:
        with open(args.lora_configs, "r") as file:
            lora_config_dict = json.load(file)
        lora_configs = LoraConfig(**lora_config_dict)
        print("Initializing LoRA model")
        model = get_peft_model(model, lora_configs)
    
    if args.prefix_configs:
        # Load JSON file
        with open(args.prefix_configs, "r") as file:
            peft_config_dict = json.load(file)
        peft_config = PrefixTuningConfig(**peft_config_dict)
        print("Initializing Prefix Tuning model")
        model = get_peft_model(model, peft_config)


    print_trainable_parameters(model)
    
    
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
            f.write(f"{training_args.run_name} Perplexity ({seq_len}): {perplexity:.2f}\n")
        logger.info(f"{training_args.run_name} Perplexity ({seq_len}): {perplexity:.2f}")
        
    
    # Elvaluate
    results = trainer.evaluate(eval_dataset=val_dataset)
    import pdb; pdb.set_trace()
    print(f"Perplexity (seq_len = {model_config.seq_len}): {math.exp(results['eval_loss']):.2f}")
    
    perplexity = math.exp(results['eval_loss'])
    log_perplexity(log_file_path,  model_config.seq_len, perplexity)


if __name__ == "__main__":
    main()