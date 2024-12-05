MODEL="llama32-1b"
# export WANDB_PROJECT="huggingface"
WANDB_PROJECT=$MODEL python train.py --model $MODEL --configs llama_configs

# Lora
# WANDB_PROJECT=$MODEL python train.py --model $MODEL --configs llama_configs --lora_configs lora_configs.json