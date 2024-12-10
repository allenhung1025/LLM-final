MODEL="llama32-1b"
PROJECT_ROOT="/home/ubuntu/LLM-final"
# export WANDB_PROJECT="huggingface"
# WANDB_PROJECT=$MODEL python train.py --model $MODEL --configs llama_configs

# Lora
# WANDB_PROJECT=$MODEL python train.py --model $MODEL --configs llama_configs --lora_configs lora_configs.json

# Prefix tuning
WANDB_PROJECT=$MODEL python train.py --model $MODEL --configs llama_configs  --prefix_configs "$PROJECT_ROOT/configs/prefix_tuning_configs.json"
