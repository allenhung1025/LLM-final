# MODEL="llama32-lora"
# full ft
# python inference.py --model_type $MODEL --output res/llama_ft.jsonl
# lora
# python inference.py --model_type $MODEL --output res/llama_lora_one_shot.jsonl --lora true --num_shots 0
MODEL="llama32-prefix"
# prefix
python inference.py --base_model llama32-1b --model_type $MODEL --output res/llama_prefix_zero_shot.jsonl --peft true --num_shots 0