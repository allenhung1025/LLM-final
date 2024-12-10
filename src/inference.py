import argparse
import json
from typing import List, Union, Dict
from tqdm import tqdm
import random
import torch
from torch.utils.data import random_split
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.determine_device import determine_device
from datasets import load_dataset
import sacrebleu
import re
from codebleu import calc_codebleu
from peft import PeftModel, PeftConfig

MODEL_MAP = {
    "llama38": "meta-llama/Meta-Llama-3-8B",
    "gemma": "google/gemma-7b",
    "salesforce": "Salesforce/codegen-350M-mono",
    "finetuned_salesforce": "./src/models/salesforce/checkpoint-15000",
    "llama_ft": "models/llama32-1b/fullft",
    "llama32-1b": "models/llama32-1b/pretrain",
    "llama32-lora": "models/llama32-1b/lora_ft",
    "llama32-prefix": "models/llama32-1b/prefixtuning_llama32-1b",
}

random.seed(123)
torch.random.manual_seed(123)
def prepare_zero_shot_messages(title: str, description: str, python_clean: str) -> List[Union[str, Dict[str, str]]]:
    # with open(file_path, "r") as f:
    #     data = [json.loads(line) for line in f]
    res = []
    for ti, des, python in zip(title, description, python_clean):
        res.append({"description": f"Solve the following Leetcode problem in Python: \n{des}", "python_clean": python, "title": ti})

    res = res[-int(len(res) * 0.2):]
    return res


def prepare_few_shots_messages(title: List[str], description: List[str], python_clean: List[str], num_shots: int = 0) -> List[Union[str, Dict[str, str]]]:
    """
    Prepare input messages for the model, with optional few-shot examples.

    Args:
        title (List[str]): List of titles of the problems.
        description (List[str]): List of problem descriptions.
        python_clean (List[str]): List of Python solutions.
        num_shots (int): Number of few-shot examples to include in the prompt.

    Returns:
        List[Dict[str, str]]: A list of dictionaries containing input messages and solutions.
    """
    res = []
    few_shot_examples = []

    # Prepare few-shot examples if num_shots > 0
    if num_shots > 0:
        indices = random.sample(range(len(title)), num_shots)
        for idx in indices:
            few_shot_examples.append({
                "title": title[idx],
                "description": description[idx],
                "python_clean": python_clean[idx]
            })

    # Prepare prompts with few-shot examples included
    for ti, des, python in zip(title, description, python_clean):
        few_shot_prompt = f"Solve a Leetcode problem in Python given the following {num_shots} of Leetcode examples:\n\n"
        if num_shots > 0:
            for example in few_shot_examples:
                few_shot_prompt += (
                    f"### Problem Title: {example['title']}\n"
                    f"### Problem Description:\n{example['description']}\n"
                    f"### Solution:\n{example['python_clean']}\n\n"
                )

        # Append the actual task after the few-shot examples
        res.append({
            "description": (
                f"{few_shot_prompt}"
                f"### Problem Title: {ti}\n"
                f"### Problem Description:\nSolve the following Leetcode problem in Python:\n{des}"
            ),
            "python_clean": python,
            "title": ti
        })
    res = res[-int(len(res) * 0.2):]
    return res



def extract_solution_code(generated_text, prefix="### Your solution:\n\n"):
    # Use regex to capture the full function definition, including "def"
    if generated_text.startswith(prefix):
        generated_text = generated_text[len(prefix):]

    pattern = r'(def\s+\w+\(.*?\):(?:.+?)(?=\n\n|\n\S|\Z))'
    match = re.search(pattern, generated_text, re.DOTALL)
    
    if match:
        # Return the full match with "def" included
        return match.group(0).strip()
    else:
        return None

bleu_list = []
codebleu_list = []
@torch.inference_mode()
def generate_output(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    messages: List[Union[str, Dict[str, str]]],
    output_file: str,
    model_type: str,
    device: str,
    batch_size: int = 32,
):
    prefix = "### Your solution:\n\n"  # Define the prefix
    with open(output_file, "w") as f:
        for i in tqdm(range(0, len(messages), batch_size)):
            batch_messages = messages[i:i+batch_size]
            batch_prompts = [message if isinstance(message, str) else message["description"] for message in batch_messages]
            
           
            inputs = tokenizer(
                batch_prompts,
                return_tensors='pt', 
                return_token_type_ids=False, 
                padding=True, 
                truncation=True
            )
            #print(inputs.shape)
           

            # if model_type != "pretrained" and not isinstance(inputs, dict):
            #     inputs = {'input_ids': inputs}
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            response = model.generate(**inputs, max_new_tokens=300)
            response_decoded = tokenizer.batch_decode(response, skip_special_tokens=True)
            # import pdb; pdb.set_trace()

            for j, message in enumerate(batch_messages):
                generate_output = prefix + response_decoded[j]
                print(response_decoded[j])
                if extract_solution_code(generate_output, prefix=prefix) != None:
                    try:
                        reference = [message["python_clean"]]
                        candidate = extract_solution_code(generate_output, prefix=prefix)
                        print("title: ", message["title"])
                        print(candidate)
                        sacrebleu_score = sacrebleu.sentence_bleu(candidate, reference) 
                        print(f"SacreBLEU: {sacrebleu_score.score}")
                        bleu_list.append(sacrebleu_score.score)


                        prediction = candidate
                        reference = reference[0]

                        result = calc_codebleu([reference], [prediction], lang="python", weights=(0.25, 0.25, 0.25, 0.25), tokenizer=None)
                        print(f"CodeBLEU: {result['codebleu'] * 100: .2f}")
                        codebleu_list.append(result['codebleu'] * 100)
                    except:
                        continue
                json.dump({"title": message["title"], "prompt": batch_prompts[j], "output": response_decoded[j], "groundtruth": message["python_clean"]}, f)
                f.write("\n")
    # Calculate average BLEU score
        avg_bleu = sum(bleu_list) / len(bleu_list)

        # Calculate average CodeBLEU score
        avg_codebleu = sum(codebleu_list) / len(codebleu_list)

        # Print with .2f formatting
        print(f"Average BLEU: {avg_bleu:.2f}")
        print(f"Average CodeBLEU: {avg_codebleu:.2f}")

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--base_model", 
        type=str,
        required=True, 
        choices=MODEL_MAP.keys(),
        help="The type of the pretrained model"
    )
    parser.add_argument(
        "--model_type", 
        type=str,
        required=True, 
        choices=MODEL_MAP.keys(),
        help="The type of the model: pretrained, sft, or instruct"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="A jsonl file where each line will have the input and the corresponding output"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Number of prompts to process in each batch"
    )
    parser.add_argument(
        "--peft", 
        type=bool,
        default=False,
        required=False, 
        help="If the model is a LORA model, specify it's lora"
    )
    # parser.add_argument(
    #     "--prefix", 
    #     type=bool,
    #     default=False,
    #     required=False, 
    #     help="If the model is a LORA model, specify it's lora"
    # )
    parser.add_argument(
        "--num_shots",
        type=int,
        default=0,
        help="Number of few-shot examples to include in the prompt"
    )
    return parser.parse_args()
    


def main():
    

    args = parse_args()
    dataset = load_dataset("allenhung1025/leetcode")
    title = dataset['train']["title"]
    python = dataset['train']["python_clean"]
    content = dataset['train']['content']
    
    
    device = determine_device()
    print(f"Model used: {MODEL_MAP[args.model_type]}")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[args.base_model], 
        torch_dtype="float16"  # we need half-precision to fit into our machine
    ).to(device)
    
    if args.peft:
        print("Loading Peft model")
        model = PeftModel.from_pretrained(model, MODEL_MAP[args.model_type]).to(device)
    
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[args.base_model])
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    if args.num_shots > 0:
        messages = prepare_few_shots_messages(title, content, python, args.num_shots)
    else:
        messages = prepare_zero_shot_messages(title, content, python)
    generate_output(model, tokenizer, messages, args.output, args.model_type, device, args.batch_size)

    print("Done!")


if __name__ == "__main__":
    main()