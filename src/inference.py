import argparse
import json
from typing import List, Union, Dict
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.determine_device import determine_device
from datasets import load_dataset
import sacrebleu
import re
from codebleu import calc_codebleu

MODEL_MAP = {
    "llama38": "meta-llama/Meta-Llama-3-8B",
    "gemma": "google/gemma-7b",
    "salesforce": "Salesforce/codegen-350M-mono",
    "finetuned_salesforce": "./src/models/salesforce/checkpoint-15000",
}


# def prepare_messages(file_path: str, model_type: str) -> List[Union[str, Dict[str, str]]]:
#     with open(file_path, "r") as f:
#         data = [json.loads(line) for line in f]
    
#     if model_type == "pretrained":
#         return [item["prompt"] for item in data]
#     else:
#         return [{"role": "user", "content": item["prompt"]} for item in data]
def prepare_messages(title: str, description: str, python_clean: str) -> List[Union[str, Dict[str, str]]]:
    # with open(file_path, "r") as f:
    #     data = [json.loads(line) for line in f]
    res = []
    for ti, des, python in zip(title, description, python_clean):
        res.append({"description": f"Solve the following Leetcode problem in Python: \n{des}", "python_clean": python, "title": ti})
    return res


def extract_solution_code(input_string):
    # Use regex to capture the full function definition, including "def"
    pattern = r'(def\s+\w+\(.*?\):(?:.+?)(?=\n\n|\n\S|\Z))'
    match = re.search(pattern, input_string, re.DOTALL)
    
    if match:
        # Return the full match with "def" included
        return match.group(1).strip()
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
    with open(output_file, "w") as f:
        for i in tqdm(range(0, len(messages), batch_size)):
            batch_messages = messages[i:i+batch_size]
            batch_prompts = [message if isinstance(message, str) else message["description"] for message in batch_messages]
            
           
            inputs =  inputs = tokenizer(
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

            for j, message in enumerate(batch_messages):
                print(response_decoded[j])
                if extract_solution_code(response_decoded[j]) != None:
                    try:
                        reference = [ message["python_clean"]]
                        candidate = extract_solution_code(response_decoded[j])
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

def main():
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()
    dataset = load_dataset("allenhung1025/leetcode")
    title = dataset['train']["title"]
    python = dataset['train']["python_clean"]
    content = dataset['train']['content']
    device = determine_device()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[args.model_type], 
        torch_dtype="float16"  # we need half-precision to fit into our machine
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[args.model_type])
    #tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    messages = prepare_messages(title, content, python)
    generate_output(model, tokenizer, messages, args.output, args.model_type, device, args.batch_size)

    print("Done!")


if __name__ == "__main__":
    main()