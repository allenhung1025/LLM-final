import argparse
import json
from typing import List, Union, Dict
from tqdm import tqdm
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.determine_device import determine_device
from datasets import load_dataset


MODEL_MAP = {
    "llama38": "meta-llama/Meta-Llama-3-8B",
    "gemma": "google/gemma-7b",
    "salesforce": "Salesforce/codegen-350M-mono"
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
        res.append({"description": f"generate python code with the following description : {des}", "python_clean": python, "title": ti})
    return res


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
                json.dump({"title": message["title"], "prompt": batch_prompts[j], "output": response_decoded[j], "groundtruth": message["python_clean"]}, f)
                f.write("\n")


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
        "--prompts",
        type=str,
        required=True,
        help="A jsonl file where each line has an input prompt"
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
    # import pdb; pdb.set_trace()
    device = determine_device()
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_MAP[args.model_type], 
        torch_dtype="float16"  # we need half-precision to fit into our machine
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_MAP[args.model_type])
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    messages = prepare_messages(title, content, python)
    generate_output(model, tokenizer, messages, args.output, args.model_type, device, args.batch_size)

    print("Done!")


if __name__ == "__main__":
    main()