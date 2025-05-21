import torch
import argparse
import json
import transformers
import tqdm
import logging
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
# from datasets import load_from_disk
from datasets import Dataset
import pandas as pd
import evaluate

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "MIG-1b220517-a869-5a19-81ee-14311ccf560f"

llm = None
sampling_params = None


def sample_model_questions(entities_dataset_path=None, output_path=None, prompt = None, is_chat=False):
    with open(entities_dataset_path, "r") as file:
        entities = [line.strip() for line in file.readlines()]
    entities_ds = Dataset.from_dict({"entity":entities})
    
    if is_chat:
        all_prompts = [[{"role": "system","content": "You are a helpful assistant. Answer in a short paragraph."},{'role': 'user', 'content': f"Tell me a bio of {entity}."}] for entity in entities_ds["entity"]]
        outputs = llm.chat(messages=all_prompts, sampling_params=sampling_params)
    else:
        entities_ds_prompted = entities_ds.map(lambda ex: {"prompt":prompt.replace("{question}",f"Tell me a bio of {ex['entity']}.")})
        all_prompts = entities_ds_prompted["prompt"]
        outputs = llm.generate(all_prompts, sampling_params)
    
    answers = []
    for example,output in zip(entities_ds,outputs):
        answer = output.outputs[0].text.strip()
        answers.append({"topic":example["entity"],"output":answer})

    with open(output_path, 'w') as f:
        for d in answers:
            json.dump(d, f)
            f.write('\n')
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities_dataset_path", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--temperature", default = 0.6, type=float, help= "default: 0.6")
    parser.add_argument("--prompt_path", type=str, default = None)
    parser.add_argument("--lora_path", type=str, default=None)

    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        level=logging.CRITICAL)
    
    prompt = None
    if args.prompt_path:
        with open(args.prompt_path) as file:
            prompt = file.read()
    
    llm = LLM(model=args.model_name_or_path,
              enable_lora=args.lora_path is not None
              # seed=42
              # dtype=torch.bfloat16 # vllm detects the torchtype
             )
    
    tokenizer = llm.get_tokenizer()
    separator = "\n----\n"
    separator_token_id = tokenizer(separator,add_special_tokens=False)["input_ids"][1]
    if tokenizer.decode(separator_token_id)!='----':
        ## In llama-2 the first token is 29871, so try the next token
        separator_token_id = tokenizer(separator,add_special_tokens=False)["input_ids"][2]
    logging.critical(f'The token "{tokenizer.decode(separator_token_id)}" will be added to the stop_token_ids, please check it is correct. If you are using a zero-shot prompt, this should not affect to the generation.')
    stop_token_ids = [tokenizer.eos_token_id, separator_token_id]
    
    sampling_params = SamplingParams(
    max_tokens=512, 
    temperature=args.temperature, 
    top_p=0.9, 
    top_k=50, 
    stop_token_ids=stop_token_ids,
    seed=42,
    **{"lora_request":LoRARequest("lora_adapter", 1, args.lora_path)} if args.lora_path else {}
)
    is_chat = "chat" in args.model_name_or_path
    print(f"is_chat set to: {is_chat}")
    
    sample_model_questions(
        entities_dataset_path=args.entities_dataset_path, output_path=args.output_path, prompt=prompt, is_chat=is_chat
    )
