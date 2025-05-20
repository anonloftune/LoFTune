import torch
import argparse
import json
import transformers
import tqdm
import logging
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


llm = None
sampling_params = None


def sample_model_questions(question_dataset_path=None, output_path=None, prompt = None, samples_per_prompt= None, is_chat=False):
    with open(question_dataset_path) as json_file:
        entity_questions_data = json.load(json_file)

    entity_question_answers = {}
    for entity, questions in tqdm.tqdm(entity_questions_data.items()):
        entity_question_answers[entity] = {}
        for question in questions:
            question = question.strip()
            if is_chat:
                prompts = [[{"role": "system","content": "You are a helpful assistant. Answer in a short paragraph."},{'role': 'user', 'content': question}] for _ in range(samples_per_prompt)]
                outputs = llm.chat(messages=prompts, sampling_params=sampling_params)
            else:
                prompts = [prompt.replace("{question}", question)] # fewshot_prompt.replace("{question}", question)
                outputs = llm.generate(prompts*samples_per_prompt, sampling_params)
            entity_question_answers[entity][question] = [output.outputs[0].text.strip() for output in outputs]

    with open(output_path, "w") as outfile:
        json.dump(entity_question_answers, outfile)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_dataset_path", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--temperature", default = 0.6, type=float, help= "default: 0.6")
    parser.add_argument("--samples_per_prompt", default = 6, type=int, help= "default: 6")
    parser.add_argument("--prompt_path", type=str)

    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        level=logging.CRITICAL)
    
    with open(args.prompt_path) as file:
        prompt = file.read()
    
    llm = LLM(model=args.model_name_or_path
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
    stop_token_ids=stop_token_ids
)
    is_chat = "chat" in args.model_name_or_path
    print(f"is_chat set to: {is_chat}")
    sample_model_questions(
        question_dataset_path=args.question_dataset_path, output_path=args.output_path, prompt=prompt, samples_per_prompt=args.samples_per_prompt, is_chat=is_chat
    )
