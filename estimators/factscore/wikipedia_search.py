import argparse
import os
import json_repair
import yaml
import re
import logging

from judge_based.llm_api_call import GPTCaller

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--openai_key', type=str, default="./api.key", help="File containing OpenAI API Key. Default: ./api.key")
    parser.add_argument("--prompt_path", type=str)
    parser.add_argument('--entity_definitions',
                        type=str,
                        default="factscore/insurance-en-entities-definitions.yml", help="Yaml file containing entities and definitions. Default: factscore/insurance-en-entities-definitions.yml")
    parser.add_argument("--output_path", type=str)

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR)

    assert os.path.exists(args.openai_key), f"Please place your OpenAI GPT Key in {args.openai_key}."
    with open(args.openai_key, 'r') as f:
        api_key = f.readline()

    with open(args.prompt_path) as file:
        prompt = file.read()

    with open(args.entity_definitions, "r") as file: 
        entity_definitions = yaml.safe_load(file)

    gpt_caller = GPTCaller(api_url="https://api.openai.com/v1/chat/completions", cache_dir=".cache_GPT4o-mini-search/", api_key=api_key, model_name="gpt-4o-mini-search-preview-2025-03-11")

    output_dict = {}
    pattern = r"```(\w*)\n(.*?)\n```"
    pattern_json = r"({.*})"
    
    for entity,definition in entity_definitions.items():
        prompt_instance = prompt.replace("{term}", entity)
        prompt_instance = prompt_instance.replace("{definition}", definition)
        response = gpt_caller.call_llm(prompt_instance)
        matches = re.findall(pattern, response, re.DOTALL)
        if len(matches)==0:
            # Let's try to parse some json in the response
            matches = re.findall(pattern_json, response, re.DOTALL)
            if len(matches)>1:
                logging.error("More than one json extracted, please review this carefully")
            extracted_answer = matches[0]
        else:
            extracted_answer = matches[0][1]
        response_dict = json_repair.repair_json(extracted_answer, return_objects=True)
        output_dict[entity] = response_dict

    with open(args.output_path, 'w') as ff:
        yaml.dump(output_dict, ff)