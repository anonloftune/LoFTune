import argparse
import json
import tqdm
import os
import logging
from judge_based.basic_utils import decorate_str_with_date

from judge_based.llms import LongGPTEval

metric = None

def evaluate_factuality_llm(paragraphs_path=None, dataset_output_path=None):
    entity_question_answers_claims_score = {}

    with open(paragraphs_path) as json_file:
        entity_question_answers = json.load(json_file)
        
    for entity, question_answers in tqdm.tqdm(entity_question_answers.items()):
        entity_question_answers_claims_score[entity] = {}
        for question, answers in question_answers.items():
            entity_question_answers_claims_score[entity][question] = {}
            for answer in answers:
                entity_question_answers_claims_score[entity][question][answer] = metric.compute(answer)

    with open(dataset_output_path, "w") as outfile:
        json.dump(entity_question_answers_claims_score, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paragraphs_path", type=str)
    parser.add_argument("--dataset_output_path", type=str)
    parser.add_argument('--openai_key', type=str, default="./api.key", help="File containing OpenAI API Key. Default: ./api.key")

    args = parser.parse_args()

    assert os.path.exists(args.openai_key), f"Please place your OpenAI GPT Key in {args.openai_key}."
    with open(args.openai_key, 'r') as f:
        api_key = f.readline()

    log_filename = f"{decorate_str_with_date('logs')}.txt"
    
    logging.basicConfig(
        filename=log_filename,
        filemode="a",
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)-8s.%(msecs)03d  %(name)s:%(lineno)-3s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    logger = logging.getLogger()
    logger.info("=================== InsuranceQA EVALUATION ===================")

    metric = LongGPTEval(api_url="https://api.openai.com/v1/chat/completions", cache_dir=".cache_GPT4o/", api_key=api_key)
    
    evaluate_factuality_llm(
        paragraphs_path=args.paragraphs_path,
        dataset_output_path=args.dataset_output_path
    )
