import argparse
import json
import tqdm
import os
from common import generate_questions_dataset


def extract_claims(text, entity, prompt):
    ref_prompt = prompt.replace("{INPUT}", text)
    ref_prompt = ref_prompt.replace("{ENTITY}", entity)
    gpt_resp = generate_questions_dataset.run_gpt(ref_prompt).strip()
    claims = gpt_resp.split("-- ")[1:]
    return claims


def extract_claims_from_paragraph(paragraphs_path=None, dataset_output_path=None,prompt_path=None):
    entity_question_answers_claims = {}

    with open(paragraphs_path) as json_file:
        entity_question_answers = json.load(json_file)

    with open(prompt_path) as file:
        prompt = file.read()

    for entity, question_answers in tqdm.tqdm(entity_question_answers.items()):
        entity_question_answers_claims[entity] = {}
        for question, answers in question_answers.items():
            entity_question_answers_claims[entity][question] = {}
            for answer in answers:
                entity_question_answers_claims[entity][question][answer] = (
                    extract_claims(answer, entity, prompt)
                )

    with open(dataset_output_path, "w") as outfile:
        json.dump(entity_question_answers_claims, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--paragraphs_path", type=str)
    parser.add_argument("--dataset_output_path", type=str)
    parser.add_argument("--prompt_path", type=str)
    parser.add_argument('--openai_key', type=str, default="./api.key", help="File containing OpenAI API Key. Default: ./api.key")
    args = parser.parse_args()

    assert os.path.exists(args.openai_key), f"Please place your OpenAI GPT Key in {args.openai_key}."
    with open(args.openai_key, 'r') as f:
        api_key = f.readline()
    generate_questions_dataset.api_key = api_key.strip()
    
    extract_claims_from_paragraph(
        paragraphs_path=args.paragraphs_path,
        dataset_output_path=args.dataset_output_path,
        prompt_path=args.prompt_path
    )
