import argparse
import json
import tqdm
import os
import generate_questions_dataset


def claim_to_question(claim, entity, prompt):
    ref_prompt = prompt.replace("{STATEMENT}", claim)
    ref_prompt = ref_prompt.replace("{ENTITY}", entity)
    gpt_resp = generate_questions_dataset.run_gpt(ref_prompt).strip()
    question = gpt_resp.split("Statement:")[
        0
    ].strip()  ## If GPT3.5 generates more than the question we remove that part
    return question


def claims_to_questions(claims_path=None, dataset_output_path=None, prompt_path=None):
    with open(claims_path) as json_file:
        entity_question_answers_claims = json.load(json_file)

    with open(prompt_path) as file:
        prompt = file.read()

    entity_question_answers_claims_questions = {}
    for entity, question_answers_claims in tqdm.tqdm(
        entity_question_answers_claims.items()
    ):
        entity_question_answers_claims_questions[entity] = {}
        for question, answers_claims in question_answers_claims.items():
            entity_question_answers_claims_questions[entity][question] = {}
            for answer, claims in answers_claims.items():
                entity_question_answers_claims_questions[entity][question][answer] = {}
                for claim in claims:
                    claim = claim.strip()
                    entity_question_answers_claims_questions[entity][question][answer][
                        claim
                    ] = claim_to_question(claim, entity, prompt)

    with open(dataset_output_path, "w") as outfile:
        json.dump(entity_question_answers_claims_questions, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--claims_path", type=str)
    parser.add_argument("--dataset_output_path", type=str)
    parser.add_argument("--prompt_path", type=str)
    parser.add_argument('--openai_key', type=str, default="./api.key", help="File containing OpenAI API Key. Default: ./api.key")
    args = parser.parse_args()
    
    assert os.path.exists(args.openai_key), f"Please place your OpenAI GPT Key in {args.openai_key}."
    with open(args.openai_key, 'r') as f:
        api_key = f.readline()
    generate_questions_dataset.api_key = api_key.strip()
    
    claims_to_questions(
        claims_path=args.claims_path,
        dataset_output_path=args.dataset_output_path,
        prompt_path = args.prompt_path
    )
