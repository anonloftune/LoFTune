import argparse
import json
import tqdm
import torch
import pathlib
import pickle
import os
import logging
from common.generate_questions_dataset import calculate_md5
from vllm import LLM, SamplingParams

llm = None
sampling_params = None
    
    
def answer_questions(question_dataset_path=None, dataset_output_path=None, prompt_path=None):
    with open(question_dataset_path) as json_file:
        entity_question_answers_claims_questions = json.load(json_file)

    with open(prompt_path) as file:
        prompt = file.read()
    
    basename = pathlib.Path(dataset_output_path).stem
    if not os.path.exists(".cache/"+basename+"/"):
        os.mkdir(".cache/"+basename+"/")
    
    entity_question_answers_claims_questions_answers = {}
    for entity, question_answers_claims_questions in tqdm.tqdm(
        entity_question_answers_claims_questions.items()
    ):
        entity_question_answers_claims_questions_answers[entity] = {}
        for (
            question,
            answers_claims_questions,
        ) in question_answers_claims_questions.items():
            entity_question_answers_claims_questions_answers[entity][question] = {}
            for answer, claims_questions in answers_claims_questions.items():
                entity_question_answers_claims_questions_answers[entity][question][
                    answer
                ] = {}
                for claim, fact_question in claims_questions.items():
                    cache_name = calculate_md5(fact_question)
                    cache_file = ".cache/" + basename + "/"+ cache_name + ".cache"
                    if os.path.exists(cache_file):
                        with open(cache_file, "rb") as cacheFile:
                            answers = pickle.load(cacheFile)
                    else:
                        prompts = [prompt.replace("{question}", fact_question)]
                        outputs = llm.generate(prompts*20, sampling_params)
                        answers = [output.outputs[0].text.strip() for output in outputs]
                        with open(cache_file, "wb") as f:
                            pickle.dump(answers, f)
                    
                    entity_question_answers_claims_questions_answers[entity][question][
                        answer
                    ][claim] = {
                        "question": fact_question,
                        "answers": answers,
                    }


    with open(dataset_output_path, "w") as outfile:
        json.dump(entity_question_answers_claims_questions_answers, outfile)

def answer_questions_batch(question_dataset_path=None, dataset_output_path=None, prompt_path=None):
    with open(question_dataset_path) as json_file:
        entity_question_answers_claims_questions = json.load(json_file)

    with open(prompt_path) as file:
        prompt = file.read()
    
    basename = pathlib.Path(dataset_output_path).stem
    if not os.path.exists(".cache/"+basename+"/"):
        os.mkdir(".cache/"+basename+"/")
    
    entity_question_answers_claims_questions_answers = {}
    for entity, question_answers_claims_questions in tqdm.tqdm(
        entity_question_answers_claims_questions.items()
    ):
        entity_question_answers_claims_questions_answers[entity] = {}
        for (
            question,
            answers_claims_questions,
        ) in question_answers_claims_questions.items():
            entity_question_answers_claims_questions_answers[entity][question] = {}
            for answer, claims_questions in answers_claims_questions.items():
                entity_question_answers_claims_questions_answers[entity][question][
                    answer
                ] = {}
                prompts = []
                saved_questions = []
                saved_claims = []
                for claim, fact_question in claims_questions.items():
                    cache_name = calculate_md5(fact_question)
                    cache_file = ".cache/" + basename + "/"+ cache_name + ".cache"
                    if os.path.exists(cache_file):
                        with open(cache_file, "rb") as cacheFile:
                            answers = pickle.load(cacheFile)
                        entity_question_answers_claims_questions_answers[entity][question][
                            answer
                        ][claim] = {
                            "question": fact_question,
                            "answers": answers,
                        }
                    else:
                        # add question to the list
                        saved_questions.append(fact_question)
                        saved_claims.append(claim)
                        prompts = prompts+[prompt.replace("{question}", fact_question)]*20
                        
                if len(prompts)>0:
                    outputs = llm.generate(prompts, sampling_params)
                    answers = [output.outputs[0].text.strip() for output in outputs]
    
                    for i in range(int(len(prompts)/20)):
                        cache_name = calculate_md5(saved_questions[i])
                        cache_file = ".cache/" + basename + "/"+ cache_name + ".cache"
                        i_answers = answers[i*20:i*20+20]
                        with open(cache_file, "wb") as f:
                            pickle.dump(i_answers, f)
                        entity_question_answers_claims_questions_answers[entity][question][
                            answer
                        ][saved_claims[i]] = {
                            "question": saved_questions[i],
                            "answers": i_answers,
                        }


    with open(dataset_output_path, "w") as outfile:
        json.dump(entity_question_answers_claims_questions_answers, outfile)

def answer_questions_batch2(question_dataset_path=None, dataset_output_path=None, prompt_path=None):
    with open(question_dataset_path) as json_file:
        entity_question_answers_claims_questions = json.load(json_file)

    with open(prompt_path) as file:
        prompt = file.read()
    
    basename = pathlib.Path(dataset_output_path).stem
    if not os.path.exists(".cache/"+basename+"/"):
        os.mkdir(".cache/"+basename+"/")
    
    entity_question_answers_claims_questions_answers = {}
    for entity, question_answers_claims_questions in tqdm.tqdm(
        entity_question_answers_claims_questions.items()
    ):
        entity_question_answers_claims_questions_answers[entity] = {}
        for (
            question,
            answers_claims_questions,
        ) in question_answers_claims_questions.items():
            entity_question_answers_claims_questions_answers[entity][question] = {}
            prompts = []
            saved_questions = []
            saved_claims = []
            saved_answers = []
            for answer, claims_questions in answers_claims_questions.items():
                entity_question_answers_claims_questions_answers[entity][question][
                    answer
                ] = {}
                for claim, fact_question in claims_questions.items():
                    cache_name = calculate_md5(fact_question)
                    cache_file = ".cache/" + basename + "/"+ cache_name + ".cache"
                    if os.path.exists(cache_file):
                        with open(cache_file, "rb") as cacheFile:
                            answers = pickle.load(cacheFile)
                        entity_question_answers_claims_questions_answers[entity][question][
                            answer
                        ][claim] = {
                            "question": fact_question,
                            "answers": answers,
                        }
                    else:
                        # add question to the list
                        saved_answers.append(answer)
                        saved_claims.append(claim)
                        saved_questions.append(fact_question)
                        prompts = prompts+[prompt.replace("{question}", fact_question)]*20
                        
            if len(prompts)>0:
                outputs = llm.generate(prompts, sampling_params)
                answers = [output.outputs[0].text.strip() for output in outputs]

                for i in range(int(len(prompts)/20)):
                    cache_name = calculate_md5(saved_questions[i])
                    cache_file = ".cache/" + basename + "/"+ cache_name + ".cache"
                    if os.path.exists(cache_file):
                        with open(cache_file, "rb") as cacheFile:
                            i_answers = pickle.load(cacheFile)
                    else:
                        i_answers = answers[i*20:i*20+20]
                        with open(cache_file, "wb") as f:
                            pickle.dump(i_answers, f)
                    entity_question_answers_claims_questions_answers[entity][question][
                        saved_answers[i]
                    ][saved_claims[i]] = {
                        "question": saved_questions[i],
                        "answers": i_answers,
                    }

                        


    with open(dataset_output_path, "w") as outfile:
        json.dump(entity_question_answers_claims_questions_answers, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_dataset_path", type=str)
    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--dataset_output_path", type=str)
    parser.add_argument("--prompt_path", type=str)

    args = parser.parse_args()

    logging.basicConfig(format='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
                        datefmt='%m-%d %H:%M:%S',
                        level=logging.CRITICAL)
    
    llm = LLM(model=args.model_name_or_path, gpu_memory_utilization=0.95
              # dtype=torch.float16, # torch.bfloat16
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
        temperature=0.6, 
        top_p=0.9, 
        top_k=50, 
        stop_token_ids=stop_token_ids
    )

    answer_questions_batch2(
        question_dataset_path=args.question_dataset_path,
        dataset_output_path=args.dataset_output_path,
        prompt_path=args.prompt_path
    )
