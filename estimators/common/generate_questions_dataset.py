import argparse
import hashlib
import os
import random
import time
import requests
import json
import traceback

gpt_made_calls = 0
api_key = None


def calculate_md5(input_string):
    md5_hash = hashlib.md5()
    input_bytes = input_string.encode("utf-8")
    md5_hash.update(input_bytes)
    return md5_hash.hexdigest()


def run_gpt_generate_question(entity, ref_prompt = None):    
    ref_prompt = ref_prompt.replace("{ENTITY}", entity)
    gpt_resp = run_gpt(ref_prompt).strip()
    questions = gpt_resp.split("-- ")[1:]
    return questions


def run_gpt(prompt):

    global gpt_made_calls

    url = "https://api.openai.com/v1/chat/completions"
    hed = {"Authorization": f"Bearer {api_key}"}

    data = {
        "model": "gpt-3.5-turbo-0125",
    }

    prompt_messages = []
    prompt_messages.append({"role": "user", "content": prompt})
    data["messages"] = prompt_messages

    resp_content = ""

    try:
        cache_name = calculate_md5(str(data))
        gtp_cache_file = ".cache/" + cache_name + ".cache"
        gpt_to_be_cached = False
        if os.path.exists(gtp_cache_file):
            with open(gtp_cache_file, "r", encoding="UTF8") as cacheFile:
                resp_content = cacheFile.read()
                # if resp_content.find("logprobs")>=0:
                #    json_resp = json.loads(resp_content)
                #    resp_content = json_resp["choices"][0]["text"]
                #    pass
        else:

            time.sleep(random.uniform(0.1, 1.5))
            response = requests.post(url, json=data, headers=hed)
            gpt_made_calls += 1

            if response.status_code == 200:
                json_resp = json.loads(response.text)
                resp_content = json_resp["choices"][0]["message"]["content"]

                # print("out json:" +str(json_resp))
                gpt_to_be_cached = True
                print(
                    "\n\n\n##########\n[ALERT] another GPT invocation was made! ["
                    + str(gpt_made_calls)
                    + "]\n############\n\n\n"
                )
            else:
                print(
                    "\n\n\n##########\n[[ALERT]  GPT error ["
                    + str(response.status_code)
                    + "] ["
                    + str(response.reason)
                    + "]\n############\n\n\n"
                )

        if gpt_to_be_cached and not os.path.exists(gtp_cache_file):
            with open(gtp_cache_file, "w", encoding="UTF8") as f:
                print(str(resp_content), file=f)

    except:
        traceback.print_exc()
        print("error with json resp (prompt:" + prompt + ")")

    return resp_content


def generate_questions_from_entities(
    entities_path=None, dataset_output_path=None, prompt_path=None
):

    with open(prompt_path) as file:
        prompt = file.read()

    # entities = []
    entity_questions = {}
    with open(entities_path, encoding="UTF-8") as entities_file:
        for line in entities_file:
            entity = line.rstrip("\n")
            entity_questions[entity] = run_gpt_generate_question(
                entity=entity, ref_prompt=prompt
            )

    with open(dataset_output_path, "w") as outfile:
        json.dump(entity_questions, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--entities_path", type=str)
    parser.add_argument("--dataset_output_path", type=str)
    parser.add_argument("--prompt_path", type=str)
    parser.add_argument('--openai_key', type=str, default="./api.key", help="File containing OpenAI API Key. Default: ./api.key")
    args = parser.parse_args()

    assert os.path.exists(args.openai_key), f"Please place your OpenAI GPT Key in {args.openai_key}."
    with open(args.openai_key, 'r') as f:
        api_key = f.readline()
    api_key = api_key.strip()
    
    generate_questions_from_entities(
        entities_path=args.entities_path,
        dataset_output_path=args.dataset_output_path,
        prompt_path = args.prompt_path
    )
