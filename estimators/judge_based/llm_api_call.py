import os
import requests
import json
import traceback
import hashlib
import logging


def calculate_md5(input_string):
    md5_hash = hashlib.md5()
    input_bytes = input_string.encode("utf-8")
    md5_hash.update(input_bytes)
    return md5_hash.hexdigest()


class GPTCaller:
    def __init__(self, api_url, cache_dir, api_key, model_name="gpt-4o-2024-08-06"):
        self.api_url = api_url
        self.cache_dir = cache_dir
        self.api_key = api_key
        self.model_name = model_name
        os.makedirs(self.cache_dir, exist_ok=True)

        self.llm_calls_count = 0
        self.cache_count = 0
        self.error_count = 0

    def call_llm(self, prompt, **kwargs):
        hed = {"Authorization": f"Bearer {self.api_key}"}
    
        data = {
            "model": self.model_name,
        }
    
        prompt_messages = []
        prompt_messages.append({"role": "user", "content": prompt})
        data["messages"] = prompt_messages

        resp_content = ""

        logger = logging.getLogger()

        try:
            cache_name = calculate_md5(str(data))
            cache_file = os.path.join(self.cache_dir, cache_name)
            to_be_cached = False
            if os.path.exists(cache_file):
                self.cache_count += 1
                logger.info(f"cache_name: {cache_name}")
                logger.info(f"Using cache for {prompt}. Cache count: [{self.cache_count}]")
                with open(cache_file, "r", encoding="UTF8") as cacheFile:
                    resp_content = cacheFile.read()
            else:
                response = requests.post(self.api_url, json=data, headers=hed)
                self.llm_calls_count += 1

                if response.status_code == 200:
                    json_resp = json.loads(response.text)
                    resp_content = json_resp["choices"][0]["message"]["content"].strip()

                    to_be_cached = True
                    logger.info(f"[ALERT] another GPT 4o / GPT 4o-mini invocation was made! LLM count: [{self.llm_calls_count}]")
                else:
                    self.error_count += 1
                    logger.info(f"[ERROR] GPT 4o / GPT 4o-mini invocation failed with status {response.status_code}; reason {response.reason}. Error count: [{self.error_count}]")
                    resp_content = ''

            if to_be_cached and not os.path.exists(cache_file):
                with open(cache_file, "w", encoding="UTF8") as f:
                    print(str(resp_content), file=f)
        except:
            traceback.print_exc()
            logger.error(f"error with json resp (prompt: {prompt})")

        return resp_content