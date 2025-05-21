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


class LLMCaller:
    def __init__(self, api_url, cache_dir):
        self.api_url = api_url
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

        self.llm_calls_count = 0
        self.cache_count = 0
        self.error_count = 0

    def call_llm(self, prompt, **kwargs):
        data = {
            "text": prompt,
            "input": "",
            "instruction": "",
            "max_length": kwargs.get("max_length", 256),
            "temperature": kwargs.get("temperature", 0.1),
            "top_p": kwargs.get("top_p", 0.75),
            "top_k": kwargs.get("top_k", 40),
            "num_beams": kwargs.get("num_beams", 3),
            "use_beam_search": kwargs.get("use_beam_search", True),
            "input_auto_trunc": kwargs.get("input_auto_trunc", True),
            "stream": False,
            "lang": kwargs.get("lang", "en"),
        }

        resp_content = ""

        logger = logging.getLogger()

        try:
            cache_name = calculate_md5(str(data))
            cache_file = os.path.join(self.cache_dir, cache_name)
            to_be_cached = False
            if os.path.exists(cache_file):
                self.cache_count += 1
                logger.info(f"Using cache for {prompt}. Cache count: [{self.cache_count}]")
                with open(cache_file, "r", encoding="UTF8") as cacheFile:
                    resp_content = cacheFile.read()
            else:
                response = requests.post(self.api_url, json=data, headers={})
                self.llm_calls_count += 1

                if response.status_code == 200:
                    json_resp = json.loads(response.text)
                    resp_content = json_resp.strip()

                    to_be_cached = True
                    logger.info(f"[ALERT] another ELMI invocation was made! LLM count: [{self.llm_calls_count}]")
                else:
                    self.error_count += 1
                    logger.info(f"[ERROR] ELMI invocation failed with status {response.status_code}; reason {response.reason}. Error count: [{self.error_count}]")
                    resp_content = ''

            if to_be_cached and not os.path.exists(cache_file):
                with open(cache_file, "w", encoding="UTF8") as f:
                    print(str(resp_content), file=f)
        except:
            traceback.print_exc()
            logger.error(f"error with json resp (prompt: {prompt})")

        return resp_content


class GPTCaller:
    def __init__(self, api_url, cache_dir, api_key):
        self.api_url = api_url
        self.cache_dir = cache_dir
        self.api_key = api_key
        os.makedirs(self.cache_dir, exist_ok=True)

        self.llm_calls_count = 0
        self.cache_count = 0
        self.error_count = 0

    def call_llm(self, prompt, **kwargs):
        # data = {
        #     "text": prompt,
        #     "input": "",
        #     "instruction": "",
        #     "max_length": kwargs.get("max_length", 256),
        #     "temperature": kwargs.get("temperature", 0.1),
        #     "top_p": kwargs.get("top_p", 0.75),
        #     "top_k": kwargs.get("top_k", 40),
        #     "num_beams": kwargs.get("num_beams", 3),
        #     "use_beam_search": kwargs.get("use_beam_search", True),
        #     "input_auto_trunc": kwargs.get("input_auto_trunc", True),
        #     "stream": False,
        #     "lang": kwargs.get("lang", "en"),
        # }

        # url = "https://api.openai.com/v1/chat/completions"
        hed = {"Authorization": f"Bearer {self.api_key}"}
    
        data = {
            "model": "gpt-4o-mini-2024-07-18",
        }
        # gpt-3.5-turbo-0125
    
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
                    logger.info(f"[ALERT] another GPT 3.5 invocation was made! LLM count: [{self.llm_calls_count}]")
                else:
                    self.error_count += 1
                    logger.info(f"[ERROR] GPT 3-5 invocation failed with status {response.status_code}; reason {response.reason}. Error count: [{self.error_count}]")
                    resp_content = ''

            if to_be_cached and not os.path.exists(cache_file):
                with open(cache_file, "w", encoding="UTF8") as f:
                    print(str(resp_content), file=f)
        except:
            traceback.print_exc()
            logger.error(f"error with json resp (prompt: {prompt})")

        return resp_content
