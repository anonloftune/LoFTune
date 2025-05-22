import abc
import logging
import os
import re

from judge_based.llm_api_call import GPTCaller
import json_repair
import json
import copy

class GPTEval(GPTCaller, metaclass=abc.ABCMeta):
    def __init__(self, api_url, cache_dir, api_key):
        super().__init__(api_url, cache_dir, api_key)

    @abc.abstractmethod
    def compute(self, text, **kwargs):
        raise NotImplementedError


class LongGPTEval(GPTEval):
    def __init__(self, api_url, cache_dir, api_key, n_generate_answers=10):
        cache_dir = os.path.join(cache_dir)
        super().__init__(api_url, cache_dir, api_key)
        self.n_generate_answers=n_generate_answers
        self.prompt_template = "You are a helpful and critical assistant. Your task is to evaluate the factuality of a given text. You should give an overall factuality score, ranging from 0 to 1, 0 meaning absolutely false and 1 completely truthful. Be extremely critical when providing the score. Structure the output as a JSON object, ensuring that all fields are included and meticulously populated according to the following structure:\r\n\r\n    class Output(BaseModel):\r\n        factuality_score: float\r\n\r\nProvide only the JSON object. Exclude comments and extra text.\r\n\r\n\"{TEXT}\"" 
        self.generate_answers_template = "Produce %s answers to the question \"{TEXT}\". Each answer must have a factuality score ranging between 0 and 1 and a precision of two decimals. Include the factuality score of each answer. Ensure a large variety of scores across answers and make sure that each one contains at least 5 sentences. Don't include explanations. Keep a professional language and stay focused on the question. Please structure the output as a JSON object, ensuring that all fields are included and meticulously populated according to the following structure:\r\n\r\n    class Answer(BaseModel):\r\n        answer: str\r\n        factuality_score: float\r\n\r\n    class Output(BaseModel):\r\n        answers: List[Answer]\r\n\r\nProvide only the JSON object. Exclude comments and extra text." % n_generate_answers

    def compute(self, text, **kwargs):
        logger = logging.getLogger()

        prompt = self.prompt_template.replace("{TEXT}", text)

        logger.info("Calling LLM")
        response = self.call_llm(prompt, **kwargs).strip()
        
        response_dict = json_repair.repair_json(response, return_objects=True)

        assert ("factuality_score" in response_dict)
        response_dict["score"] = response_dict["factuality_score"]
        del response_dict["factuality_score"]

        return response_dict

    def generate_answers_and_score(self, text, **kwargs):
        logger = logging.getLogger()

        prompt = self.generate_answers_template.replace("{TEXT}", text)

        logger.info("Calling LLM")
        response = self.call_llm(prompt, **kwargs).strip()
        
        response_dict = json_repair.repair_json(response, return_objects=True)

        assert ("answers" in response_dict) and (len(response_dict["answers"]) >= self.n_generate_answers), 'len(response_dict["answers"]: %s' % len(response_dict["answers"])
        if len(response_dict["answers"])!=self.n_generate_answers:
            response_dict["answers"]= response_dict["answers"][:self.n_generate_answers]
        
        return response_dict