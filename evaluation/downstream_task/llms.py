import abc
import logging
import os
import re

from llm_api_call import LLMCaller, GPTCaller

def find_first_number(input_string, start_pattern=""):
    _input_string = input_string
    if len(start_pattern) > 0:
        start_pos = _input_string.find(start_pattern)
        if start_pos > 0:
            offset = len(start_pattern)
            _input_string = _input_string[start_pos + offset :]

    pattern = r"\d+"
    numbers = re.findall(pattern, _input_string)
    if numbers:
        return numbers[0]
    else:
        return 0


class MistralEval(LLMCaller, metaclass=abc.ABCMeta):
    def __init__(self, api_url, cache_dir):
        super().__init__(api_url, cache_dir)

    @abc.abstractmethod
    def compute(self, predictions, references, **kwargs):
        raise NotImplementedError

class GPTEval(GPTCaller, metaclass=abc.ABCMeta):
    def __init__(self, api_url, cache_dir, api_key):
        super().__init__(api_url, cache_dir, api_key)

    @abc.abstractmethod
    def compute(self, predictions, references, **kwargs):
        raise NotImplementedError


class LongMistralEval(MistralEval):
    def __init__(self, api_url, cache_dir):
        cache_dir = os.path.join(cache_dir, "long")
        super().__init__(api_url, cache_dir)
        self.name = "elmi-check-long"
        self.prompt_template = "### TEXT_A:\n{TEXT_A}\n\n### TEXT_B:\n{TEXT_B}\n\n### QUESTION:\nHow similar is the information (facts, names, dates, values)  present in the TEXT_A with information present in the TEXT_B?\nAnswer with a number between 0 and 100.\n\n### ANSWER:\nSimilarity Number: "
        ## self.prompt_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n### TEXT_A:\n{TEXT_A}\n\n### TEXT_B:\n{TEXT_B}\n\n### QUESTION:\nHow similar is the information (facts, names, dates, values)  present in the TEXT_A with information present in the TEXT_B?\nAnswer with a number between 0 and 100.\n\n### ANSWER:\nSimilarity Number:<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

    def compute(self, predictions, references, **kwargs):
        logger = logging.getLogger()
        text_a = references[0].replace("\\n", "\n")
        text_b = predictions[0].replace("\\n", "\n")

        if text_a == text_b:
            logger.info(f"predictions {predictions} and references {references} are exactly the same. Returning 100.")
            return 100

        prompt = self.prompt_template.replace("{TEXT_A}", text_a)
        prompt = prompt.replace("{TEXT_B}", text_b)

        logger.info("Calling LLM text_a vs text_b")
        elmi_response = self.call_llm(prompt, **kwargs).strip()
        ab_elmi_number = find_first_number(elmi_response)

        logger.info("Calling LLM text_b vs text_a")
        rev_prompt = self.prompt_template.replace("{TEXT_A}", text_b)
        rev_prompt = rev_prompt.replace("{TEXT_B}", text_a)

        rev_elmi_response = self.call_llm(rev_prompt, **kwargs).strip()
        ba_elmi_number = find_first_number(rev_elmi_response)

        elmi_numb = max(int(ab_elmi_number), int(ba_elmi_number))

        if elmi_numb is not None:
            return elmi_numb
        else:
            return 0

class LongGPTEval(GPTEval):
    def __init__(self, api_url, cache_dir, api_key):
        cache_dir = os.path.join(cache_dir, "long")
        super().__init__(api_url, cache_dir, api_key)
        self.name = "gpt-check-long"
        self.prompt_template = "### TEXT_A:\n{TEXT_A}\n\n### TEXT_B:\n{TEXT_B}\n\n### QUESTION:\nHow similar is the information (facts, names, dates, values)  present in the TEXT_A with information present in the TEXT_B?\nAnswer with a number between 0 and 100.\n\n### ANSWER:\nSimilarity Number: "

    def compute(self, predictions, references, **kwargs):
        logger = logging.getLogger()
        text_a = references[0].replace("\\n", "\n")
        text_b = predictions[0].replace("\\n", "\n")

        if text_a == text_b:
            logger.info(f"predictions {predictions} and references {references} are exactly the same. Returning 100.")
            return 100

        prompt = self.prompt_template.replace("{TEXT_A}", text_a)
        prompt = prompt.replace("{TEXT_B}", text_b)

        logger.info("Calling LLM text_a vs text_b")
        gpt_response = self.call_llm(prompt, **kwargs).strip()
        ab_gpt_number = find_first_number(gpt_response)

        logger.info("Calling LLM text_b vs text_a")
        rev_prompt = self.prompt_template.replace("{TEXT_A}", text_b)
        rev_prompt = rev_prompt.replace("{TEXT_B}", text_a)

        rev_gpt_response = self.call_llm(rev_prompt, **kwargs).strip()
        ba_gpt_number = find_first_number(rev_gpt_response)

        gpt_numb = max(int(ab_gpt_number), int(ba_gpt_number))

        if gpt_numb is not None:
            return gpt_numb
        else:
            return 0

