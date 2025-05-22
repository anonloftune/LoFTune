# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import math
import time
import json
import numpy as np
import torch
from tqdm import tqdm
from collections import defaultdict

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

from factscore.utils import convert_model_to_int8_on_gpu
from factscore.lm import LM
# from vllm import LLM, SamplingParams


class CLM(LM):
    def __init__(self, model_name, model_dir=None, cache_file=None):
        self.model_name = model_name
        # self.model_dir = model_dir
        if cache_file:
            super().__init__(cache_file)

    def load_model(self):
        # self.model = AutoModelForCausalLM.from_pretrained(self.model_name,
        #                                                   torch_dtype = torch.float16,
        #                                                   # quantization_config = {"load_in_4bit": True},
        #                                                   #low_cpu_mem_usage = True
        #                                                   device_map="auto"
        #                                                  )
        # self.model = convert_model_to_int8_on_gpu(self.model, device='cuda')
        self.model = LLM(model=self.model_name,dtype = "float16", enable_prefix_caching=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        # self.tokenizer = self.model.tokenizer

    def _generate(self, prompts, max_sequence_length=2048, max_output_length=128,
                  end_if_newline=False, end_if_second_newline=False, verbose=False):
        is_single = type(prompts)==str
        if is_single:
            messages = [{"role": "user", "content": prompts},]
            
            prompts = [self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=True
            )]

            # print(prompts[0])

        else:
            prompts = [self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt},], 
                    tokenize=False, 
                    add_generation_prompt=True
            ) for prompt in prompts] 

            # prompts = [prompts]

        sampling_params = SamplingParams(
            temperature=0.2,
            top_p=0.1,
            top_k=50,
            max_tokens=max_output_length,
            seed=42
        )
        outputs = self.model.generate(prompts, sampling_params, use_tqdm=False)
        scores = [None for output in outputs]
        generations = [output.outputs[0].text for output in outputs]
        # input_ids = self.tokenizer(prompts).input_ids
        # if verbose:
        #     input_ids = tqdm(input_ids)

        # generations = []
        # scores = []
        # for curr_input_ids in input_ids:
        #     if len(curr_input_ids) > max_sequence_length - max_output_length:
        #         curr_input_ids = curr_input_ids[-(max_sequence_length - max_output_length):]
        #     curr_input_ids = torch.LongTensor([curr_input_ids]).cuda()
        #     gen_outputs = self.model.generate(
        #         curr_input_ids,
        #         max_length=curr_input_ids.shape[1]+max_output_length,
        #         return_dict_in_generate=True,
        #         output_scores=True,
        #         pad_token_id=self.tokenizer.eos_token_id 
        #     )
        #     gen_tokens = gen_outputs["sequences"]
        #     # saving the logits for the very first token
        #     gen_scores = gen_outputs["scores"][0][0].detach().cpu().numpy()
        #     gen = self.tokenizer.decode(gen_tokens[0, curr_input_ids.shape[-1]:])

        #     if end_if_newline:
        #         gen = gen.split("\n")[0].strip()
        #     elif end_if_second_newline:
        #         gen = "\n".join(gen.split("\n")[:2]).strip()

        #     if verbose and len(generations)==0:
        #         print ("Input:", prompts[0])
        #         print ("Prediction:", gen)

        #     if self.model_name.startswith("llama-sni"):
        #         gen = gen.split("</s>")[0]
                
        #     generations.append(gen)
        #     scores.append(gen_scores)

        assert len(generations)==len(prompts)==len(scores)
        if is_single:
            # return generations[0], scores[0]
            return generations[0], None

        return generations, None
        # return generations, scores

