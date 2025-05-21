import pickle
import os
import time

class LM(object):

    def __init__(self, cache_file):
        self.cache_file = cache_file
        self.cache_dict = self.load_cache()
        self.model = None
        self.add_n = 0

    def load_model(self):
        # load the model and put it as self.model
        raise NotImplementedError()

    def generate(self, prompt, sample_idx=0, max_sequence_length=2048, max_output_length=128):
        prompt = prompt.strip() # it's important not to end with a whitespace
        cache_key = f"{prompt}_{sample_idx}"

        if cache_key in self.cache_dict:
            return self.cache_dict[cache_key]

        if self.model is None:
            self.load_model()

        if prompt.endswith(" True or False?\nAnswer:"):
            generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=1)
        else:
            generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=max_output_length)

        self.cache_dict[cache_key] = generated
        self.add_n += 1
        return generated

    def generate_batch(self, prompts, sample_idx=0, max_sequence_length=2048, max_output_length=128):
        outputs = [""] * len(prompts)
        to_generate_prompts = [None] * len(prompts)
        to_generate_prompts_filtered = []
        for i, prompt in enumerate(prompts):
            prompt = prompt.strip() # it's important not to end with a whitespace
            cache_key = f"{prompt}_{sample_idx}"
    
            if cache_key in self.cache_dict:
                outputs[i]=self.cache_dict[cache_key]
                continue

            to_generate_prompts[i] = prompt
            to_generate_prompts_filtered.append(prompt)

        if len(to_generate_prompts_filtered)==0: ## All prompts were cached
            return outputs

        if self.model is None:
            self.load_model()

        # if prompt.endswith(" True or False?\nAnswer:"):
        #     generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=1)
        # else:
        #     generated = self._generate(prompt, max_sequence_length=max_sequence_length, max_output_length=max_output_length)

        generated = self._generate(to_generate_prompts_filtered, max_sequence_length=max_sequence_length, max_output_length=max_output_length)

        j=0
        for i, prompt in enumerate(to_generate_prompts):
            if prompt==None:
                continue
            outputs[i] = (generated[0][j],None)
            prompt = prompt.strip() # it's important not to end with a whitespace
            cache_key = f"{prompt}_{sample_idx}"
            self.cache_dict[cache_key] = outputs[i]
            self.add_n += 1
            j+=1
            
        return outputs

    def save_cache(self):
        if self.add_n == 0:
            return

        # load the latest cache first, since if there were other processes running in parallel, cache might have been updated
        for k, v in self.load_cache().items():
            self.cache_dict[k] = v

        with open(self.cache_file, "wb") as f:
            pickle.dump(self.cache_dict, f)

    def load_cache(self, allow_retry=True):
        if os.path.exists(self.cache_file):
            while True:
                try:
                    with open(self.cache_file, "rb") as f:
                        cache = pickle.load(f)
                    break
                except Exception:
                    if not allow_retry:
                        assert False
                    print ("Pickle Error: Retry in 5sec...")
                    time.sleep(5)        
        else:
            cache = {}
        return cache



