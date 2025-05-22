===============================
Towards Factual Large Language Models in Low-Resource Domains
===============================

|PyPI pyversions|

Source code and datasets for the paper Towards Factual Large Language Models in Low-Resource Domains

Reproducibility
---------------

Factuality estimators
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   git clone https://github.com/anonloftune/LoFTune.git
   cd LoFTune/
   conda create -n loftune python=3.12
   conda activate loftune
   pip install requests tqdm
   cd estimators/common
   mkdir .cache
   touch api.key # Here you need to copy your OpenAI API token
   python generate_questions_dataset.py --entities_path ../../data/insurance/train_entities.txt \
   --dataset_output_path ../../data/insurance/train_entities_questions.json \
   --prompt_path prompts/generate_questions_dataset_insurance_prompt.txt \
   --openai_key ./api.key

It is recommended to create another conda environment to use vllm. To run gated models such as Llama-2-7b-hf, you need to have access.

.. code:: bash

   conda create -n vllm python=3.9
   conda activate vllm
   pip install vllm==0.4.1 transformers==4.40.1 "numpy<2.0"
   huggingface-cli login # Only required for gated models like Llama-2-7b-hf
   python sample_model_vllm.py --question_dataset_path ../../data/insurance/train_entities_questions.json \
   --model_name_or_path meta-llama/Llama-2-7b-hf \
   --output_path ../../data/insurance/train_entities_questions_answers_Llama-2-7b-hf.json \
   --temperature 1.0 \
   --samples_per_prompt 5 \
   --prompt_path prompts/sample_model_insurance_few-shot_prompt.txt 

Now we activate back the loftune conda environment to extract the claims from the responses.

.. code:: bash

   conda activate loftune
   python extract_claims.py --paragraphs_path ../../data/insurance/train_entities_questions_answers_Llama-2-7b-hf.json \
   --dataset_output_path ../../data/insurance/train_entities_questions_answers_claims_Llama-2-7b-hf.json \
   --prompt_path prompts/extract_claims_prompt.txt \
   --openai_key api.key
   

**Model confidence**

**FactScore/Expanded FactScore**

**Judge-based**


Training
~~~~~~~~~~~~~~~~~~~~~
**Supervised Fine-Tuning (SFT)**

**Direct Preference Optimization (DPO)**


Evaluation
~~~~~~~~~~~~~~~~~~~~~
**Factscore (Insurance)**

**Factscore (Biomedicine)**

**Downstream tasks: InsuranceQA and CovidQA**

**SelfAware**

**FreshQA**

**FacTool-QA**

**FactScore-Bio**


.. |PyPI pyversions| image:: https://badgen.net/pypi/python/black
   :target: https://www.python.org/
