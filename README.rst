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
   pip install requests
   cd estimators/common
   mkdir .cache
   touch api.key # Here you need to copy your OpenAI API token
   python generate_questions_dataset.py --entities_path ../../data/insurance/train_entities.txt \
   --dataset_output_path ../../data/insurance/train_entities_questions.json \
   --prompt_path prompts/generate_questions_dataset_insurance_prompt.txt \
   --openai_key ./api.key

   

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
