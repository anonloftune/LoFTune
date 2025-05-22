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
   pip install requests tqdm numpy sentence_transformers json_repair accelerate==0.28.0 datasets==2.18.0 peft==0.9.0 trl==0.7.11 transformers==4.38.2 bitsandbytes==0.43.0 torch==2.2.1 "numpy<2.0" tensorboard
   cd estimators
   mkdir .cache
   touch api.key # Here you need to copy your OpenAI API token
   python -m common.generate_questions_dataset --entities_path ../data/insurance/train_entities.txt \
   --dataset_output_path ../data/insurance/train_entities_questions.json \
   --prompt_path common/prompts/generate_questions_dataset_insurance_prompt.txt \
   --openai_key ./api.key

It is recommended to create another conda environment to use vllm. To run gated models such as Llama-2-7b-hf, you need to have access.

.. code:: bash

   conda create -n vllm python=3.9
   conda activate vllm
   pip install vllm==0.4.1 transformers==4.40.1 "numpy<2.0"
   huggingface-cli login # Only required for gated models like Llama-2-7b-hf
   python -m common.sample_model_vllm --question_dataset_path ../data/insurance/train_entities_questions.json \
   --model_name_or_path meta-llama/Llama-2-7b-hf \
   --output_path ../data/insurance/train_entities_questions_answers_Llama-2-7b-hf.json \
   --temperature 1.0 \
   --samples_per_prompt 5 \
   --prompt_path common/prompts/sample_model_insurance_few-shot_prompt.txt

Now we activate back the loftune conda environment to extract the claims from the responses.

.. code:: bash

   conda activate loftune
   python -m common.extract_claims --paragraphs_path ../data/insurance/train_entities_questions_answers_Llama-2-7b-hf.json \
   --dataset_output_path ../data/insurance/train_entities_questions_answers_claims_Llama-2-7b-hf.json \
   --prompt_path common/prompts/extract_claims_prompt.txt \
   --openai_key api.key


**Model confidence**

.. code:: bash

   python -m model_confidence.claims_to_questions --claims_path ../data/insurance/train_entities_questions_answers_claims_Llama-2-7b-hf.json \
   --dataset_output_path ../data/insurance/train_entities_questions_answers_claims_questions_Llama-2-7b-hf.json \
   --prompt_path model_confidence/prompts/claims_to_questions_insurance_prompt.txt \
   --openai_key api.key
   conda activate vllm
   python -m model_confidence.answer_questions_vllm --question_dataset_path ../data/insurance/train_entities_questions_answers_claims_questions_Llama-2-7b-hf.json \
   --model_name_or_path meta-llama/Llama-2-7b-hf \
   --dataset_output_path ../data/insurance/train_entities_questions_answers_claims_questions_answers_Llama-2-7b-hf.json \
   --prompt_path model_confidence/prompts/answer_questions_insurance_prompt.txt
   conda activate loftune
   python -m model_confidence.truthfulness_score --answers_dataset_path ../data/insurance/train_entities_questions_answers_claims_questions_answers_Llama-2-7b-hf.json \
   --dataset_output_path ../data/insurance/train_entities_questions_answers_claims_questions_answers_clustering_scores_Llama-2-7b-hf.json
   

**FactScore/Expanded FactScore**

.. code:: bash

   cd factscore/
   conda create -n factscore python=3.10
   conda activate factscore
   pip install -r requirements.txt
   pip install gdown
   mkdir -p .cache/factscore
   gdown https://drive.google.com/uc?id=1Qu4JHWjpUKhGPaAW5UHhS5RJ545CVy4I
   mv enwiki-20230401.db .cache/factscore/

If we want to get the Expanded Factscore, we have to set to "insurance-en-new_distribution_train_dev-entities-synonyms-hypernyms.yml" the entity_articles_mapping param.

.. code:: bash

   python factscorer_dpo.py \
   --claims_path ../../data/insurance/train_entities_questions_answers_claims_Llama-2-7b-hf.json \
   --model_name retrieval+ChatGPT \
   --cache_dir .cache/factscore \
   --gamma 0 \
   --openai_key ../api.key \
   --use_atomic_facts  \
   --entity_articles_mapping insurance-en-new_distribution_train_dev-entities-synonyms-hypernyms.yml \
   --dataset_output_path  ../../data/insurance/train_entities_questions_answers_claims_expanded-factscore_Llama-2-7b-hf.json

To get the factscore without the term expansion, we set to "insurance-en-new_distribution_train_dev-entities-no-expansion.yml" the entity_articles_mapping param.

.. code:: bash

   python factscorer_dpo.py \
   --claims_path ../../data/insurance/train_entities_questions_answers_claims_Llama-2-7b-hf.json \
   --model_name retrieval+ChatGPT \
   --cache_dir .cache/factscore \
   --gamma 0 \
   --openai_key ../api.key \
   --use_atomic_facts  \
   --entity_articles_mapping insurance-en-new_distribution_train_dev-entities-no-expansion.yml \
   --dataset_output_path  ../../data/insurance/train_entities_questions_answers_claims_factscore-no-expansion_Llama-2-7b-hf.json

**Judge-based**

From the "estimators" folder, we run:

.. code:: bash

   conda activate loftune
   python -m judge_based.judge_based --paragraphs_path ../data/insurance/train_entities_questions_answers_Llama-2-7b-hf.json \
   --dataset_output_path ../data/insurance/train_entities_questions_answers_judge-based_Llama-2-7b-hf.json \
   --openai_key ./api.key
   

SFT and preference dataset generation
~~~~~~~~~~~~~~~~~~~~~

.. code:: bash

   python -m common.prepare_sft_data --dataset_input_path ../data/insurance/train_entities_questions_answers_claims_questions_answers_clustering_scores_Llama-2-7b-hf.json \
   --dataset_output_path ../data/insurance/train_entities_questions_answers_Llama-2-7b-hf.jsonlines
   python -m common.generate_preferences_dataset --scores_dataset_path ../data/insurance/train_entities_questions_answers_claims_questions_answers_clustering_scores_Llama-2-7b-hf.json \
   --dataset_output_path ../data/insurance/train_entities_preferences_clustering_Llama-2-7b-hf.jsonlines \
   --chosen_threshold 0.0
   

Training
~~~~~~~~~~~~~~~~~~~~~
**Supervised Fine-Tuning (SFT)**

.. code:: bash

   cd training
   accelerate config

The default_config.yaml file in ~/.cache/huggingface/defaul_config.yaml

.. code:: yaml

   compute_environment: LOCAL_MACHINE
   debug: false
   distributed_type: 'NO'
   downcast_bf16: 'no'
   gpu_ids: all
   machine_rank: 0
   main_training_function: main
   mixed_precision: bf16
   num_machines: 1
   num_processes: 1
   rdzv_backend: static
   same_network: true
   tpu_env: []
   tpu_use_cluster: false
   tpu_use_sudo: false
   use_cpu: false

We run the SFT training, "max_steps" params has to be changed depending on the size of the SFT dataset, if size is for example 2730 examples, we divide 2730 by 8 (batch size) = 341,25 steps/epoch, and as an heuristic we train for 1.3 epochs so 341,25*1,3 = ~443:

.. code:: bash

   accelerate launch sft_llama2.py \
       --train_data_path="../data/insurance/train_entities_questions_answers_Llama-2-7b-hf.jsonlines" \
       --valid_data_path="../data/insurance/validation_entities_questions_answers_Llama-2-7b-hf.jsonlines" \
       --output_dir="insurance_m_5/sft" \
       --max_steps=443 \
       --logging_steps=10 \
       --save_steps=10 \
       --per_device_train_batch_size=2 \
       --per_device_eval_batch_size=64 \
       --gradient_accumulation_steps=4 \
       --gradient_checkpointing=False \
       --group_by_length=False \
       --learning_rate=1e-4 \
       --lr_scheduler_type="cosine" \
       --warmup_steps=100 \
       --weight_decay=0.05 \
       --optim="paged_adamw_32bit" \
       --bf16=True \
       --remove_unused_columns=True \
       --run_name="insurance_m_5_sft" \
       --report_to="tensorboard" \
       --eval_steps=10 \
       --evaluation_strategy="steps"

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
