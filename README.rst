===============================
Towards Factual Large Language Models in Low-Resource Domains
===============================

|PyPI pyversions|

Content
---------------
This repository contains **code** and **datasets** for the paper titled *Towards Factual Large Language Models in Low-Resource Domains*. It is organized as follows:

- `data <data>`_: Contains the train/validation/test entity splits for the insurance and health domain, and the test questions.
- `estimators <estimators>`_: Contains the code for running the different factuality estimators: model confidence, (expanded) FactScore, and the judge-based estimator.
- `evaluation <evaluation>`_: Code and data for running the different evaluations: FactScore on insurance and health domain, downstream tasks (InsuranceQA and CovidQA), and the general domain evaluation: SelfAware, FreshQA, Factool-QA, and FactScore biographies.
- `training <training>`_: Scripts to run the traininig of the Supervised Fine-Tuning (SFT) and the Direct Preference Optimization (DPO) models  for both the Llama and Pythia models.


Reproducibility
---------------

Factuality estimators
~~~~~~~~~~~~~~~~~~~~~

In this section, we explain how to generate a dataset of question-answer pairs along with their factuality scores. The scores can be computed using one of three estimators: model confidence, (expanded) FactScore, or a judge-based estimator.
Some steps are common across all estimators, such as question generation and question sampling. Additionally, both the model confidence and FactScore methods share the claim extraction step.
The examples provided use 5 samples per prompt. To create, for example, a dataset with 10 samples per prompt from the 5 samples per prompt version, you can generate another set of 5 samples and merge it with the existing dataset.

First we create a conda environment with the required libraries, next we can generate the questions from the entities.

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

For the model confidence estimator, we take the extracted claims, generate questions from them, sample the model 20 times, cluster the answers, and we use the size of the largest cluster to estimate the model's confidence.
 
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

For the factscore we will create another conda environment, install the requirements, and download the reference dataset.

.. code:: bash

   cd factscore/
   conda create -n factscore python=3.10
   conda activate factscore
   pip install -r requirements.txt
   pip install gdown
   mkdir -p .cache/factscore
   gdown https://drive.google.com/uc?id=1Qu4JHWjpUKhGPaAW5UHhS5RJ545CVy4I
   mv enwiki-20230401.db .cache/factscore/

The expansion of terms is done with the "wikipedia_search.py" script, we can run it with the following commands, from the "estimators" folder:

.. code:: bash

   conda activate loftune
   mkdir .cache_GPT4o-mini-search
   python -m factscore.wikipedia_search --openai_key api.key \
   --prompt_path factscore/prompts/wikipedia_search-multiple-articles_prompt.txt \
   --entity_definitions factscore/insurance-en-entities-definitions.yml \
   --output_path factscore/insurance-en-entities-mapping.yml

This will create the file "insurance-en-entities-mapping.yml", where for each entity, there will be an wikipedia article used for factuality evaluation.

If we want to get the Expanded Factscore, we have to run the following from the "estimators/factscore" folder, and set to "insurance-en-new_distribution_train_dev-entities-synonyms-hypernyms.yml" the entity_articles_mapping param:

.. code:: bash

   conda activate factscore
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



For the health domain you must use the factscorer_es_dpo.py script. Before run it, don't forget to fill the fields in service_config.ini file. The verification dataset from Pubmed used in our experiments will be uploaded in the near future to be indexed in your own elasticsearch instances.

.. code:: bash

   python factscorer_es_dpo.py \
   --claims_path ../../data/biomedicine/train_entities_questions_answers_claims_Llama-2-7b-hf.json \
   --model_name retrieval+ChatGPT \
   --knowledge_source elastic \
   --cache_dir .cache/factscore \
   --gamma 0 \
   --openai_key ../api.key \
   --use_atomic_facts  \
   --es_config service_config.ini
   --dataset_output_path  ../../data/biomedicine/train_entities_questions_answers_claims_factscore_Llama-2-7b-hf.json


**Judge-based**

From the "estimators" folder, we run:

.. code:: bash

   conda activate loftune
   python -m judge_based.judge_based --paragraphs_path ../data/insurance/train_entities_questions_answers_Llama-2-7b-hf.json \
   --dataset_output_path ../data/insurance/train_entities_questions_answers_judge-based_Llama-2-7b-hf.json \
   --openai_key ./api.key
   

SFT and preference dataset generation
~~~~~~~~~~~~~~~~~~~~~

The SFT and DPO datasets can be generated with the following commands:

.. code:: bash

   python -m common.prepare_sft_data --dataset_input_path ../data/insurance/train_entities_questions_answers_claims_questions_answers_clustering_scores_Llama-2-7b-hf.json \
   --dataset_output_path ../data/insurance/train_entities_questions_answers_Llama-2-7b-hf.jsonlines
   python -m common.generate_preferences_dataset --scores_dataset_path ../data/insurance/train_entities_questions_answers_claims_questions_answers_clustering_scores_Llama-2-7b-hf.json \
   --dataset_output_path ../data/insurance/train_entities_preferences_clustering_Llama-2-7b-hf.jsonlines \
   --chosen_threshold 0.0
   

Training
~~~~~~~~~~~~~~~~~~~~~

For the training of SFT and DPO models, we used accelerate, we show our configuration file as reference:

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

**Supervised Fine-Tuning (SFT)**

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

The merged model will be placed at insurance_m_5/sft/final_merged_checkpoint.

**Direct Preference Optimization (DPO)**

The "max_steps" param has to be adjusted according to the preference dataset size, for example if we have a size of 5195 examples, we divide by 64 (batch size) = 81,17, and multiply by 20 (we train for up to 20 epochs but apply early stopping) = ~1623. The "eval_steps" and "save_steps" have to be also changed according to the size, in this case we eval the trainig with the validation set every half epoch, so in our example we divide 81,17 by 2 = ~40

.. code:: bash

   accelerate launch dpo_llama2.py \
      --train_data_path="../data/insurance/train_entities_preferences_clustering_Llama-2-7b-hf.jsonlines" \
      --valid_data_path="../data/insurance/validation_entities_preferences_clustering_Llama-2-7b-hf.jsonlines" \
      --model_name_or_path="insurance_m_5/sft/final_merged_checkpoint" \
      --output_dir="insurance_m_5/factune_mc" \
      --lr_scheduler_type="cosine" \
      --warmup_steps=150 \
      --gradient_accumulation_steps=16 \
      --max_steps=1623 \
      --lora_r=8 \
      --lora_alpha=16 \
      --learning_rate=0.00001 \
      --report_to="tensorboard" \
      --model_dtype="bfloat16" \
      --per_device_eval_batch_size=32 \
      --eval_steps=40 \
      --save_steps=40 \
      --early_stopping=True \
      --early_stopping_patience=4

The LoRA weights will be found in our case at "insurance_m_5/factune_mc/". We can merge the weights to the SFT model with:

.. code:: bash

   cp insurance_m_5/sft/*token* insurance_m_5/sft/final_merged_checkpoint/
   python merge_peft_adapter.py --adapter_model_name insurance_m_5/factune_mc/ --base_model_name insurance_m_5/sft/final_merged_checkpoint/ --output_name insurance_m_5/factune_mc_merged
   

Evaluation
~~~~~~~~~~~~~~~~~~~~~

In this section we show how to evaluate the LLMs on diferent tasks: FactScore in insurance and health domain, downstream tasks and in general domain.

**Factscore (Insurance)**

From the "estimators" folder:

.. code:: bash

   conda activate vllm
   python -m common.sample_model_vllm --question_dataset_path ../data/insurance/test_entities_questions_dataset.json \
   --model_name_or_path ../training/insurance_m_5/factune_mc_merged \
   --output_path ../data/insurance/test_entities_questions_answers_insurance-m-5-factune-mc.json \
   --temperature 0.6 \
   --samples_per_prompt 6 \
   --prompt_path common/prompts/sample_model_zero-shot_prompt.txt # Here we use the zero-shot prompt
   conda activate loftune
   python -m common.extract_claims --paragraphs_path ../data/insurance/test_entities_questions_answers_insurance-m-5-factune-mc.json \
   --dataset_output_path ../data/insurance/test_entities_questions_answers_claims_insurance-m-5-factune-mc.json \
   --prompt_path common/prompts/extract_claims_prompt.txt \
   --openai_key api.key
   cd factscore
   conda activate factscore
   python factscorer.py \
   --claims_path ../../data/insurance/test_entities_questions_answers_claims_insurance-m-5-factune-mc.json \
   --model_name retrieval+ChatGPT \
   --cache_dir .cache/factscore \
   --gamma 0 \
   --openai_key ../api.key \
   --use_atomic_facts  \
   --dataset_output_path  ../../data/insurance/test_entities_questions_answers_claims_factscore_insurance-m-5-factune-mc.json

If we open the "test_entities_questions_answers_claims_factscore_insurance-m-5-factune-mc.json" file, we can see the obtained factscore in the insurance domain, and the average supported, refute, and not enough information per response.

**Factscore (Biomedicine)**

From the "estimators" folder:

.. code:: bash

   conda activate vllm
   python -m common.sample_model_vllm --question_dataset_path ../data/biomedicine/covid_entities_subset_test_questions.json \
   --model_name_or_path ../training/insurance_m_5/factune_mc_merged/ \
   --output_path ../data/biomedicine/covid_entities_subset_test_questions_answers_insurance-m-5-factune-mc.json \
   --temperature 0.6 \
   --samples_per_prompt 6 \
   --prompt_path common/prompts/sample_model_zero-shot_prompt.txt # Here we use the zero-shot prompt
   conda activate loftune
   python -m common.extract_claims --paragraphs_path ../data/biomedicine/covid_entities_subset_test_questions_answers_insurance-m-5-factune-mc.json \
   --dataset_output_path ../data/biomedicine/covid_entities_subset_test_questions_answers_claims_insurance-m-5-factune-mc.json \
   --prompt_path common/prompts/extract_claims_prompt.txt \
   --openai_key api.key
   cd factscore
   conda activate factscore
   python factscorer_es.py \
   --claims_path ../../data/biomedicine/covid_entities_subset_test_questions_answers_claims_insurance-m-5-factune-mc.json \
   --model_name retrieval+ChatGPT \
   --knowledge_source elastic \
   --cache_dir .cache/factscore \
   --gamma 0 \
   --openai_key ../api.key \
   --use_atomic_facts  \
   --es_config service_config.ini
   --dataset_output_path  ../../data/biomedicine/covid_entities_subset_test_questions_answers_claims_factscore_insurance-m-5-factune-mc.json

If we open the "covid_entities_subset_test_questions_answers_claims_factscore_insurance-m-5-factune-mc.json" file, we can see the obtained factscore in the biomedicine domain, and the average supported, refute, and not enough information per response.

**Downstream tasks: InsuranceQA and CovidQA**


From the folder "evaluation/downstream_task", we run:

.. code :: bash

   conda activate vllm
   python sample_model_vllm.py --question_dataset_path ./insuranceQA/ \
   --model_name_or_path ../../training/insurance_m_5/factune_mc_merged/ \
   --output_path insuranceQA_insurance-m-5-factune-mc.jsonl \
   --temperature 0.6  \
   --prompt_path ../../estimators/common/prompts/sample_model_zero-shot_prompt.txt
   conda activate loftune
   mkdir eval_results_gpt4o_mini
   python get_metrics.py --question_dataset_path ./insuranceQA \
   --predictions_path insuranceQA_insurance-m-5-factune-mc.jsonlines \
   --openai_key ../../estimators/api.key

If we open the most recent "logs_*.txt" file, at the end of the file, we will see the "gpt-check-long" metric, which corresponds to the similaity score according to GPT 4o-mini.

To evaluate on the CovidQA dataset, we run:

.. code :: bash

   conda activate vllm
   python sample_model_vllm.py --question_dataset_path covidqa_bio/ \
   --model_name_or_path ../../training/insurance_m_5/factune_mc_merged/ \
   --output_path covidqa-bio_insurance-m-5-factune-mc.jsonl \
   --temperature 0.6 \
   --prompt_path ../../estimators/common/prompts/sample_model_zero-shot_prompt.txt
   conda activate loftune
   python get_metrics.py --question_dataset_path ./covidqa_bio/ \
   --predictions_path covidqa-bio_insurance-m-5-factune-mc.jsonl \
   --openai_key ../../estimators/api.key

**SelfAware**

From the "evaluation/selfAware/code" folder:

.. code :: bash

   conda create -n selfaware python=3.8
   conda activate selfaware
   pip install -r requirements.txt
   python run_model.py --input-form Direct --model-name insurance-m-5-factune-mc --temperature 0.7
   python eval_model.py --filename insurance-m-5-factune-mc/Direct_insurance-m-5-factune-mc_T_0.7.jsonl --threshold 0.75 --model princeton-nlp/sup-simcse-roberta-large

If you want to evaluate a new model, you have to modify the "run_model.py" script, first by adding a model name to the variable "choices" (line 46), adding the model name to "llama_list" (line 141), and adding a new entry to "model_dict" (line 142) where the key is the model name, and the value is the path to the model.

**FreshQA**

.. code :: bash

   conda activate vllm
   python sample_model_vllm.py --question_dataset_path data/FreshQA_v03172025\ -\ freshqa.csv \
   --model_name_or_path ../../training/insurance_m_5/factune_mc_merged/ \
   --output_path FreshQA_v03172025_insurance-m-5-factune-mc.csv \
   --temperature 0.6 \
   --prompt_path ../../estimators/common/prompts/sample_model_zero-shot_prompt.txt

Next steps are in the `FreshLLMs Github repository <https://github.com/freshllms/freshqa?tab=readme-ov-file#automatic-evaluation>`_, we used the "Relaxed" evaluation mode and "gpt-4o-mini-2024-07-18" as model_name in the "fresheval_relaxed.ipynb" notebook.

**FacTool-QA**

For this evaluation, appart from the OpenAI API token, we need a Sperper API token. You can generate one with 2,500 free queries.

From the "evaluation/factool" folder, we run:

.. code :: bash

   conda activate vllm
   python sample_model_vllm.py --question_dataset_path data/knowledge_qa/knowledge_qa.jsonl \
   --model_name_or_path ../../training/insurance_m_5/factune_mc_merged/ \
   --output_path knowledge-qa_insurance-m-5-factune-mc.jsonl \
   --temperature 0.6 \
   --prompt_path ../../estimators/common/prompts/sample_model_zero-shot_prompt.txt
   mkdir eval_results_gpt4o_mini
   conda create -n factool python=3.9
   conda activate factool
   pip install factool datasets httpx==0.27.2
   export OPENAI_API_KEY= # Here you need to put your OpenAI API Key
   export SERPER_API_KEY= # Here you need to put your Serper API Key
   export SCRAPER_API_KEY= # Here you need to put your Scraper API Key, although it is not used
   python get_metrics.py --predictions_path knowledge-qa_insurance-m-5-factune-mc.jsonl
   
**FactScore-Bio**

From the "evaluation/factscore_bio" folder, we run:

.. code:: bash

   conda activate vllm
   python sample_model_vllm.py --entities_dataset_path data/unlabeled/prompt_entities.txt \
   --model_name_or_path ../../training/insurance_m_5/factune_mc_merged/ \
   --output_path data/unlabeled/insurance-m-5-factune-mc.jsonl \
   --temperature 0.6 \
   --prompt_path ../../estimators/common/prompts/sample_model_zero-shot_prompt.txt
   conda create -n factscore_bio python=3.10
   conda activate factscore_bio
   pip install .
   mkdir -p .cache/factscore
   cd .cache/factscore/
   ln -s ../../../../estimators/factscore/.cache/factscore/enwiki-20230401.db
   cd -
   python -m spacy download en_core_web_sm
   python -m factscore.download_data
   python -m factscore.factscorer --input_path data/unlabeled/insurance-m-5-factune-mc.jsonl \
   --model_name retrieval+ChatGPT \
   --openai_key ../../estimators/api.key


Datasets and Models
---------------

All the training datasets and models used in this research are available in our `Huggingface profile <https://huggingface.co/anonloftune>`_. Insurance models are provided as PEFT models. We recommend merging the LoRA weights into the corresponding base model before performing any evaluation.


.. |PyPI pyversions| image:: https://badgen.net/pypi/python/black
   :target: https://www.python.org/
