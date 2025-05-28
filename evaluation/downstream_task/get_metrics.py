# import evaluate
import argparse
from datasets import load_from_disk, load_dataset
from factory import EvalMetricsFactory
import logging
from tqdm import tqdm
from basic_utils import decorate_str_with_date
import numpy as np
import os

def apply_score_fn(example,score_fn,score_kwargs):
    fn_results = []
    for reference in example["output"]:
        fn_res = score_fn.compute(
            predictions=[example["answer"]],
            references=[reference],
            **score_kwargs,
        )
        fn_results.append(fn_res)
        if fn_res==100:
            break
    example[score_fn.name] = max(fn_results)
    return example


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--question_dataset_path", type=str)
    parser.add_argument("--predictions_path", type=str)
    parser.add_argument('--openai_key', type=str, default="./api.key", help="File containing OpenAI API Key. Default: ./api.key")

    args = parser.parse_args()

    assert os.path.exists(args.openai_key), f"Please place your OpenAI GPT Key in {args.openai_key}."
    with open(args.openai_key, 'r') as f:
        api_key = f.readline()

    log_filename = f"{decorate_str_with_date('logs')}.txt"
    
    logging.basicConfig(
        filename=log_filename,
        filemode="a",
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)-8s.%(msecs)03d  %(name)s:%(lineno)-3s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger()
    logger.info("=================== InsuranceQA EVALUATION ===================")
    tqdm.pandas()
    
    # rouge = evaluate.load('rouge')
    metrics_factory = EvalMetricsFactory()
    # metric = metrics_factory.load("MistralEval", api_url="https://elmilab.expertcustomers.ai/elmi/generate", cache_dir=".cache/", device="cpu")
    metric = metrics_factory.load("GPTEval", api_url="https://api.openai.com/v1/chat/completions", cache_dir=".cache_GPT4omini/", api_key=api_key)
    # max_length = 256
    # use_beam_search = True
    # temperature = 0.1
    # num_beams = 3
    # top_p = 0.75
    # top_k = 40
    # request_params = {
    #     "max_length": max_length,
    #     "use_beam_search": use_beam_search,
    #     "temperature": temperature,
    #     "num_beams": num_beams,
    #     "top_p": top_p,
    #     "top_k": top_k
    # }
    request_params = {}
    
    questions_ds = load_from_disk(args.question_dataset_path)
    prediction_ds = load_dataset("json", data_files= args.predictions_path)["train"]

    references = questions_ds["output"]
    predictions = prediction_ds["answer"]
    prediction_ds = prediction_ds.add_column("output",questions_ds["output"])

    prediction_ds = prediction_ds.to_pandas()
    prediction_ds = prediction_ds.progress_apply(lambda x: apply_score_fn(x,metric,request_params), axis=1)
    _, pred_dataset_name = os.path.split(args.predictions_path)
    pred_dataset_name_output =  "eval_results_gpt4o_mini/" + pred_dataset_name.replace(".jsonl", ".csv")
    prediction_ds.to_csv(pred_dataset_name_output, index=False)  
    # results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)
    results = {
        # **results,
        "gpt-check-long":np.nanmean(prediction_ds["gpt-check-long"])
    }
    logger.info(results)
   