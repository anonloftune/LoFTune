import argparse
from datasets import load_dataset
import logging
from basic_utils import decorate_str_with_date
import os

from factool import Factool
factool_instance = Factool("gpt-4o-mini-2024-07-18") #gpt-4o-2024-08-06
import json

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_path", type=str)

    args = parser.parse_args()

    log_filename = f"{decorate_str_with_date('logs')}.txt"
    
    logging.basicConfig(
        filename=log_filename,
        filemode="a",
        level=logging.INFO,
        format="[%(levelname)s] %(asctime)-8s.%(msecs)03d  %(name)s:%(lineno)-3s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    logger = logging.getLogger()
    logger.info("=================== Factool EVALUATION ===================")
    
    prediction_ds = load_dataset("json", data_files= args.predictions_path)["train"]
    prediction_ds = prediction_ds.rename_column("input", "prompt")
    prediction_ds = prediction_ds.rename_column("answer", "response")
    new_column = ["kbqa"] * len(prediction_ds)
    prediction_ds = prediction_ds.add_column("category", new_column)

    inputs = prediction_ds.to_list()

    response_list = factool_instance.run(inputs)
    _, pred_dataset_name = os.path.split(args.predictions_path)
    pred_dataset_name_output =  "eval_results_gpt4o_mini/" + pred_dataset_name.replace(".jsonl", ".json")
    with open(pred_dataset_name_output, 'w') as f:
        json.dump(response_list, f)
   