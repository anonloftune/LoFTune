import argparse
import logging
import json
import tqdm
import numpy as np

from elasticsearch import Elasticsearch
import configparser
import os

from factscore.factscorer import FactScorer

class ESSearcher:
    def __init__(
        self,
        endpoint,
        es_user,
        es_pwd,
        index_name,
    ):
        self.es_client = Elasticsearch(endpoint, basic_auth=(es_user, es_pwd))
        self.index_name = index_name

    def get_passages(self, topic, text, k=10):
        resp = self.es_client.search(
            index=self.index_name, query={"match": {"abstract": text}}
        )
        passages = [{"title": "", "text": hit["_source"]["abstract"]} for hit in resp["hits"]["hits"][:k]]
        return passages

    def save_cache(self):
        print("save_cache function activated. ESSearcher does not have it. Sorry")


def truthfulness_score(claims_path=None, dataset_output_path=None, fs=None, searcher=None, args = None):
    with open(claims_path) as json_file:
        entity_question_answers_claims = json.load(json_file)

    entity_question_answers_claims_score = {}
    score_count = 0
    for entity, question_answers_claims in tqdm.tqdm(
        entity_question_answers_claims.items()
    ):
        entity_question_answers_claims_score[entity] = {}
        for question, answers_claims in question_answers_claims.items():
            entity_question_answers_claims_score[entity][question] = {}
            for answer, claims in answers_claims.items():
                generations, topics, atomic_facts = [], [], []
                if len(claims) > 0:
                    generations.append(answer)
                    topics.append("")
                    if type(claims)==dict:
                        claims = list(claims["claims_questions_answers_scores"].keys())
                    atomic_facts.append([claim.strip() for claim in claims])
                    out = fs.get_score(topics=topics,
                       generations=generations,
                       gamma=args.gamma,
                       atomic_facts=atomic_facts if args.use_atomic_facts else None,
                       knowledge_source=args.knowledge_source,
                       searcher=searcher,
                       verbose=args.verbose)
                    score_count+=1
                    if score_count % 100 == 0:
                        fs.save_cache()

                    entity_question_answers_claims_score[entity][
                        question
                    ][answer] = {
                        "score": out["score"] if not np.isnan(out["score"]) else None,
                        "out": {key:(value if type(value)==list or not np.isnan(value) else None) for key,value in out.items()},
                    }
    fs.save_cache()

    with open(dataset_output_path, "w") as outfile:
        json.dump(entity_question_answers_claims_score, outfile)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--claims_path", type=str)
    parser.add_argument('--model_name',
                        type=str,
                        default="retrieval+llama")
    parser.add_argument('--gamma',
                        type=int,
                        default=10,
                        help="hyperparameter for length penalty")

    parser.add_argument('--openai_key',
                        type=str,
                        default="api.key")
    parser.add_argument('--data_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--cache_dir',
                        type=str,
                        default=".cache/factscore/")
    parser.add_argument('--knowledge_source',
                        type=str,
                        default=None)

    parser.add_argument('--cost_estimate',
                        type=str,
                        default="consider_cache",
                        choices=["consider_cache", "ignore_cache"])
    parser.add_argument('--abstain_detection_type',
                        type=str,
                        default=None,
                        choices=["perplexity_ai", "generic", "none"])
    parser.add_argument('--use_atomic_facts',
                        action="store_true")
    parser.add_argument('--verbose',
                        action="store_true",
                        help="for printing out the progress bar")
    parser.add_argument('--print_rate_limit_error',
                        action="store_true",
                        help="for printing out rate limit error when using OpenAI keys")

    parser.add_argument("--dataset_output_path", type=str)
    parser.add_argument('--es_config', type=str, default="service_config.ini")

    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR if args.print_rate_limit_error else logging.CRITICAL)

    fs = FactScorer(model_name=args.model_name,
                    data_dir=args.data_dir,
                    cache_dir=args.cache_dir,
                    openai_key=args.openai_key,
                    cost_estimate=args.cost_estimate,
                    abstain_detection_type=args.abstain_detection_type)

    cfg = configparser.ConfigParser()
    searcher_cfg = cfg.read(args.es_config)
    searcher = ESSearcher(**cfg["elastic"])

    truthfulness_score(claims_path=args.claims_path, dataset_output_path=args.dataset_output_path, fs=fs, searcher=searcher, args = args)
