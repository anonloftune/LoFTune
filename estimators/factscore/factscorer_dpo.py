import argparse
import logging
import json
import yaml
import tqdm
import numpy as np

from factscore.factscorer import FactScorer

entity_articles_mapping = None

def entity_to_topic(entity):
    topic = None
    entry = entity_articles_mapping[entity]
    if type(entry) == str:
        topic = entry
    elif type(entry) == dict:
        synonyms = list(entry["synonyms"].values()) if "synonyms" in entry else []
        hypernyms = list(entry["hypernyms"].values()) if "hypernyms" in entry else []
        topic = synonyms+hypernyms

    return topic


def truthfulness_score(claims_path=None, dataset_output_path=None, fs=None, args = None):
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
                topics, generations, atomic_facts = [], [], []
                topic = entity_to_topic(entity)
                if topic is not None and len(claims)>0:
                    topics.append(topic)
                    generations.append(answer)
                    if type(claims)==dict:
                        claims = list(claims["claims_questions_answers_scores"].keys())
                    atomic_facts.append([claim.strip() for claim in claims])
                
                out = fs.get_score(topics=topics,
                       generations=generations,
                       gamma=args.gamma,
                       atomic_facts=atomic_facts if args.use_atomic_facts else None,
                       knowledge_source=args.knowledge_source,
                       verbose=args.verbose)
                # decisions = out["decisions"]
                # discarted_decisions = out["discarted_decisions"]
                # discarted_ration_threshold = 0.72
                # scores = []
                # for decisions_items, discarted_decisions_items in zip (decisions, discarted_decisions):
                #     if (len(discarted_decisions_items)+len(decisions_items))>0:
                #         discarted_ratio = len(discarted_decisions_items)/(len(discarted_decisions_items)+len(decisions_items))
                #         if (discarted_ratio <= discarted_ration_threshold):
                #             score = np.mean([d["is_supported"] for d in decisions_items])
                #             scores.append(score)
                # out["score_with_threshold"] = np.mean(scores)
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
    
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR if args.print_rate_limit_error else logging.CRITICAL)

    with open("insurance-en-new_distribution_train_dev-entities-synonyms-hypernyms.yml", "r") as file:
        entity_articles_mapping = yaml.safe_load(file)

    
    fs = FactScorer(model_name=args.model_name,
                    data_dir=args.data_dir,
                    # model_dir=args.model_dir,
                    cache_dir=args.cache_dir,
                    openai_key=args.openai_key,
                    cost_estimate=args.cost_estimate,
                    abstain_detection_type=args.abstain_detection_type)    

    truthfulness_score(claims_path=args.claims_path, dataset_output_path=args.dataset_output_path, fs=fs, args = args)
    
#     logging.critical("FActScore = %.1f%%" % (100*out["score"]))
#     if "init_score" in out:
#         logging.critical("FActScore w/o length penalty = %.1f%%" % (100*out["init_score"]))
#     logging.critical("Respond ratio = %.1f%%" % (100*out["respond_ratio"]))
#     logging.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))

#     with open(args.dataset_output_path, "w") as outfile:
#         json.dump(out, outfile)
