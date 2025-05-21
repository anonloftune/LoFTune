import argparse
import logging
import json
import yaml
import tqdm
import copy

import numpy as np
import pandas as pd

from factscore.factscorer import FactScorer

entity_articles_mapping = None
articles_entity_mapping = None

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
    

def prepare_atomic_facts(claims_path=None,n_samples=None):
    with open(claims_path) as json_file:
        entity_question_answers_claims = json.load(json_file)

    entity_question_answers_claims_questions = {}
    topics, generations, atomic_facts = [], [], []
    tot = 0
    
    for entity, question_answers_claims in tqdm.tqdm(
        entity_question_answers_claims.items()
    ):
        # assert len(question_answers_claims)==6
        entity_question_answers_claims_questions[entity] = {}
        for question, answers_claims in question_answers_claims.items():
            entity_question_answers_claims_questions[entity][question] = {}
            for answer, claims in answers_claims.items():
                tot += 1
                topic = entity_to_topic(entity)
                # if topic is not None and len(claims)>0:
                topics.append(topic)
                generations.append(answer)
                atomic_facts.append([claim.strip() for claim in claims])
                if n_samples is not None and tot==n_samples:
                    return topics, generations, atomic_facts 
            # if len(answers_claims)!=6:
            #     for i in range(6-len(answers_claims)):
            #         topics.append(None)
            #         generations.append(None)
            #         atomic_facts.append([])
    return topics, generations, atomic_facts 


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
    
    parser.add_argument('--n_samples',
                        type=int,
                        default=None)
    
    args = parser.parse_args()

    logging.basicConfig(format='%(asctime)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.ERROR if args.print_rate_limit_error else logging.CRITICAL)

    with open("insurance-en-new_distribution_test-entities-synonyms-hypernyms.yml", "r") as file: # insurance-en-test-entities-synonyms-hypernyms.yml
        entity_articles_mapping = yaml.safe_load(file)

    articles_entity_mapping = {}

    for k, v in entity_articles_mapping.items():
        if type(v) == str:
            topic = v
        elif type(v) == dict:
            synonyms = list(v["synonyms"].values()) if "synonyms" in v else []
            hypernyms = list(v["hypernyms"].values()) if "hypernyms" in v else []
            topic = synonyms + hypernyms
        articles_entity_mapping[str(topic)] = k
    
    fs = FactScorer(model_name=args.model_name,
                    data_dir=args.data_dir,
                    # model_dir=args.model_dir,
                    cache_dir=args.cache_dir,
                    openai_key=args.openai_key,
                    cost_estimate=args.cost_estimate,
                    abstain_detection_type=args.abstain_detection_type)

    topics, generations, atomic_facts = prepare_atomic_facts(
        claims_path=args.claims_path, n_samples = args.n_samples
    )

    out = fs.get_score(topics=topics,
                       generations=generations,
                       gamma=args.gamma,
                       atomic_facts=atomic_facts if args.use_atomic_facts else None,
                       knowledge_source=args.knowledge_source,
                       verbose=args.verbose)
    fs.save_cache()

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
    
    # decisions_splits=[decisions[i::6] for i in range(6)]
    # for split in decisions_splits:
    #     assert len(split)==66
    # factscore_splits = [np.nanmean([np.mean([d["is_supported"] for d in internal_decisions if d["is_supported"] is not None]) for internal_decisions in split]) for split in decisions_splits]
    # out["splits_factscore"] = factscore_splits
    # out["splits_factscore_mean"] = np.mean(factscore_splits)
    # out["splits_factscore_std"] = np.std(factscore_splits)

    # Convert discarted topics to entities
    # entity_metrics={}    
    # for i, decisions in enumerate(out["discarted_decisions"]):
    #     for decision in decisions:
    #         is_expanded = type(decision["topic"]) == list
    #         decision["topic"] = articles_entity_mapping[str(decision["topic"])]

    #     if len(decisions) > 0:
    #         if decision["topic"] not in entity_metrics:
    #             entity_metrics[decision["topic"]] = {i:copy.deepcopy(decisions)}
    #         else:
    #             entity_metrics[decision["topic"]][i] = copy.deepcopy(decisions)
            
    
    # for i, decisions in enumerate(out["decisions"]):
    #     for decision in decisions:
    #         decision["topic"] = articles_entity_mapping[str(decision["topic"])]

    #     if decision["topic"] not in entity_metrics:
    #         entity_metrics[decision["topic"]] = {i:decisions}
    #     else:
    #         if i not in entity_metrics[decision["topic"]]:
    #             entity_metrics[decision["topic"]][i] = copy.deepcopy(decisions)
    #         else:
    #             entity_metrics[decision["topic"]][i] += copy.deepcopy(decisions)

    # final_entity_metrics = {}
    # for entity,values in entity_metrics.items():
    #     # Get score per answer
    #     for i, decisions in values.items():
    #         entity_metrics[entity][i] = {
    #             "score": np.mean([d["is_supported"] for d in decisions if d["is_supported"] is not None]),
    #             "avg_support": np.mean(np.array([d["is_supported"] for d in decisions]) == True),
    #             "avg_contradict": np.mean(np.array([d["is_supported"] for d in decisions]) == False),
    #             "avg_nei": np.mean(np.array([d["is_supported"] for d in decisions]) == None)
    #         }
    #     final_entity_metrics[entity] = {
    #         "score": np.nanmean([i_metrics["score"] for i,i_metrics in entity_metrics[entity].items()]),
    #         "avg_support": np.nanmean([i_metrics["avg_support"] for i,i_metrics in entity_metrics[entity].items()]),
    #         "avg_contradict": np.nanmean([i_metrics["avg_contradict"] for i,i_metrics in entity_metrics[entity].items()]),
    #         "avg_nei": np.nanmean([i_metrics["avg_nei"] for i,i_metrics in entity_metrics[entity].items()])
    #     }

    # out["entity_metrics"] = {k:json.loads(pd.DataFrame.from_records(list(v.values())).to_json(orient="records")) for k,v in entity_metrics.items()} # Convert possible NaN to null
    # out["final_entity_metrics"] = final_entity_metrics
    
    logging.critical("FActScore = %.1f%%" % (100*out["score"]))
    if "init_score" in out:
        logging.critical("FActScore w/o length penalty = %.1f%%" % (100*out["init_score"]))
    logging.critical("Respond ratio = %.1f%%" % (100*out["respond_ratio"]))
    logging.critical("# Atomic facts per valid response = %.1f" % (out["num_facts_per_response"]))

    with open(args.dataset_output_path, "w") as outfile:
        json.dump(out, outfile)
