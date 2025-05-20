import argparse
import json
import tqdm
import string
# from nltk.corpus import stopwords
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

embedder = SentenceTransformer("all-mpnet-base-v2")
# stop_words = set(stopwords.words("english"))


# def preprocess_string(s):
#     # Convert to lowercase and remove punctuation
#     s = s.lower()
#     s = s.translate(str.maketrans("", "", string.punctuation))
#     return s


# def remove_stopwords(s):
#     # Split the string into words and remove stopwords
#     words = s.split()
#     filtered_words = [word for word in words if word not in stop_words]
#     return " ".join(filtered_words)


def string_match(s1, s2):
    # Preprocess and remove stopwords
    s1_processed = remove_stopwords(preprocess_string(s1))
    s2_processed = remove_stopwords(preprocess_string(s2))
    # Compare the processed strings
    return s1_processed == s2_processed


def process_answers(answers, linkage = "average", distance_threshold=0.1):
    # bins = {}
    # for answer in answers:
    #     for key in bins.keys():
    #         if string_match(answer, key):
    #             bins[key] += 1
    #             break
    #     else:
    #         bins[answer] = 1

    corpus_embeddings = embedder.encode(answers)

    # Some models don't automatically normalize the embeddings, in which case you should normalize the embeddings:
    # corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1, keepdims=True)

    # Perform kmean clustering
    clustering_model = AgglomerativeClustering(
        metric="cosine", 
        linkage=linkage,  
        distance_threshold=distance_threshold, 
        n_clusters=None
    )
    clustering_model.fit(corpus_embeddings)
    cluster_assignment = clustering_model.labels_
    
    clustered_sentences = {}
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        cluster_id = int(cluster_id)
        if cluster_id not in clustered_sentences:
            clustered_sentences[cluster_id] = []

        clustered_sentences[cluster_id].append(answers[sentence_id])

    return clustered_sentences


def get_score(processed_answers):
    sorted_processed_answer = sorted(
        processed_answers.items(), key=lambda x: x[1], reverse=True
    )
    score = sorted_processed_answer[0][1] / 20.0
    return score


def paragraph_truthfulness(claims_questions_answers):
    claims_questions_answers_score = {}
    scores = []
    for claim, questions_answers in claims_questions_answers.items():
        answers = questions_answers["answers"]
        processed_answers = process_answers(answers, linkage = "complete", distance_threshold=0.1)
        sorted_processed_answers = sorted(
            processed_answers.values(),
            key=lambda x: len(x),
            reverse=True,
        )
        score = len(sorted_processed_answers[0]) / float(len(answers))

        claims_questions_answers_score[claim] = {
            "score": score,
            "question": questions_answers["question"],
            "processed_answers": sorted_processed_answers,
        }

        scores.append(claims_questions_answers_score[claim]["score"])

    return claims_questions_answers_score, np.mean(scores)


def truthfulness_score(answers_dataset_path=None, dataset_output_path=None):
    with open(answers_dataset_path) as json_file:
        entity_question_answers_claims_questions_answers = json.load(json_file)

    entity_question_answers_claims_questions_answers_score = {}
    for entity, question_answers_claims_questions_answers in tqdm.tqdm(
        entity_question_answers_claims_questions_answers.items()
    ):
        entity_question_answers_claims_questions_answers_score[entity] = {}
        for (
            question,
            answers_claims_questions_answers,
        ) in question_answers_claims_questions_answers.items():
            entity_question_answers_claims_questions_answers_score[entity][
                question
            ] = {}
            for (
                answer,
                claims_questions_answers,
            ) in answers_claims_questions_answers.items():
                claims_questions_answers_scores, score = paragraph_truthfulness(
                    claims_questions_answers
                )
                entity_question_answers_claims_questions_answers_score[entity][
                    question
                ][answer] = {
                    "score": score if not np.isnan(score) else None,
                    "claims_questions_answers_scores": claims_questions_answers_scores,
                }

    with open(dataset_output_path, "w") as outfile:
        json.dump(entity_question_answers_claims_questions_answers_score, outfile)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--answers_dataset_path", type=str)
    parser.add_argument("--dataset_output_path", type=str)

    args = parser.parse_args()

    truthfulness_score(
        answers_dataset_path=args.answers_dataset_path,
        dataset_output_path=args.dataset_output_path,
    )
