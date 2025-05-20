import argparse
import json
from itertools import combinations
import math


def generate_preferences_dataset(scores_dataset_path=None, dataset_output_path=None, chosen_threshold=0.0):
    with open(scores_dataset_path) as json_file:
        entity_question_answers_claims_questions_answers_score = json.load(json_file)

    preferences_dataset = []
    for (
        entity,
        question_answers_claims_questions_answers_score,
    ) in entity_question_answers_claims_questions_answers_score.items():
        for (
            question,
            answers_claims_questions_answers_score,
        ) in question_answers_claims_questions_answers_score.items():
            answers_claims_questions_answers_score = {
                key: value
                for key, value in answers_claims_questions_answers_score.items()
                if (value["score"] is not None) and not math.isnan(value["score"])
            }
            for key_1, key_2 in combinations(answers_claims_questions_answers_score, 2):
                answer_1 = answers_claims_questions_answers_score[key_1]
                answer_2 = answers_claims_questions_answers_score[key_2]

                if answer_1["score"] > answer_2["score"]:
                    if answer_1["score"] < chosen_threshold:
                        continue
                    preferences_dataset.append(
                        {
                            "question": question,  ## We will give format later
                            "chosen": key_1,
                            "rejected": key_2,
                        }
                    )
                elif answer_1["score"] < answer_2["score"]:
                    if answer_2["score"] < chosen_threshold:
                        continue
                    preferences_dataset.append(
                        {
                            "question": question,
                            "chosen": key_2,
                            "rejected": key_1,
                        }
                    )
    with open(dataset_output_path, "w") as outfile:
        for preference_element in preferences_dataset:
            json.dump(preference_element, outfile)
            outfile.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores_dataset_path", type=str)
    parser.add_argument("--dataset_output_path", type=str)
    parser.add_argument("--chosen_threshold", type=float, default=0.0, help="Default: 0.0 (no limit)")

    args = parser.parse_args()

    generate_preferences_dataset(
        scores_dataset_path=args.scores_dataset_path,
        dataset_output_path=args.dataset_output_path,
        chosen_threshold=args.chosen_threshold        
    )
