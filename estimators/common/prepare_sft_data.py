import json
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_input_path", type=str)
    parser.add_argument("--dataset_output_path", type=str)

    args = parser.parse_args()

    with open(args.dataset_input_path) as file:
        data = json.load(file)
    
    dataset = []
    entity_question_answers_claims_questions = data
    for entity, question_answers_claims_questions in entity_question_answers_claims_questions.items():
      for question, answers_claims_questions in question_answers_claims_questions.items():
        for answer, claims_questions in answers_claims_questions.items():
          dataset.append({"question": question, "answer": answer})
    
    with open(args.dataset_output_path, 'w') as f:
        for d in dataset:
            json.dump(d, f)
            f.write('\n')