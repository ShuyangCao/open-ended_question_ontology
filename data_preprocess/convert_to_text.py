import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('raw_jsonl')
    parser.add_argument('out_prefix')
    args = parser.parse_args()

    answers = []
    questions = []
    with open(args.raw_jsonl) as f:
        for line in f:
            dp = json.loads(line)
            questions.append(dp['question'])
            answers.append(dp['answer'])

    with open(f'{args.out_prefix}_answer.txt', 'w') as f:
        for answer in answers:
            f.write(answer + '\n')

    with open(f'{args.out_prefix}_question.txt', 'w') as f:
        for question in questions:
            f.write(question + '\n')


if __name__ == '__main__':
    main()
