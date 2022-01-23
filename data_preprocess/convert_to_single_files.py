import argparse
import os
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonl_file')
    parser.add_argument('out_dir')
    parser.add_argument('path_file')
    args = parser.parse_args()

    samples = []
    with open(args.jsonl_file) as f:
        for line in f:
            dp = json.loads(line)
            samples.append((dp['id'].replace('/', '_'), dp['answer'], dp['question']))

    os.makedirs(args.out_dir)
    out_paths = []
    for id, answer, question in samples:
        out_path = os.path.join(args.out_dir, f'{id}.answer')
        with open(out_path, 'w') as f:
            f.write(answer)
        out_paths.append(out_path)

        out_path = os.path.join(args.out_dir, f'{id}.question')
        with open(out_path, 'w') as f:
            f.write(question)
        out_paths.append(out_path)

    with open(args.path_file, 'w') as f:
        for out_path in out_paths:
            f.write(out_path + '\n')


if __name__ == '__main__':
    main()