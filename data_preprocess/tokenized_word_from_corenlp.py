import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonl_file')
    parser.add_argument('out_file')
    args = parser.parse_args()

    out_jsons = []
    with open(args.jsonl_file) as f:
        for line in f:
            dp = json.loads(line)
            dp_lemmas = [tok['lemma'] for sent in dp['sents_parsed'] for tok in sent['tokens']]
            dp_pos = [tok['pos'] for sent in dp['sents_parsed'] for tok in sent['tokens']]
            out_jsons.append({'id': dp['id'], 'doc_words': dp['coref']['document'], 'doc_lemmas': dp_lemmas,
                              'doc_pos': dp_pos})

    with open(args.out_file, 'w') as f:
        for out_json in out_jsons:
            f.write(json.dumps(out_json) + '\n')


if __name__ == '__main__':
    main()