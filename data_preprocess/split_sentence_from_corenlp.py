import json
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parsed_file')
    parser.add_argument('out_file')
    args = parser.parse_args()

    out_dps = []
    with open(args.parsed_file) as f:
        for line in f:
            dp = json.loads(line)
            sents = [' '.join([tok['word'] for tok in sent['tokens']]) for sent in dp['sents_parsed']]
            out_dps.append({'id': dp['id'], 'sents': sents})

    with open(args.out_file, 'w') as f:
        for dp in out_dps:
            f.write(json.dumps(dp) + '\n')


if __name__ == '__main__':
    main()