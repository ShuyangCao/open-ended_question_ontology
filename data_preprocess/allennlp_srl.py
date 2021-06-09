from allennlp.predictors.predictor import Predictor
import argparse
import json
from tqdm import tqdm

semantic_parser = Predictor.from_path(
    "https://storage.googleapis.com/allennlp-public-models/bert-base-srl-2020.03.24.tar.gz")


def parse_one(dp):
    sents = dp['sents']

    sents_srl = []
    for sent in sents:
        try:
            sent_srl = semantic_parser.predict(sentence=sent)
        except IndexError:
            sent_srl = None
        sents_srl.append(sent_srl)
    dp['sents_srl'] = sents_srl
    return dp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonl_file')
    parser.add_argument('out_file')
    args = parser.parse_args()

    i = 0
    dps = []
    with open(args.jsonl_file) as f:
        for line in tqdm(f):
            i += 1
            dp = json.loads(line)
            dps.append(parse_one(dp))

    with open(args.out_file, 'w') as f:
        for dp in dps:
            f.write(json.dumps(dp) + '\n')


if __name__ == '__main__':
    main()