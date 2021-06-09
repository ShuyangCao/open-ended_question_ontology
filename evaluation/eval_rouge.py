import argparse
import numpy as np
from rouge_score import rouge_scorer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('hyp')
    parser.add_argument('ref')
    args = parser.parse_args()

    with open(args.hyp) as f:
        hyps = f.readlines()
        hyps = [l.strip().lower() for l in hyps]

    with open(args.ref) as f:
        refs = f.readlines()
        refs = [l.strip().lower() for l in refs]

    print(len(refs), len(hyps))

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    rls = []
    for i, (hyp, ref) in enumerate(zip(hyps, refs)):
        rouge = scorer.score(ref, hyp)
        rls.append(rouge['rougeL'])

    print('ROUGEL', np.mean(rls))


if __name__ == '__main__':
    main()