import argparse
import nltk
import numpy as np


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

    mtrs = []
    for i, (hyp, ref) in enumerate(zip(hyps, refs)):
        mtr = nltk.translate.meteor_score.single_meteor_score(ref, hyp)
        mtrs.append(mtr)

    print('METEOR', np.mean(mtrs))


if __name__ == '__main__':
    main()