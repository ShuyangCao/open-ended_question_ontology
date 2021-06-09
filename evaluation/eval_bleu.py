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

    smooth_method = nltk.translate.bleu_score.SmoothingFunction()

    b4s = []
    for i, (hyp, ref) in enumerate(zip(hyps, refs)):
        hyp = hyp.split(' ')
        ref = ref.split(' ')
        b4 = nltk.translate.bleu_score.sentence_bleu([ref], hyp, smoothing_function=smooth_method.method2)
        b4s.append(b4)

    print('BLEU-4', np.mean(b4s))


if __name__ == '__main__':
    main()