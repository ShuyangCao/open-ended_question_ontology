import os

from fairseq.models.roberta import RobertaModel

from tqdm import tqdm
import torch

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('test_questions')
    parser.add_argument('output_dir')
    parser.add_argument('--k', type=int, default=9)
    args = parser.parse_args()

    roberta = RobertaModel.from_pretrained(
        args.model_dir,
        checkpoint_file='model.pt',
        data_name_or_path='.'
    )

    roberta.cuda()
    roberta.eval()

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )

    preds = []

    test_questions = []
    with open(args.test_questions) as f:
        for line in f:
            question = line.strip().split('\t')[-1]
            test_questions.append(question)

    with torch.no_grad():
        for test_question in tqdm(test_questions):
            tokens = roberta.encode(test_question)

            pred_lprob = roberta.predict('qt_head', tokens)
            pred = torch.topk(pred_lprob, dim=-1, k=args.k)[1][0]

            pred = [label_fn(l) for l in pred.tolist()]

            preds.append(pred)

    os.makedirs(args.output_dir, exist_ok=True)
    for k in range(args.k):
        with open(os.path.join(args.output_dir, f'control_type{k+1}'), 'w') as f:
            for pred in preds:
                f.write(str(pred[k]) + '\n')


if __name__ == '__main__':
    main()
