from fairseq.models.roberta import RobertaModel

from tqdm import tqdm
import torch
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('answer_bpes')
    parser.add_argument('question_types')
    parser.add_argument('output')
    args = parser.parse_args()

    answer_bpes = []
    with open(args.answer_bpes) as f:
        for line in f:
            question = line.strip().split(' ') if line.strip() else []
            answer_bpes.append(question)

    with open(args.question_types) as f:
        question_types = [line.strip() for line in f]

    unique_qts = ['1', '2', '3', '5', '7', '8', '9', '10', '11', '13']
    qt_answer_data = {qt: [] for qt in unique_qts}

    for i, (answer_bpe, question_type) in enumerate(zip(answer_bpes, question_types)):
        qt_answer_data[question_type].append((i, answer_bpe))

    qt_preds = [None for _ in answer_bpes]
    for qt in unique_qts:

        roberta = RobertaModel.from_pretrained(
            os.path.join(args.model_dir, 'type{}'.format(qt)),
            checkpoint_file='model.pt',
            data_name_or_path='.'
        )

        roberta.cuda()
        roberta.eval()

        label_fn = lambda label: roberta.task.label_dictionary.string(
            [label + roberta.task.label_dictionary.nspecial]
        )

        preds = []

        test_questions = qt_answer_data[qt]
        test_question_ids = [x[0] for x in test_questions]
        test_questions = [x[1] for x in test_questions]

        bos = roberta.task.source_dictionary.bos()
        eos = roberta.task.source_dictionary.eos()

        with torch.no_grad():
            for question in tqdm(test_questions):
                question = [bos] + [roberta.task.source_dictionary.index(sym) for sym in question[:510]] + [eos]
                tokens = torch.tensor(question, dtype=torch.long)
                pred_lprob = roberta.predict('qt_head', tokens)
                prob, pred = pred_lprob.max(-1)

                pred = label_fn(pred.squeeze().item())

                preds.append(pred)

        assert len(test_question_ids) == len(preds)

        for id, pred in zip(test_question_ids, preds):
            qt_preds[id] = pred

    with open(args.output, 'w') as f:
        for pred in qt_preds:
            f.write(str(pred) + '\n')


if __name__ == '__main__':
    main()
