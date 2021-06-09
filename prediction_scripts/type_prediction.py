from fairseq.models.roberta import RobertaModel

from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig

import argparse


class SimpleDataset(Dataset):
    def __init__(self, inputs):
        self.inputs = inputs

    def __getitem__(self, item):
        return self.inputs[item]

    def __len__(self):
        return len(self.inputs)


class SimpleLoader(DataLoader):
    def __init__(self, dataset, src_dict, bpe):
        def _collate(batch):
            pad_token = src_dict.pad()
            batch_tokens = []
            batch_lengths = []
            for sentence in batch:
                bpe_sentence = "<s> " + " ".join(map(str, bpe.encode(sentence))) + " </s>"
                tokens = src_dict.encode_line(
                    bpe_sentence, append_eos=False, add_if_not_exist=False
                )
                tokens = tokens.long()
                tokens = torch.cat([tokens[:-1][:511], tokens[-1:]])
                batch_lengths.append(tokens.numel())
                batch_tokens.append(tokens)

            bsz = len(batch_lengths)
            max_length = max(batch_lengths)
            batch_input = batch_tokens[0].new_full((bsz, max_length), pad_token)
            for i in range(bsz):
                batch_input[i, :batch_lengths[i]].copy_(batch_tokens[i])

            return batch_input
        super().__init__(dataset, batch_size=64, num_workers=16, collate_fn=_collate)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_dir')
    parser.add_argument('test_questions')
    parser.add_argument('output')
    args = parser.parse_args()

    bpe = GPT2BPE(GPT2BPEConfig()).bpe

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

    dataset = SimpleDataset(test_questions)
    data_loader = SimpleLoader(dataset, roberta.task.source_dictionary, bpe)

    with torch.no_grad():
        for batch_input in tqdm(data_loader):
            pred_lprob = roberta.predict('qt_head', batch_input)
            prob, pred = pred_lprob.max(-1)

            pred = [label_fn(l) for l in pred.tolist()]

            preds += pred

    with open(args.output, 'w') as f:
        for pred in preds:
            f.write(pred + '\n')


if __name__ == '__main__':
    main()
