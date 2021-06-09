import argparse
import os
import re
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig


NP_TOKEN = 44320
VERB_TOKEN = 45003
ADJP_TOKEN = 45199
ADVP_TOKEN = 45544
OTHER_TOKEN = 45545


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-dir', nargs='+')
    args = parser.parse_args()

    bpe = GPT2BPE(GPT2BPEConfig()).bpe

    for sp in ['train', 'valid', 'test']:
        for generate_dir in args.generate_dir:

            all_samples = []
            sample = []
            all_bpe_samples = []
            bpe_sample = []
            if not os.path.exists(os.path.join(generate_dir, 'generate-{}.txt'.format(sp))):
                continue
            with open(os.path.join(generate_dir, 'generate-{}.txt'.format(sp))) as f:
                for line in f:
                    if line[0] == 'S':
                        if sample:
                            all_samples.append((sample_id, sample))
                            all_bpe_samples.append((sample_id, bpe_sample))
                            sample = []
                            bpe_sample = []
                        sample_id, sent = line.strip().split('\t')
                        sample_id = sample_id.split('-')[1]
                        sent = re.sub(r'\b44320\b', '<TEMP-NP>', sent)
                        sent = re.sub(r'\b45003\b', '<TEMP-VERB>', sent)
                        sent = re.sub(r'\b45199\b', '<TEMP-ADJP>', sent)
                        sent = re.sub(r'\b45544\b', '<TEMP-ADVP>', sent)
                        sent = re.sub(r'\b45545\b', '<TEMP-OTHER>', sent)
                        sent = bpe.decode([
                            int(tok) if tok not in {'<unk>', '<mask>', '<TEMP-NP>', '<TEMP-VERB>', '<TEMP-ADJP>', '<TEMP-ADVP>', '<TEMP-OTHER>'} else tok
                            for tok in sent.split()
                        ])
                        sent = sent.replace('<TEMP-NP>', ' <TEMP-NP>')
                        sent = sent.replace('<TEMP-VERB>', ' <TEMP-VERB>')
                        sent = sent.replace('<TEMP-ADJP>', ' <TEMP-ADJP>')
                        sent = sent.replace('<TEMP-ADVP>', ' <TEMP-ADVP>')
                        sent = sent.replace('<TEMP-OTHER>', ' <TEMP-OTHER>')
                        sent = sent.strip()
                        sample.append(sent)
                    elif line[0] == 'T':
                        sent = line.strip().split('\t')[1]
                        sent = re.sub(r'\b44320\b', '<TEMP-NP>', sent)
                        sent = re.sub(r'\b45003\b', '<TEMP-VERB>', sent)
                        sent = re.sub(r'\b45199\b', '<TEMP-ADJP>', sent)
                        sent = re.sub(r'\b45544\b', '<TEMP-ADVP>', sent)
                        sent = re.sub(r'\b45545\b', '<TEMP-OTHER>', sent)
                        sent = bpe.decode([
                            int(tok) if tok not in {'<unk>', '<mask>', '<TEMP-NP>', '<TEMP-VERB>', '<TEMP-ADJP>',
                                                    '<TEMP-ADVP>', '<TEMP-OTHER>'} else tok
                            for tok in sent.split()
                        ])
                        sent = sent.replace('<TEMP-NP>', ' <TEMP-NP>')
                        sent = sent.replace('<TEMP-VERB>', ' <TEMP-VERB>')
                        sent = sent.replace('<TEMP-ADJP>', ' <TEMP-ADJP>')
                        sent = sent.replace('<TEMP-ADVP>', ' <TEMP-ADVP>')
                        sent = sent.replace('<TEMP-OTHER>', ' <TEMP-OTHER>')
                        sent = sent.strip()
                        sample.append(sent)
                    elif line[0] == 'H':
                        sent = line.strip().split('\t')[-1]
                        bpe_sample.append(sent)
                        sent = re.sub(r'\b44320\b', '<TEMP-NP>', sent)
                        sent = re.sub(r'\b45003\b', '<TEMP-VERB>', sent)
                        sent = re.sub(r'\b45199\b', '<TEMP-ADJP>', sent)
                        sent = re.sub(r'\b45544\b', '<TEMP-ADVP>', sent)
                        sent = re.sub(r'\b45545\b', '<TEMP-OTHER>', sent)
                        sent = bpe.decode([
                            int(tok) if tok not in {'<unk>', '<mask>', '<TEMP-NP>', '<TEMP-VERB>', '<TEMP-ADJP>',
                                                    '<TEMP-ADVP>', '<TEMP-OTHER>'} else tok
                            for tok in sent.split()
                        ])
                        sent = sent.replace('<TEMP-NP>', ' <TEMP-NP>')
                        sent = sent.replace('<TEMP-VERB>', ' <TEMP-VERB>')
                        sent = sent.replace('<TEMP-ADJP>', ' <TEMP-ADJP>')
                        sent = sent.replace('<TEMP-ADVP>', ' <TEMP-ADVP>')
                        sent = sent.replace('<TEMP-OTHER>', ' <TEMP-OTHER>')
                        sent = sent.strip()
                        sample.append(sent)
            if sample:
                all_samples.append((sample_id, sample))
                all_bpe_samples.append((sample_id, bpe_sample))
                sample = []

            all_samples = sorted(all_samples, key=lambda x: int(x[0]))
            all_samples = [x[1] for x in all_samples]
            all_bpe_samples = sorted(all_bpe_samples, key=lambda x: int(x[0]))
            all_bpe_samples = [x[1] for x in all_bpe_samples]

            with open(os.path.join(generate_dir, 'formatted-{}.txt'.format(sp)), 'w') as f:
                for sample in all_samples:
                    out = re.sub(r'^\[.*?\]', '', sample[2]).strip()
                    f.write(out + '\n')
            with open(os.path.join(generate_dir, 'bpe-{}.txt'.format(sp)), 'w') as f:
                for sample in all_bpe_samples:
                    f.write(sample[0] + '\n')


if __name__ == '__main__':
    main()