import argparse
import json
from collections import OrderedDict
import string
from concurrent.futures import ProcessPoolExecutor
import regex as re
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig


puncts = set(string.punctuation)

MAX_WORD = 256

tokenizer = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
bpe_encoder = GPT2BPE(GPT2BPEConfig()).bpe

with open('stopwords') as f:
    stopwords = set([line.strip() for line in f])


def tune_threshold(words, prob, bpe2word, node2bpe):
    prob = [float(x) for x in prob.strip().split(' ')] if prob.strip() else []
    bpe2word = [int(x) for x in bpe2word.strip().split(' ')]
    assert max(bpe2word) == len(words) - 1, '{} {}'.format(max(bpe2word), words)
    node2bpe = [[int(xx) for xx in x.split(' ')] for x in node2bpe.strip().split('\t')]

    node2words = [[bpe2word[x] for x in n2b] for n2b in node2bpe]
    node2contentlemma = [[words[x]['lemma'].lower() for x in n2w if words[x]['lemma'].lower() not in stopwords]
                         for n2w in node2words]

    pred_lemma = set()
    for i, p in enumerate(prob):
        if p > 0.5:
            pred_lemma.update(node2contentlemma[i])

    return pred_lemma


def focus_overlap(sents, prediction_lemma):
    for sent in sents:
        sent_words = sent['words']
        for i, word in enumerate(sent_words):
            lower_word = word['lemma'].lower()
            if lower_word not in stopwords and lower_word not in puncts:
                if lower_word in prediction_lemma:
                    word['lemma_focus'] = True
            else:
                word['lemma_focus'] = False
    return sents


def get_focus_out(sents_words):
    trimmed_sents_words = []
    accumulated_sents_words = 0
    for sent_words in sents_words:
        if accumulated_sents_words + len(sent_words) > MAX_WORD:
            break
        trimmed_sents_words.append(sent_words)
        accumulated_sents_words += len(sent_words)

    lexicals = OrderedDict()
    for sent_words in trimmed_sents_words:
        for wi, word in enumerate(sent_words):
            if 'lemma_focus' in word and word['lemma_focus']:
                lexicals[word['lemma']] = None

    return list(lexicals)


def process_one(focus):
    focus_text = ' // '.join(focus)
    focus_bpe_tokens = []
    for token in re.findall(tokenizer, focus_text):
        token = ''.join(bpe_encoder.byte_encoder[b] for b in token.encode('utf-8'))
        focus_bpe_tokens.extend([bpe_encoder.encoder[bpe_token] for bpe_token in bpe_encoder.bpe(token).split(' ')])
    return focus_text, focus_bpe_tokens


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('prob_file')
    parser.add_argument('tplgen_data_prefix')
    parser.add_argument('focus_jsonl')
    parser.add_argument('answer_converted_jsonl')
    parser.add_argument('out_prefix')
    args = parser.parse_args()

    focus_jsonls = []
    with open(args.focus_jsonl) as f:
        for i, line in enumerate(f):
            sents = json.loads(line)['focus_sents']
            words = []
            for sent in sents:
                if len(words) + len(sent['words']) > MAX_WORD:
                    break
                words.extend(sent['words'])
            focus_jsonls.append(words)

    with open(args.prob_file) as f:
        probs = f.readlines()

    with open(f'{args.tplgen_data_prefix}.bpe2word') as f:
        bpe2words = f.readlines()

    with open(f'{args.tplgen_data_prefix}.node2bpe') as f:
        node2bpes = f.readlines()

    with ProcessPoolExecutor() as executor:
        predicted_lemmas = []
        for focus_jsonl, prob, bpe2word, node2bpe in zip(focus_jsonls, probs, bpe2words, node2bpes):
            predicted_lemmas.append(executor.submit(tune_threshold, focus_jsonl, prob, bpe2word, node2bpe))
        predicted_lemmas = [future.result() for future in predicted_lemmas]  # list of set

    answer_parses = []
    with open(args.answer_converted_jsonl) as f:
        for line in f:
            dp = json.loads(line)
            answer_parses.append(dp)

    answer_parses = [answer_parse['sents_converted'] for answer_parse in answer_parses]

    overlap_focuses = []
    with ProcessPoolExecutor() as executor:
        for predicted_lemma, answer_parse in zip(predicted_lemmas, answer_parses):
            overlap_focuses.append(executor.submit(focus_overlap, answer_parse, predicted_lemma))
        overlap_focuses = [x.result() for x in overlap_focuses]

    with ProcessPoolExecutor() as executor:
        focus_lemmas = []
        for overlap_focus in overlap_focuses:
            sents_words = [sent['words'] for sent in overlap_focus]
            focus_lemmas.append(executor.submit(get_focus_out, sents_words))
        focus_lemmas = [future.result() for future in focus_lemmas]

    with ProcessPoolExecutor() as executor:
        outputs = []
        for focus_lemma in focus_lemmas:
            outputs.append(executor.submit(process_one, focus_lemma))
        outputs = [future.result() for future in outputs]

    with open(args.out_prefix + '.source', 'w') as fsrc, open(args.out_prefix + '.bpe.source', 'w') as fbpe:
        for src_text, src_bpe in outputs:
            fsrc.write(src_text + '\n')
            fbpe.write(' '.join([str(bpe) for bpe in src_bpe]) + '\n')


if __name__ == '__main__':
    main()
