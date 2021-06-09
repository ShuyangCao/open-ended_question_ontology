import json
import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig
import regex as re


tokenizer = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

bpe_encoder = GPT2BPE(GPT2BPEConfig()).bpe

NP_TOKEN = 44320
VERB_TOKEN = 45003
ADJP_TOKEN = 45199
ADVP_TOKEN = 45544
OTHER_TOKEN = 45545


type_words = {
    '1': set(),
    '2': set(['or']),
    '3': set(['mean']),
    '5': set(['many', 'much', 'long', 'take', 'get']),
    '7': set(['good', 'best', 'find', 'anyone', 'get']),
    '8': set(['difference', 'best', 'better', 'and', 'or']),
    '9': set(['think', 'would', 'like', 'anyone']),
    '10': set(['people']),
    '11': set(['happens', 'would', 'affect', 'happen', 'effects', 'effect']),
    '13': set(['get', 'way', 'make', 'best', 'know'])
}


def process_one(line, type):
    data = json.loads(line)
    type_word = type_words[type]

    abstracted_words = []
    template_bpe_tokens = []
    template_fill_tgt_bpe_tokens = []
    template_fill_tgt_words = []

    for abs_sent in data['abs_sents']:
        curr_temp = False
        curr_constituency = None
        curr_temp_length = 0
        for wid, word in enumerate(abs_sent['words']):
            curr_token_bpe_tokens = []
            if wid != 0:
                word_text = ' ' + word['word']
            else:
                word_text = word['word']
                word_text = word_text[0].upper() + word_text[1:]
            for token in re.findall(tokenizer, word_text):
                token = ''.join(bpe_encoder.byte_encoder[b] for b in token.encode('utf-8'))
                curr_token_bpe_tokens.extend([bpe_encoder.encoder[bpe_token] for bpe_token in bpe_encoder.bpe(token).split(' ')])
            template_fill_tgt_bpe_tokens.extend(curr_token_bpe_tokens)
            template_fill_tgt_words.append(word_text.strip())

            if word_text.strip().lower() in type_word:
                word['abstract'] = False
                word['replaced_constituency'] = None

            if wid + 1 < len(abs_sent['words']) and word['pos'] == 'DT':
                next_word = abs_sent['words'][wid + 1]['word']
                if type == '7' and next_word.lower() in ['good', 'best']:
                    word['abstract'] = False
                    word['replaced_constituency'] = None
                elif type == '8' and next_word.lower() in ['difference', 'best']:
                    word['abstract'] = False
                    word['replaced_constituency'] = None
                elif type == '10' and next_word.lower() == 'people':
                    word['abstract'] = False
                    word['replaced_constituency'] = None
                elif type == '11' and next_word.lower() in ['effects', 'effect']:
                    word['abstract'] = False
                    word['replaced_constituency'] = None

            # print(curr_temp_length, curr_token_bpe_tokens)
            if word['abstract'] and not curr_temp:
                curr_temp = True
                curr_constituency = word['replaced_constituency']
                # assert len(curr_token_bpe_tokens) == 0
                curr_temp_length += len(curr_token_bpe_tokens)
                if word['replaced_constituency'] is not None:
                    if curr_constituency[0] == 'NP':
                        template_bpe_tokens.append(NP_TOKEN)
                    elif curr_constituency[0] == 'ADJP':
                        template_bpe_tokens.append(ADJP_TOKEN)
                    elif curr_constituency[0] == 'ADVP':
                        template_bpe_tokens.append(ADVP_TOKEN)
                    else:
                        raise NotImplementedError
                    abstracted_words.append('<TEMP-{}>'.format(word['replaced_constituency'][0]))
                elif word['pos'] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                    template_bpe_tokens.append(VERB_TOKEN)
                    curr_constituency = 'VERB'
                    abstracted_words.append('<TEMP-VERB>')
                else:
                    template_bpe_tokens.append(OTHER_TOKEN)
                    abstracted_words.append('<TEMP-OTHER>')
            elif not word['abstract']:
                curr_temp_length = 0
                abstracted_words.append(word_text.strip())
                template_bpe_tokens.extend(curr_token_bpe_tokens)
                curr_temp = False
                curr_constituency = None
            elif curr_constituency == 'VERB' and word['replaced_constituency'] is None:
                if word['pos'] not in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:

                    curr_temp_length = len(curr_token_bpe_tokens)
                    curr_constituency = None
                    abstracted_words.append('<TEMP-OTHER>')
                    template_bpe_tokens.append(OTHER_TOKEN)
                else:
                    curr_temp_length += len(curr_token_bpe_tokens)
            elif curr_constituency is None and word['replaced_constituency'] is None \
                    and word['pos'] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:

                curr_temp_length = len(curr_token_bpe_tokens)
                curr_constituency = 'VERB'
                abstracted_words.append('<TEMP-VERB>')
                template_bpe_tokens.append(VERB_TOKEN)
            elif curr_constituency != word['replaced_constituency']:
                curr_constituency = word['replaced_constituency']

                curr_temp_length = len(curr_token_bpe_tokens)

                if word['replaced_constituency'] is not None:
                    abstracted_words.append('<TEMP-{}>'.format(word['replaced_constituency'][0]))
                    if curr_constituency[0] == 'NP':
                        template_bpe_tokens.append(NP_TOKEN)
                    elif curr_constituency[0] == 'ADJP':
                        template_bpe_tokens.append(ADJP_TOKEN)
                    elif curr_constituency[0] == 'ADVP':
                        template_bpe_tokens.append(ADVP_TOKEN)
                    else:
                        raise NotImplementedError
                elif word['pos'] in ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']:
                    abstracted_words.append('<TEMP-VERB>')
                    template_bpe_tokens.append(VERB_TOKEN)
                    curr_constituency = 'VERB'
                else:
                    template_bpe_tokens.append(OTHER_TOKEN)
                    abstracted_words.append('<TEMP-OTHER>')
            else:
                curr_temp_length += len(curr_token_bpe_tokens)

    return {
        'source': abstracted_words,
        'bpe.source': template_bpe_tokens,
        'target': template_fill_tgt_words,
        'bpe.target': template_fill_tgt_bpe_tokens
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('jsonl_file')
    parser.add_argument('type')
    parser.add_argument('output_prefix')
    args = parser.parse_args()

    with open(args.type) as f:
        question_types = [line.strip() for line in f]

    with ProcessPoolExecutor() as executor:
        futures = []
        with open(args.jsonl_file) as f:
            for i, line in enumerate(f):
                futures.append(executor.submit(process_one, line, question_types[i]))
        results = [future.result() for future in futures]

    dict_results = defaultdict(list)
    for result in results:
        for k, v in result.items():
            dict_results[k].append(v)

    for k, v in dict_results.items():
        with open(args.output_prefix + '.' + k, 'w') as f:
            for sample_v in v:
                f.write(' '.join(str(ele) for ele in sample_v) + '\n')


if __name__ == '__main__':
    main()