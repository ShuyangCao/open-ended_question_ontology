import json
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import string


puncts = set(string.punctuation)


with open('stopwords') as f:
    stopwords = set([line.strip() for line in f])


def overlap_focus(sents, question_words):
    for sent in sents:
        sent_words = sent['words']
        for i, word in enumerate(sent_words):
            lower_word = word['lemma'].lower()
            if lower_word not in stopwords and lower_word not in puncts and lower_word in question_words:
                word['lemma_focus'] = True
            else:
                word['lemma_focus'] = False
    return sents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('question_converted_jsonl')
    parser.add_argument('answer_converted_jsonl')
    parser.add_argument('out_jsonl')
    args = parser.parse_args()

    question_parses = []
    with open(args.question_converted_jsonl) as f:
        for line in f:
            question_parses.append(json.loads(line))

    ids = [question_parse['id'] for question_parse in question_parses]
    question_parses = [question_parse['sents_converted'] for question_parse in question_parses]
    question_words = [[word for sent in question_parse for word in sent['words']] for question_parse in question_parses]

    print('Question loaded.')

    answer_parses = []
    with open(args.answer_converted_jsonl) as f:
        for line in f:
            dp = json.loads(line)
            answer_parses.append(dp)

    answer_parses = [answer_parse['sents_converted'] for answer_parse in answer_parses]

    print('Answer loaded.')

    overlap_focuses = []

    with ProcessPoolExecutor() as executor:
        for question_word, answer_parse in zip(question_words, answer_parses):
            question_lemma_cnt = Counter([w['lemma'].lower() for w in question_word if w['lemma'].lower() not in stopwords and w['lemma'].lower() not in puncts])
            overlap_focuses.append(executor.submit(overlap_focus, answer_parse, question_lemma_cnt))
        overlap_focuses = [x.result() for x in overlap_focuses]

    print('Overlap focus.')

    with open(args.out_jsonl, 'w') as f:
        for id, focus_result in zip(ids, overlap_focuses):
            f.write(json.dumps({'id': id, 'focus_sents': focus_result}) + '\n')


if __name__ == '__main__':
    main()
