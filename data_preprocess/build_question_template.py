import json
import argparse
import os
import numpy as np
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import string
import tempfile

from sklearn.metrics.pairwise import cosine_similarity
import subprocess as sp


puncts = set(string.punctuation)


SPLIT_SIZE = 60000


with open('stopwords') as f:
    stopwords = set([line.strip() for line in f])


concept_net_word_vectors = {}
with open('numberbatch-en-19.08.txt') as f:
    for line in f:
        tmp = line.strip().split()
        word = tmp[0]
        vector = [float(v) for v in tmp[1:]]
        concept_net_word_vectors[word] = np.array(vector)


def abstract_ne(sents):
    for sent in sents:
        sent_words = sent['words']
        entity_mentions = sent['entity_mentions']
        for mention in entity_mentions:
            ner_type = mention[0]
            ner_start = mention[1]
            ner_end = mention[2]

            if (ner_end - ner_start) == 1 and sent_words[ner_start]['word'] in puncts and sent_words[ner_start]['pos'] in ['DT', 'CC', 'IN']:
                continue

            if ner_type == 'DATE' and (ner_end - ner_start) == 1 and (sent_words[ner_start]['lemma'] in
                    ['once', 'present', 'fall', 'supper', 'about', 'Falls', 'Fall', 'Once']):
                continue

            for i in range(ner_start, ner_end):
                sent_words[i]['abstract'] = True
    return sents


def abstract_overlap(sents, answer_words):
    for sent in sents:
        sent_words = sent['words']
        for i, word in enumerate(sent_words):
            lower_word = word['lemma'].lower()
            if lower_word not in stopwords and lower_word not in puncts and lower_word in answer_words:
                word['abstract'] = True
    return sents


def update_replaced_constituency(old_constituency, new_constituency):
    if old_constituency is None:
        return new_constituency
    old_type, old_start, old_end = old_constituency
    new_type, new_start, new_end = new_constituency

    if new_start <= old_start and new_end >= old_end:
        return new_constituency

    return old_constituency


def abstract_constituent(sents):
    for sent in sents:
        sent_words = sent['words']
        constituency_spans = sent['constituency_spans']
        head2span = defaultdict(list)
        block_spans = []
        all_block_spans = []
        for ctype, start, end, head in constituency_spans:
            if ctype in ['NP', 'ADJP', 'ADVP']:
                head2span[head].append((ctype, start, end))
            elif ctype in ['S', 'SQ', 'SBAR']:
                block_spans.append((start, end))
                all_block_spans.append((start, end))
            elif ctype == 'PP':
                block_spans.append((start, end))

        for i, word in enumerate(sent_words):
            if word['abstract']:
                if i in head2span:
                    for span in head2span[i]:
                        block_indices = []
                        if span[0] == 'NP':
                            span_start = span[1]
                            span_end = span[2]
                            for block_start, block_end in all_block_spans:
                                if block_start >= span_start and block_end <= span_end:
                                    for j in range(block_start, block_end):
                                        block_indices.append(j)
                        elif span[0] in ['ADJP', 'ADVP']:
                            span_start = span[1]
                            span_end = span[2]
                            for block_start, block_end in block_spans:
                                if block_start >= span_start and block_end <= span_end:
                                    for j in range(block_start, block_end):
                                        block_indices.append(j)
                        for j in range(span[1], span[2]):
                            if j in block_indices:
                                continue
                            sent_words[j]['replaced_constituency'] = update_replaced_constituency(
                                sent_words[j]['replaced_constituency'], span
                            )
                            sent_words[j]['abstract'] = True
    return sents


def abstract_embedding_content_percent(sents, answer_words):
    answer_word2vec_vectors = []
    for answer_word, answer_lemma, answer_pos in zip(answer_words['words'], answer_words['lemmas'], answer_words['pos']):
        if answer_lemma.lower() in stopwords or answer_lemma in puncts:
            continue
        if answer_word.lower() in concept_net_word_vectors:
            answer_word2vec_vectors.append((answer_word, concept_net_word_vectors[answer_word.lower()]))
        elif answer_lemma.lower() in concept_net_word_vectors:
            answer_word2vec_vectors.append((answer_word, concept_net_word_vectors[answer_lemma.lower()]))

    # merge into matrix
    word2vec_words = [x[0] for x in answer_word2vec_vectors]
    answer_word2vec_vecs = [x[1] for x in answer_word2vec_vectors]
    answer_word2vec_matrix = np.stack(answer_word2vec_vecs)

    if not word2vec_words:
        return sents


    cnt_content = 0
    cnt_content_abstracted = 0
    for sent in sents:
        sent_words = sent['words']
        for word in sent_words:
            if word['lemma'].lower() in stopwords or word['lemma'].lower() in puncts:
                continue
            cnt_content += 1
            if word['abstract']:
                cnt_content_abstracted += 1

    further_abstract = int(cnt_content * 0.8) - cnt_content_abstracted
    if further_abstract <= 0:
        return sents

    all_words = []


    for j, sent in enumerate(sents):
        sent_words = sent['words']
        for i, word in enumerate(sent_words):
            if word['lemma'].lower() in stopwords or word['lemma'].lower() in puncts:
                continue
            q_word2vec_word_vec = None

            if word['abstract']:
                continue

            if word['word'].lower() in concept_net_word_vectors:
                q_word2vec_word_vec = np.expand_dims(concept_net_word_vectors[word['word'].lower()], 0)
            elif word['lemma'].lower() in concept_net_word_vectors:
                q_word2vec_word_vec = np.expand_dims(concept_net_word_vectors[word['lemma'].lower()], 0)

            if q_word2vec_word_vec is not None and word2vec_words:
                similarity_scores = cosine_similarity(q_word2vec_word_vec, answer_word2vec_matrix).squeeze()

                max_id = np.argmax(similarity_scores).item()

                all_words.append((j, i, similarity_scores[max_id].item()))

    for sid, wid, _ in sorted(all_words, key=lambda x: x[2], reverse=True)[:further_abstract]:
        sents[sid]['words'][wid]['abstract'] = True

    return sents


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('question_converted_jsonl')
    parser.add_argument('answer_jsonl')
    parser.add_argument('tmp_dir')
    parser.add_argument('out_jsonl')
    args = parser.parse_args()

    question_parses = []
    with open(args.question_converted_jsonl) as f:
        for line in f:
            question_parses.append(json.loads(line))

    ids = [question_parse['id'] for question_parse in question_parses]
    question_parses = [question_parse['sents_converted'] for question_parse in question_parses]

    print('Question loaded.')

    for sents in question_parses:
        for sent in sents:
            for w in sent['words']:
                w['replaced_constituency'] = None

    answer_words = []
    with open(args.answer_jsonl) as f:
        for line in f:
            dp = json.loads(line)
            answer_words.append({'words': dp['doc_words'], 'lemmas': dp['doc_lemmas'], 'pos': dp['doc_pos']})

    print('Answer loaded.')

    with tempfile.TemporaryDirectory(dir=args.tmp_dir) as tmp_dir:
        part_out_jsonls = []
        for part_id, part_start in enumerate(range(0, len(ids), SPLIT_SIZE)):
            print('===PART {}===='.format(part_id))

            part_ids = ids[part_start: part_start + SPLIT_SIZE]
            part_question_parses = question_parses[part_start: part_start + SPLIT_SIZE]
            part_answer_words = answer_words[part_start: part_start + SPLIT_SIZE]

            with ProcessPoolExecutor() as executor:
                ne_abstracted = []
                for question_parse in part_question_parses:
                    ne_abstracted.append(executor.submit(abstract_ne, question_parse))
                ne_abstracted = [x.result() for x in ne_abstracted]

            print('NE abstracted.')

            overlap_abstracted = []

            with ProcessPoolExecutor() as executor:
                for ne_abs, answer_word in zip(ne_abstracted, part_answer_words):
                    overlap_abstracted.append(executor.submit(abstract_overlap, ne_abs, set([w.lower() for w in answer_word['lemmas']])))
                overlap_abstracted = [x.result() for x in overlap_abstracted]

            print('Overlap abstracted.')

            constituent_abstracted = []

            with ProcessPoolExecutor() as executor:
                for overlap_abs in overlap_abstracted:
                    constituent_abstracted.append(executor.submit(abstract_constituent, overlap_abs))
                constituent_abstracted = [x.result() for x in constituent_abstracted]

            print('Constituent abstracted.')

            overlap_abstracted = constituent_abstracted

            embedding_abstracted = []
            with ProcessPoolExecutor() as executor:
                for overlap_abs, answer_word in zip(overlap_abstracted, part_answer_words):
                    embedding_abstracted.append(executor.submit(abstract_embedding_content_percent, overlap_abs, answer_word))
                embedding_abstracted = [x.result() for x in embedding_abstracted]

            print('Embedding abstracted.')

            constituent_abstracted = []

            with ProcessPoolExecutor() as executor:
                for embedding_abs in embedding_abstracted:
                    constituent_abstracted.append(executor.submit(abstract_constituent, embedding_abs))
                constituent_abstracted = [x.result() for x in constituent_abstracted]

            print('Constituent abstracted.')

            assert len(part_ids) == len(constituent_abstracted)

            part_out_jsonl = os.path.join(tmp_dir, 'part{}'.format(part_id))
            part_out_jsonls.append(part_out_jsonl)

            with open(part_out_jsonl, 'w') as f:
                for id, abs in zip(part_ids, constituent_abstracted):
                    f.write(json.dumps({'id': id, 'abs_sents': abs}) + '\n')

        with open(args.out_jsonl, 'w') as f:
            sp.run(['cat'] + part_out_jsonls, stdout=f)


if __name__ == '__main__':
    main()
