import os
import argparse
import json
from concurrent.futures import ProcessPoolExecutor


def convert_one(parsed_file):
    with open(parsed_file) as f:
        parsed_output = json.load(f)
    id = parsed_output['docId'].split('.')[0]

    cumulative_start = []
    doc_words = []

    sents_parsed = []
    for sentence in parsed_output['sentences']:
        cumulative_start.append(len(doc_words))
        sentence_words = [tok['word'] for tok in sentence['tokens']]
        doc_words.extend(sentence_words)
        sents_parsed.append(sentence)

    clusters = []
    clusters_heads = []
    for cluster in parsed_output['corefs'].values():
        spans = []
        heads = []
        for span in cluster:
            span_start = span['startIndex']
            span_end = span['endIndex']
            span_sent = span['sentNum']
            span_head = span['headIndex']
            span_head = span_head + cumulative_start[span_sent - 1]
            span_start = span_start + cumulative_start[span_sent - 1]
            span_end = span_end + cumulative_start[span_sent - 1]
            spans.append([span_start, span_end])
            heads.append(span_head)
        clusters.append(spans)
        clusters_heads.append(heads)

    coref = {'document': doc_words, 'clusters': clusters, 'clusters_heads': clusters_heads}

    return {'id': id, 'sents_parsed': sents_parsed, 'coref': coref}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parsed_dir')
    parser.add_argument('order_file')
    parser.add_argument('answer_out_file')
    parser.add_argument('question_out_file')
    args = parser.parse_args()

    parsed_files = os.listdir(args.parsed_dir)

    order = []
    with open(args.order_file) as f:
        for line in f:
            if 'answer' in line:
                order.append(line.strip().split('/')[-1].split('.')[0])

    with ProcessPoolExecutor() as executor:
        answer_parsed_results = []
        question_parsed_results = []
        for parsed_file in parsed_files:
            if 'answer' in parsed_file:
                answer_parsed_results.append(executor.submit(convert_one, os.path.join(args.parsed_dir, parsed_file)))
            else:
                question_parsed_results.append(executor.submit(convert_one, os.path.join(args.parsed_dir, parsed_file)))
        answer_parsed_results = [x.result() for x in answer_parsed_results]
        question_parsed_results = [x.result() for x in question_parsed_results]

    answer_tmp = {parsed_result['id']: parsed_result for parsed_result in answer_parsed_results}
    question_tmp = {parsed_result['id']: parsed_result for parsed_result in question_parsed_results}
    answer_parsed_results = [answer_tmp[id] for id in order]
    question_parsed_results = [question_tmp[id] for id in order]

    with open(args.answer_out_file, 'w') as f:
        for parsed_result in answer_parsed_results:
            f.write(json.dumps(parsed_result) + '\n')

    with open(args.question_out_file, 'w') as f:
        for parsed_result in question_parsed_results:
            f.write(json.dumps(parsed_result) + '\n')


if __name__ == '__main__':
    main()
