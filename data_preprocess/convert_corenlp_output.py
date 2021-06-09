import json
import argparse
import numpy as np
from nltk import Tree
from concurrent.futures import ProcessPoolExecutor


def get_depth(word_i, span, words, depth):
    if word_i < span[0] or word_i >= span[1]:
        return depth
    current_depth = depth
    if words[word_i]['head'] is None:
        return depth
    else:
        depth = max(get_depth(words[word_i]['head'], span, words, current_depth + 1), depth)
        return depth


def get_span_head(span, words):
    span_node_depth = [get_depth(i, span, words, 0) for i in range(span[0], span[1])]
    return span[0] + np.argmin(span_node_depth).item()


def get_constituency_spans(tree, start_id):
    spans = []
    if tree.label() in ['VP', 'NP', 'ADVP', 'ADJP', 'PP', 'SBAR', 'S']:
        spans.append((tree.label(), start_id, start_id + len(tree.leaves())))
    current_start = start_id
    for child in tree:
        if isinstance(child, Tree):
            spans.extend(get_constituency_spans(child, current_start))
            current_start += len(child.leaves())
    return spans


def constituency_tree(sent_parsed):
    tree = Tree.fromstring(sent_parsed['parse'], leaf_pattern="[^\s()]+(\s*[^\s()]+)*")
    constituency_spans = get_constituency_spans(tree, 0)
    return constituency_spans


def construct_tree(sent_parsed):
    words = sent_parsed['tokens']
    constituency_spans = constituency_tree(sent_parsed)
    for word in words:
        word['head'] = None
        word['dep'] = None
        word['children'] = []
        word['abstract'] = False
        word['replaced_constituency'] = None
    for dependency in sent_parsed['basicDependencies']:
        if dependency['governor'] > 0:
            governor = dependency['governor'] - 1
            dependent = dependency['dependent'] - 1
            dep = dependency['dep']
            words[dependent]['head'] = governor
            words[dependent]['dep'] = dep
            words[governor]['children'].append(dependent)

    entity_mentions = [(mention['ner'], mention['tokenBegin'], mention['tokenEnd']) for mention in sent_parsed['entitymentions']]

    try:
        constituency_spans = [(ctype, cstart, cend, get_span_head((cstart, cend), words)) for ctype, cstart, cend in
                              constituency_spans]
    except IndexError:
        tree = Tree.fromstring(sent_parsed['parse'], leaf_pattern="[^\s()]+(\s*[^\s()]+)*")
        print(tree.leaves())
        print([w['word'] for w in words], constituency_spans, sent_parsed['parse'])
        exit(0)

    return {'words': words, 'constituency_spans': constituency_spans, 'parse_tree': sent_parsed['parse'],
            'entity_mentions': entity_mentions}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('parse_jsonl')
    parser.add_argument('out_jsonl')
    args = parser.parse_args()

    parses = []
    with open(args.parse_jsonl) as f:
        for line in f:
            dp = json.loads(line)
            parses.append((dp['id'], dp['sents_parsed']))

    with ProcessPoolExecutor() as executor:
        converted_parses = []
        for id, parse in parses:
            converted_parses.append((id, [executor.submit(construct_tree, sent) for sent in parse]))
        converted_parses = [{'id': id, 'sents_converted': [sent.result() for sent in converted_parse]} for id, converted_parse in converted_parses]

    with open(args.out_jsonl, 'w') as f:
        for converted_parse in converted_parses:
            f.write(json.dumps(converted_parse) + '\n')


if __name__ == '__main__':
    main()