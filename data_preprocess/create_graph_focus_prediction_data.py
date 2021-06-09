import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from fairseq.data.encoders.gpt2_bpe import GPT2BPE, GPT2BPEConfig
import regex as re


tokenizer = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

bpe_encoder = GPT2BPE(GPT2BPEConfig()).bpe

MAX_WORDS = 256


def focus_prediction_graph(trimmed_sents_words, sents_graphs):

    assert [w['word'] for w in trimmed_sents_words] == sents_graphs['words'], '{} {} {} {}'.format(
        sents_graphs['id'], len(trimmed_sents_words), [w['word'] for w in trimmed_sents_words], sents_graphs['words']
    )

    word_focus = [('lemma_focus' in w and w['lemma_focus']) for w in trimmed_sents_words]

    nodes = sents_graphs['nodes']
    disable_nodes = set(sents_graphs['disable_nodes'])
    edges = sents_graphs['edges']
    focus_nodes = []
    for nid, node in enumerate(nodes):
        if nid in disable_nodes:
            continue
        if node[3]:
            for coref in node[3]:
                if any(word_focus[coref[0]:coref[1]]):
                    focus_nodes.append(nid)
        else:
            if any(word_focus[node[1]:node[2]]):
                focus_nodes.append(nid)
    focus_nodes = set(focus_nodes)

    word_bpes = []
    bpe2word = []
    word_bpe_end = []
    current_end = 0
    for i, word in enumerate([w['word'] for w in trimmed_sents_words]):
        if i != 0:
            word = ' ' + word

        word_bpe = []
        for token in re.findall(tokenizer, word):
            token = ''.join(bpe_encoder.byte_encoder[b] for b in token.encode('utf-8'))
            word_bpe.extend([bpe_encoder.encoder[bpe_token] for bpe_token in bpe_encoder.bpe(token).split(' ')])
        word_bpes.append(word_bpe)

        word_bpe_end.append((current_end, current_end + len(word_bpe)))
        current_end += len(word_bpe)

        bpe2word.extend([i] * len(word_bpe))

    all_bpes = [x for word_bpe in word_bpes for x in word_bpe]

    node_offset = []
    current_offset = 0

    bpe2node = [-1 for _ in all_bpes]
    node2bpe = []
    node_cnt = []
    for i, (_, node_start, _, coref_spans) in enumerate(nodes):
        if i in disable_nodes:
            current_offset -= 1
            node_offset.append(1)
            continue
        node_offset.append(current_offset)
        if coref_spans:
            bpe_spans = []
            for span_start, _ in coref_spans:
                bpe_start, bpe_end = word_bpe_end[span_start]
                for j in range(bpe_start, bpe_end):
                    bpe2node[j] = len(node2bpe)  # the new node id
                bpe_spans.extend(list(range(bpe_start, bpe_end)))
            node2bpe.append(bpe_spans)
            node_cnt.append(len(coref_spans))
        else:
            bpe_start, bpe_end = word_bpe_end[node_start]
            for j in range(bpe_start, bpe_end):
                bpe2node[j] = len(node2bpe)  # the new node id
            bpe_spans = list(range(bpe_start, bpe_end))
            node2bpe.append(bpe_spans)
            node_cnt.append(1)
    node_target = [0 for _ in node2bpe]
    for nf in focus_nodes:
        node_target[nf + node_offset[nf]] = 1

    assert len(node2bpe) == len(node_cnt)

    processed_edges = []
    for i, (edge_type, edge_src, edge_tgt) in enumerate(edges):
        assert node_offset[edge_src] != 1 and node_offset[edge_tgt] != 1

        edge_src = edge_src + node_offset[edge_src]
        edge_tgt = edge_tgt + node_offset[edge_tgt]

        processed_edges.append((edge_type, edge_src, edge_tgt))

    return {
        'node2bpe': '\t'.join([' '.join([str(x) for x in n2b]) for n2b in node2bpe]),
        'bpe2word': ' '.join([str(x) for x in bpe2word]),
        'node_target': ' '.join([str(x) for x in node_target]),
        'bpe.source': ' '.join([str(x) for x in all_bpes]),
        'node_cnt': ' '.join([str(x) for x in node_cnt]),
        'normal_edge': '\t'.join(['{} {} {}'.format(et, src, tgt) for et, src, tgt in processed_edges])
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('focus_jsonl')
    parser.add_argument('graph_jsonl')
    parser.add_argument('out_prefix')
    args = parser.parse_args()

    out_files = {}

    with open(args.focus_jsonl) as ffocus, open(args.graph_jsonl) as fgraph:
        with ProcessPoolExecutor() as executor:
            futures = []
            for focus_line, graph_line in zip(ffocus, fgraph):
                focus_sample = json.loads(focus_line)

                sents_words = [sent['words'] for sent in focus_sample['focus_sents']]
                trimmed_sents_words = []
                for sent_words in sents_words:
                    if len(trimmed_sents_words) + len(sent_words) > MAX_WORDS:
                        break
                    trimmed_sents_words.extend(sent_words)

                graph_sample = json.loads(graph_line)

                futures.append(executor.submit(focus_prediction_graph, trimmed_sents_words, graph_sample))

                if len(futures) == 30000:
                    print('Processing existing 30000 jobs. Pause submitting new jobs.')
                    results = [future.result() for future in futures]

                    all_dict_results = defaultdict(list)
                    for dict_result in results:
                        for k, v in dict_result.items():
                            all_dict_results[k].append(v)

                    for k, vs in all_dict_results.items():
                        if k not in out_files:
                            out_files[k] = open(args.out_prefix + '.' + k, 'w')
                        for v in vs:
                            out_files[k].write(v + '\n')

                    futures = []
                    results = []
                    print('Resume submitting new jobs.')
            if futures:
                print('Processing remaining jobs. No new job to submit.')
                results = [future.result() for future in futures]

                all_dict_results = defaultdict(list)
                for dict_result in results:
                    for k, v in dict_result.items():
                        all_dict_results[k].append(v)

                for k, vs in all_dict_results.items():
                    if k not in out_files:
                        out_files[k] = open(args.out_prefix + '.' + k, 'w')
                    for v in vs:
                        out_files[k].write(v + '\n')
                print('Done.')

    for k, f in out_files.items():
        f.close()


if __name__ == '__main__':
    main()
