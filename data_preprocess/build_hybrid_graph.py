import json
import argparse
import re
import numpy as np
from concurrent.futures import ProcessPoolExecutor

ARG_RE = re.compile(r'B-ARG[0-9]+')

MAX_WORD = 256


with open('stopwords') as f:
    stopwords = set([line.strip() for line in f])


class Node:
    def __init__(self, start_idx, end_idx):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.coref_mentions = []
        self.children = []  # a list of node instances
        self.heads = []  # a list of [node, dep]


def get_depth(node, span_nodes, root_node, depth, visited):
    if node not in span_nodes:
        return depth
    if node in visited:
        return depth
    visited.append(node)
    current_depth = depth
    if not node.heads and node != root_node:
        return 100000
    for head_node, dep in node.heads:
        if not dep.startswith('SRL'):
            depth = max(get_depth(head_node, span_nodes, root_node, current_depth + 1, visited), depth)
    return depth


def get_span_head(span, nodes, root_node):
    span_nodes = [nodes[i] for i in range(span[0], span[1])]
    span_node_depth = [get_depth(node, span_nodes, root_node, 0, []) for node in span_nodes]
    if min(span_node_depth) >= 100000:
        return None
    return span[0] + np.argmin(span_node_depth)


def walk_tree_add_nodes_edges(root_node, head_node, head_id, nodes, edges, visited):
    if root_node in visited:
        return
    visited.append(root_node)
    node_id = len(nodes)
    nodes.append((None, root_node.start_idx, root_node.end_idx, []))
    if head_node is not None and head_id is not None:
        relation = [dep for node, dep in root_node.heads if node == head_node]
        for rel in relation:
            edges.append((rel, head_id, node_id))
    for child_node in root_node.children:
        walk_tree_add_nodes_edges(child_node, root_node, node_id, nodes, edges, visited)


def build_graph(stanford, allen_srl, raw):
    nodes = []
    edges = []

    root_nodes = []
    doc_words_dp = []
    doc_pos = []
    doc_lemma = []
    word2sent = []

    sent_boundary = []
    for sent_id, sent_dp in enumerate(stanford['sents_parsed']):
        sent_words = [x['word'] for x in sent_dp['tokens']]
        if len(doc_words_dp) + len(sent_words) > MAX_WORD:
            break

        sent_pos = [x['pos'] for x in sent_dp['tokens']]
        sent_lemma = [x['lemma'] for x in sent_dp['tokens']]
        doc_pos.extend(sent_pos)
        doc_lemma.extend(sent_lemma)
        word2sent.extend([sent_id] * len(sent_words))
        sent_nodes = [Node(len(doc_words_dp) + i, len(doc_words_dp) + i + 1) for i, _ in enumerate(sent_words)]

        root_node = None
        for enhanced_dep in sent_dp['enhancedPlusPlusDependencies']:
            governor_id = enhanced_dep['governor'] - 1
            dependent_id = enhanced_dep['dependent'] - 1
            dep = enhanced_dep['dep']
            if dep == 'ROOT':
                assert root_node is None
                root_node = dependent_id
                continue
            if dep in ['case', 'mark', 'cc', 'cc:preconj', 'aux', 'aux:pass', 'cop', 'det', 'discourse', 'expl', 'det:predet', 'punct', 'ref']:
                continue

            if sent_nodes[dependent_id] in sent_nodes[governor_id].children:
                continue

            sent_nodes[governor_id].children.append(sent_nodes[dependent_id])
            sent_nodes[dependent_id].heads.append([sent_nodes[governor_id], dep])

        assert root_node is not None
        root_node = sent_nodes[root_node]

        root_nodes.append(root_node)

        sent_srl = allen_srl['sents_srl'][sent_id]
        if sent_srl is not None:
            srl_words = sent_srl['words']
            sent_word_offset = [0 for _ in srl_words]

            if sent_words != srl_words:
                srl_c2w = []
                for i, word in enumerate(srl_words):
                    srl_c2w.extend([i] * len(word))
                dp_c2w = []
                for i, word in enumerate(sent_words):
                    dp_c2w.extend([i] * len(word))

                sent_char_srl = ''.join(srl_words)
                sent_char_dp = ''.join(sent_words)
                assert len(srl_c2w) == len(sent_char_srl)
                assert len(dp_c2w) == len(sent_char_dp)

                srl_i = 0
                dp_i = 0
                while dp_i < len(sent_char_dp) and srl_i < len(sent_char_srl):
                    if sent_char_dp[dp_i] == sent_char_srl[srl_i]:
                        dp_wid = dp_c2w[dp_i]
                        srl_wid = srl_c2w[srl_i]
                        sent_word_offset[srl_wid] = dp_wid - srl_wid
                        dp_i += 1
                        srl_i += 1
                    elif not sent_char_dp[dp_i].strip():
                        dp_i += 1
                    elif not sent_char_srl[srl_i].strip():
                        dp_wid = dp_c2w[dp_i]
                        srl_wid = srl_c2w[srl_i]
                        sent_word_offset[srl_wid] = dp_wid - srl_wid - 1
                        srl_i += 1
                    else:
                        raise NotImplementedError

            for relation in sent_srl['verbs']:
                arg_roles = set([tag[2:] for tag in relation['tags'] if ARG_RE.fullmatch(tag)])

                roles2span = {}
                current_tag = None
                start_idx = 0
                for i, tag in enumerate(relation['tags']):
                    if tag[0] == 'B':
                        if current_tag is None:
                            current_tag = tag[2:]
                            start_idx = i
                        else:
                            current_subtag = current_tag.split('-')
                            if len(current_subtag) > 1 and current_subtag[1].startswith('ARG') and current_subtag[
                                1] not in arg_roles:
                                current_tag = tag[2:]
                                start_idx = i
                            else:
                                roles2span[current_tag] = (start_idx + sent_word_offset[start_idx], i + sent_word_offset[i - 1])
                                current_tag = tag[2:]
                                start_idx = i
                    elif tag[0] == 'O':
                        if current_tag is not None:
                            current_subtag = current_tag.split('-')
                            if len(current_subtag) > 1 and current_subtag[1].startswith('ARG') and current_subtag[
                                1] not in arg_roles:
                                current_tag = None
                            else:
                                roles2span[current_tag] = (start_idx + sent_word_offset[start_idx], i + sent_word_offset[i - 1])
                                current_tag = None
                if current_tag is not None:
                    current_subtag = current_tag.split('-')
                    if len(current_subtag) > 1 and current_subtag[1].startswith('ARG') and current_subtag[
                        1] not in arg_roles:
                        current_tag = None
                    else:
                        roles2span[current_tag] = (start_idx + sent_word_offset[start_idx], len(relation['tags']) + sent_word_offset[len(relation['tags']) - 1])
                        current_tag = None
                role2head = {role: get_span_head(span, sent_nodes, root_node) for role, span in roles2span.items()}

                for role in role2head:
                    if role2head[role] is None:
                        continue
                    role_split = role.split('-')
                    if len(role_split) > 1 and role_split[1].startswith('ARG') and role2head[role_split[1]] is not None:
                        sent_nodes[role2head[role_split[1]]].children.append(sent_nodes[role2head[role]])
                        sent_nodes[role2head[role]].heads.append((sent_nodes[role2head[role_split[1]]], 'SRL-' + role_split[0]))
                    elif role != 'V':
                        if 'V' in role2head and role2head['V'] is not None:
                            sent_nodes[role2head[role]].heads.append((sent_nodes[role2head['V']], 'SRL-' + role))
                            sent_nodes[role2head['V']].children.append(sent_nodes[role2head[role]])

        doc_words_dp.extend(sent_words)

        sent_boundary.append(len(doc_words_dp))

    for root_node in root_nodes:
        walk_tree_add_nodes_edges(root_node, None, None, nodes, edges, [])

    tok2nodeid = [None for _ in enumerate(doc_words_dp)]
    for i, node in enumerate(nodes):
        for j in range(node[1], node[2]):
            tok2nodeid[j] = i

    # connect sentences
    for i in range(len(sent_boundary) - 1):
        prev_sent_boundary = sent_boundary[i - 1] if i > 0 else 0
        curr_sent_boundary = sent_boundary[i]
        next_sent_boundary = sent_boundary[i + 1]

        current_sent_last = None
        for nodeid in tok2nodeid[prev_sent_boundary: curr_sent_boundary]:
            if nodeid is not None:
                current_sent_last = nodeid
        next_sent_first = None
        for nodeid in tok2nodeid[curr_sent_boundary: next_sent_boundary]:
            if nodeid is not None:
                next_sent_first = nodeid
                break

        assert current_sent_last is not None and next_sent_first is not None

        edges.append(('FOLLOW', current_sent_last, next_sent_first))

    # merge corefered nodes
    coref = stanford['coref']
    doc_words_coref = coref['document']

    doc_word_offset = [0 for _ in doc_words_coref]

    if doc_words_dp != doc_words_coref:  # need to set offset
        srl_c2w = []
        for i, word in enumerate(doc_words_dp):
            srl_c2w.extend([i] * len(word))
        coref_c2w = []
        for i, word in enumerate(doc_words_coref):
            coref_c2w.extend([i] * len(word))

        doc_characters_srl = ''.join(doc_words_dp)
        doc_characters_coref = ''.join(doc_words_coref)
        assert len(srl_c2w) == len(doc_characters_srl)
        assert len(coref_c2w) == len(doc_characters_coref)

        srl_i = 0
        coref_i = 0
        while srl_i < len(doc_characters_srl) and coref_i < len(doc_characters_coref):
            if doc_characters_srl[srl_i] == doc_characters_coref[coref_i]:
                srl_wid = srl_c2w[srl_i]
                coref_wid = coref_c2w[coref_i]
                doc_word_offset[coref_wid] = srl_wid - coref_wid
                srl_i += 1
                coref_i += 1
            elif not doc_characters_srl[srl_i].strip():
                srl_i += 1
            elif not doc_characters_coref[coref_i].strip():
                srl_wid = srl_c2w[srl_i]
                coref_wid = coref_c2w[coref_i]
                doc_word_offset[coref_wid] = srl_wid - coref_wid - 1
                coref_i += 1
            else:
                raise NotImplementedError

    disable_nodes = []
    for cluster in coref['clusters_heads']:
        cluster_node_ids = []
        for span_head in cluster:
            span_head = span_head + doc_word_offset[span_head - 1] - 1
            if span_head < len(tok2nodeid) and tok2nodeid[span_head] is not None:
                cluster_node_ids.append(tok2nodeid[span_head])
        if len(cluster_node_ids) <= 1:
            continue
        cluster_node_ids_set = set(cluster_node_ids)
        if len(cluster_node_ids) > 1:
            first_node_id = min(cluster_node_ids)
            first_node = nodes[first_node_id]
            first_node[3].append((first_node[1], first_node[2]))
            for node_id in cluster_node_ids:
                if node_id != first_node_id:
                    coref_node = nodes[node_id]
                    first_node[3].append((coref_node[1], coref_node[2]))
                    disable_nodes.append(node_id)
            edges = [(rel, first_node_id if head_id in cluster_node_ids_set else head_id,
                      first_node_id if node_id in cluster_node_ids_set else node_id) for rel, head_id, node_id in edges]

    for i, node in enumerate(nodes):
        if i in disable_nodes:
            continue
        if node[3]:
            node_word = {(doc_lemma[start], doc_pos[start]) for start, _ in node[3] if doc_lemma[start].lower() not in stopwords}
        else:
            node_word = {(doc_lemma[node[1]], doc_pos[node[1]])} if doc_lemma[node[1]].lower() not in stopwords else set()

        for j, else_node in enumerate(nodes[i+1:], i + 1):
            if j in disable_nodes:
                continue
            if else_node[3]:
                else_node_word = {(doc_lemma[start], doc_pos[start]) for start, _ in else_node[3] if doc_lemma[start].lower() not in stopwords}
            else:
                else_node_word = {(doc_lemma[else_node[1]], doc_pos[else_node[1]])} if doc_lemma[
                                                                            else_node[1]].lower() not in stopwords else set()
            if len(node_word & else_node_word) > 0:
                if not node[3]:
                    node[3].append((node[1], node[2]))
                if else_node[3]:
                    node[3].extend(else_node[3])
                else:
                    node[3].append((else_node[1], else_node[2]))
                disable_nodes.append(j)
                node_word = node_word | else_node_word

                edges = [(rel, i if head_id == j else head_id,
                      i if node_id == j else node_id) for rel, head_id, node_id in edges]

    # deduplicate
    edges = list(set(edges))

    return {'id': raw['id'], 'answer': raw['answer'], 'question': raw['question'], 'words': doc_words_dp, 'nodes': nodes, 'edges': edges,
            'disable_nodes': disable_nodes, 'pos': doc_pos, 'lemmas': doc_lemma}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stanford_out_jsonl')
    parser.add_argument('srl_out_jsonl')
    parser.add_argument('raw_jsonl')
    parser.add_argument('out_jsonl')
    args = parser.parse_args()

    dp_stanford = []
    with open(args.stanford_out_jsonl) as f:
        for line in f:
            dp = json.loads(line)
            dp_stanford.append(dp)

    dp_srl = []
    with open(args.srl_out_jsonl) as f:
        for line in f:
            dp_srl.append(json.loads(line))

    dp_raw = []
    with open(args.raw_jsonl) as f:
        for line in f:
            dp = json.loads(line)
            dp_raw.append(dp)

    graphs = []
    with ProcessPoolExecutor() as executor:
        for stanford, srl, raw in zip(dp_stanford, dp_srl, dp_raw):
            sents_parsed = []
            accumulated = 0
            for sent_id, sent_dp in enumerate(stanford['sents_parsed']):
                sent_words = [x['word'] for x in sent_dp['tokens']]
                if accumulated + len(sent_words) > MAX_WORD:
                    break
                accumulated += len(sent_words)
                sents_parsed.append(sent_dp)
            stanford['sents_parsed'] = sents_parsed
            srl['sents_srl'] = srl['sents_srl'][:len(sents_parsed)]
            graphs.append(executor.submit(build_graph, stanford, srl, raw))
        graphs = [graph.result() for graph in graphs]

    with open(args.out_jsonl, 'w') as f:
        for graph in graphs:
            f.write(json.dumps(graph) + '\n')


if __name__ == '__main__':
    main()