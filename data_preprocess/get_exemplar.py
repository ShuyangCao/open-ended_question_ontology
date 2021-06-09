import argparse
import nltk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('templates')
    parser.add_argument('types')
    parser.add_argument('exemplars')
    parser.add_argument('out')
    args = parser.parse_args()

    with open(args.templates) as f:
        templates = [line.strip() for line in f]

    with open(args.types) as f:
        types = [line.strip() for line in f]

    with open(args.exemplars) as f:
        exemplars = [[[w.lower() for w in t.split()] for t in line.strip().split('\t')] for line in f]

    unique_qts = ['1', '2', '3', '5', '7', '8', '9', '10', '11', '13']
    unique_qts_dict = {qt:i for i, qt in enumerate(unique_qts)}

    selected_exemplars = []
    qt_selected_exemplars = {qt: [] for qt in unique_qts}
    same = 0
    great_diff = 0
    for template, type in zip(templates, types):
        template = template.split()
        if len(template) > 1 and template[1] == "'s":
            template[1] = 'is'

        template = [w.lower() for w in template]
        type_exemplars = exemplars[unique_qts_dict[type]]

        min_edit = 10000
        min_i = None
        for i, exemplar in enumerate(type_exemplars):
            ed = nltk.edit_distance(template, exemplar)
            if ed < min_edit:
                min_edit = ed
                min_i = i
        if min_edit == 0:
            same += 1

        if min_edit > 10:
            great_diff += 1

        qt_selected_exemplars[type].append(min_i)
        selected_exemplars.append(min_i)

    with open(args.out, 'w') as f:
        for selected_exemplar in selected_exemplars:
            f.write(str(selected_exemplar) + '\n')


if __name__ == '__main__':
    main()
