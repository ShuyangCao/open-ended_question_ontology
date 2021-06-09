import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('selected_exemplar')
    parser.add_argument('type')
    parser.add_argument('exemplar_file')
    parser.add_argument('out_file')
    args = parser.parse_args()

    with open(args.selected_exemplar) as f:
        selected_exemplars = [int(line.strip()) for line in f]

    with open(args.type) as f:
        types = [line.strip() for line in f]

    with open(args.exemplar_file) as f:
        exemplar_bpes = [line.strip().split('\t') for line in f]

    unique_qts = ['1', '2', '3', '5', '7', '8', '9', '10', '11', '13']
    unique_qts_dict = {qt: i for i, qt in enumerate(unique_qts)}

    with open(args.out_file, 'w') as f:
        for selected_exemplar, type in zip(selected_exemplars, types):
            f.write(exemplar_bpes[unique_qts_dict[type]][selected_exemplar] + '\n')


if __name__ == '__main__':
    main()