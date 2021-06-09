from fairseq import options, tasks, checkpoint_utils, progress_bar, utils

import torch
import logging
import sys


def main():
    logging.basicConfig(
        format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=logging.INFO,
        stream=sys.stdout,
    )
    parser = options.get_generation_parser()
    args = options.parse_args_and_arch(parser)

    task = tasks.setup_task(args)

    logging.info('Load model from {}'.format(args.path))
    models, _ = checkpoint_utils.load_model_ensemble([args.path], task=task)
    model = models[0]

    if args.fp16:
        model.half()
    model.cuda()

    model.eval()

    task.load_dataset(args.gen_subset)

    test_itr = task.get_batch_iterator(
        dataset=task.dataset(args.gen_subset),
        max_sentences=16
    ).next_epoch_itr(shuffle=False)

    progress = progress_bar.build_progress_bar(
        args, test_itr)

    all_probs = []
    for sample in progress:
        sample = utils.move_to_cuda(sample)

        with torch.no_grad():
            net_output, probs = model(**sample['net_input'])

            probs = probs.squeeze(-1)
            bsz = probs.size(0)
            for i in range(bsz):
                id = sample['id'][i]
                length = sample['graph_labels'][i].ge(0).sum()

                prob = probs[i, :length]

                all_probs.append((id.item(), prob.tolist()))

    all_probs = sorted(all_probs, key=lambda x: x[0])
    all_probs = [x[1] for x in all_probs]
    with open(args.results_path, 'w') as f:
        for pred in all_probs:
            f.write(' '.join(['{:.5f}'.format(p) for p in pred]) + '\n')


if __name__ == '__main__':
    main()
