# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from sklearn.metrics import average_precision_score
import numpy as np
from collections import OrderedDict


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss.masked_fill_(pad_mask, 0.0)
        smooth_loss.masked_fill_(pad_mask, 0.0)
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1.0 - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion("joint_generation_loss")
class JointGenerationCriterion(FairseqCriterion):
    def __init__(
        self,
        task,
        sentence_avg,
        label_smoothing,
        ignore_prefix_size=0,
        report_accuracy=False,
        focus_weight=1.,
        temp_weight=1.
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.eps = label_smoothing
        self.ignore_prefix_size = ignore_prefix_size
        self.report_accuracy = report_accuracy
        self.focus_weight = focus_weight
        self.temp_weight = temp_weight

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0., type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        parser.add_argument('--report-accuracy', action='store_true',
                            help='report accuracy metric')
        parser.add_argument('--ignore-prefix-size', default=0, type=int,
                            help='Ignore first N tokens')
        parser.add_argument('--focus-weight', default=1, type=float)
        parser.add_argument('--temp-weight', default=1, type=float)
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output, gate = model(**sample["net_input"])
        gate = gate.squeeze(-1)
        bsz = gate.size(0)

        target = sample["target"]
        target_mask = target.ne(self.padding_idx)
        gate_target = sample["graph_labels"]

        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=False)
        loss = loss.squeeze(-1)

        loss = loss.sum(-1) / target_mask.sum(-1) * self.temp_weight
        nll_loss = nll_loss.sum()

        gate_mask = gate_target.ne(-1)
        mask_gate_target = gate_target.masked_fill(~gate_mask, 0)

        gate_loss = F.binary_cross_entropy(gate.float(), mask_gate_target.float(), reduction='none')
        gate_loss = gate_loss.masked_fill(~gate_mask, 0.)
        gate_loss = gate_loss.sum(-1) / gate_mask.sum(-1)

        loss = loss.sum() + gate_loss.sum() * self.focus_weight

        auc = 0.
        nauc = 0
        for i in range(bsz):
            try:
                aps = average_precision_score(gate_target[i][gate_mask[i]].long().tolist(),
                                              gate[i][gate_mask[i]].tolist(), average='weighted')
                auc += np.nan_to_num(aps)
                nauc += 1
            except ValueError:
                pass

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "nll_loss": nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "auc": auc,
            "nauc": nauc
        }

        return loss, sample_size, logging_output

    def get_lprobs_and_target(self, model, net_output, sample, non_flat=False):
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        target = model.get_targets(sample, net_output)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                target = target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                target = target[self.ignore_prefix_size :, :].contiguous()
        if non_flat:
            return lprobs, target
        return lprobs.view(-1, lprobs.size(-1)), target.view(-1)

    def compute_loss(self, model, net_output, sample, reduce=True):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample, non_flat=True)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce,
        )
        return loss, nll_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample, non_flat=True)
        n_correct = torch.sum(
            lprobs.argmax(-1)[:, 0].eq(target[:, 0])
        )
        total = lprobs.size(0)
        return n_correct, total

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        auc = sum(log.get('auc', 0) for log in logging_outputs)
        nauc = sum(log.get('nauc', 0) for log in logging_outputs)

        metrics.log_scalar('auc', (auc / nauc) if nauc != 0 else 0, round=4)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        metrics.log_scalar(
            "overall", ((auc / nauc) if nauc != 0 else 0) - nll_loss_sum / ntokens / math.log(2),
            round=4
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
