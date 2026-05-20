# Copyright (c) OpenMMLab. All rights reserved.
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch
from mmengine.evaluator import BaseMetric

from mmaction.registry import METRICS

Token = Union[int, str]


def _to_token_list(sequence: Union[torch.Tensor, Sequence[Token]]) -> List[Token]:
    """Convert a tensor or sequence to a plain Python token list."""
    if isinstance(sequence, torch.Tensor):
        sequence = sequence.detach().cpu().flatten().tolist()
    return list(sequence)


def _remove_blank(sequence: List[Token], blank_idx: Optional[int]) -> List[Token]:
    """Remove blank tokens when a blank index is explicitly configured."""
    if blank_idx is None:
        return sequence
    return [token for token in sequence if token != blank_idx]


def _has_field(data_sample: Any, field: str) -> bool:
    """Return whether a dict or data sample contains a field."""
    if isinstance(data_sample, dict):
        return field in data_sample
    return hasattr(data_sample, field)


def _get_field(data_sample: Any, field: str) -> Any:
    """Get a field from a dict or data sample."""
    if isinstance(data_sample, dict):
        return data_sample[field]
    return getattr(data_sample, field)


def calculate_wer(reference: Union[torch.Tensor, Sequence[Token]],
                  hypothesis: Union[torch.Tensor, Sequence[Token]],
                  blank_idx: Optional[int] = None) -> Dict[str, Union[float, int]]:
    """Calculate WER and S/D/I counts for one token sequence pair.

    Args:
        reference (list[int] | list[str] | Tensor): Ground-truth sequence.
        hypothesis (list[int] | list[str] | Tensor): Predicted sequence.
        blank_idx (int, optional): Blank index to remove before evaluation.

    Returns:
        dict: WER statistics with keys ``wer``, ``substitutions``,
        ``deletions``, ``insertions`` and ``num_ref_tokens``.

    Notes:
        When the reference length is zero, the usual WER denominator is zero.
        If both sequences are empty, WER is defined as 0. If the hypothesis is
        non-empty, WER is set to the number of insertions so the metric remains
        finite and still penalizes extra predicted tokens.
    """
    ref = _remove_blank(_to_token_list(reference), blank_idx)
    hyp = _remove_blank(_to_token_list(hypothesis), blank_idx)
    num_ref_tokens = len(ref)

    if num_ref_tokens == 0:
        insertions = len(hyp)
        return dict(
            wer=0.0 if insertions == 0 else float(insertions),
            substitutions=0,
            deletions=0,
            insertions=insertions,
            num_ref_tokens=0)

    # DP cell stores: (edit_cost, substitutions, deletions, insertions).
    # Candidate order gives deterministic ties: match > substitution > deletion
    # > insertion. For equal edit cost, the first candidate is kept.
    dp: List[List[Tuple[int, int, int, int]]] = [
        [(0, 0, 0, 0) for _ in range(len(hyp) + 1)]
        for _ in range(len(ref) + 1)
    ]

    for i in range(1, len(ref) + 1):
        cost, sub, dele, ins = dp[i - 1][0]
        dp[i][0] = (cost + 1, sub, dele + 1, ins)
    for j in range(1, len(hyp) + 1):
        cost, sub, dele, ins = dp[0][j - 1]
        dp[0][j] = (cost + 1, sub, dele, ins + 1)

    for i in range(1, len(ref) + 1):
        for j in range(1, len(hyp) + 1):
            candidates: List[Tuple[int, int, int, int]] = []

            diag_cost, diag_sub, diag_del, diag_ins = dp[i - 1][j - 1]
            if ref[i - 1] == hyp[j - 1]:
                candidates.append((diag_cost, diag_sub, diag_del, diag_ins))
            else:
                candidates.append(
                    (diag_cost + 1, diag_sub + 1, diag_del, diag_ins))

            del_cost, del_sub, del_del, del_ins = dp[i - 1][j]
            candidates.append((del_cost + 1, del_sub, del_del + 1, del_ins))

            ins_cost, ins_sub, ins_del, ins_ins = dp[i][j - 1]
            candidates.append((ins_cost + 1, ins_sub, ins_del, ins_ins + 1))

            dp[i][j] = min(candidates, key=lambda item: item[0])

    edits, substitutions, deletions, insertions = dp[len(ref)][len(hyp)]
    assert edits == substitutions + deletions + insertions
    return dict(
        wer=edits / num_ref_tokens,
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        num_ref_tokens=num_ref_tokens)


def calculate_corpus_wer(
        pairs: Sequence[Tuple[Union[torch.Tensor, Sequence[Token]],
                              Union[torch.Tensor, Sequence[Token]]]],
        blank_idx: Optional[int] = None) -> Dict[str, Union[float, int]]:
    """Calculate corpus-level WER by accumulating S/D/I/N."""
    total_substitutions = 0
    total_deletions = 0
    total_insertions = 0
    total_ref_tokens = 0

    for reference, hypothesis in pairs:
        result = calculate_wer(reference, hypothesis, blank_idx=blank_idx)
        total_substitutions += int(result['substitutions'])
        total_deletions += int(result['deletions'])
        total_insertions += int(result['insertions'])
        total_ref_tokens += int(result['num_ref_tokens'])

    total_errors = total_substitutions + total_deletions + total_insertions
    if total_ref_tokens == 0:
        corpus_wer = 0.0 if total_errors == 0 else float(total_errors)
    else:
        corpus_wer = total_errors / total_ref_tokens

    return dict(
        wer=corpus_wer,
        substitutions=total_substitutions,
        deletions=total_deletions,
        insertions=total_insertions,
        num_ref_tokens=total_ref_tokens)


@METRICS.register_module()
class WERMetric(BaseMetric):
    """Gloss-level Word Error Rate metric for CTC sign recognition.

    The metric reads ``pred_gloss`` from model predictions and ``gt_gloss``
    from packed data samples, then reports corpus-level WER.

    Args:
        blank_idx (int, optional): Blank index to remove from reference and
            hypothesis before scoring. Defaults to None because the CTC greedy
            decoder is expected to remove blanks already.
        collect_device (str): Device name used for collecting results.
        prefix (str, optional): Metric prefix.
    """

    default_prefix: Optional[str] = None

    def __init__(self,
                 blank_idx: Optional[int] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.blank_idx = blank_idx

    def process(self, data_batch: Sequence[Tuple[Any, Dict]],
                data_samples: Sequence[Dict]) -> None:
        """Process one batch of CTC recognition results."""
        for data_sample in data_samples:
            if not _has_field(data_sample, 'pred_gloss'):
                raise KeyError('WERMetric requires `pred_gloss` in outputs.')
            if not _has_field(data_sample, 'gt_gloss'):
                raise KeyError('WERMetric requires `gt_gloss` in outputs.')

            hypothesis = _remove_blank(
                _to_token_list(_get_field(data_sample, 'pred_gloss')),
                self.blank_idx)
            reference = _remove_blank(
                _to_token_list(_get_field(data_sample, 'gt_gloss')),
                self.blank_idx)
            self.results.append(dict(reference=reference, hypothesis=hypothesis))

    def compute_metrics(self, results: List) -> Dict[str, Union[float, int]]:
        """Compute corpus-level WER from processed results."""
        pairs = [(result['reference'], result['hypothesis'])
                 for result in results]
        return calculate_corpus_wer(pairs, blank_idx=None)
