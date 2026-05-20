# Copyright (c) OpenMMLab. All rights reserved.
import pytest
import torch

from mmaction.evaluation import WERMetric, calculate_corpus_wer, calculate_wer
from mmaction.registry import METRICS
from mmaction.structures import ActionDataSample


@pytest.mark.parametrize(
    'reference,hypothesis,substitutions,deletions,insertions,wer', [
        ([1, 2, 3], [1, 4, 3], 1, 0, 0, 1 / 3),
        ([1, 2, 3], [1, 3], 0, 1, 0, 1 / 3),
        ([1, 2, 3], [1, 9, 2, 3], 0, 0, 1, 1 / 3),
        ([1, 2, 3], [1, 9, 3, 4], 1, 0, 1, 2 / 3),
    ])
def test_calculate_wer(reference, hypothesis, substitutions, deletions,
                       insertions, wer):
    result = calculate_wer(reference, hypothesis)
    assert result['substitutions'] == substitutions
    assert result['deletions'] == deletions
    assert result['insertions'] == insertions
    assert result['num_ref_tokens'] == len(reference)
    assert result['wer'] == pytest.approx(wer)


def test_calculate_corpus_wer():
    result = calculate_corpus_wer([
        ([1, 2, 3], [1, 4, 3]),
        ([1, 2, 3], [1, 9, 2, 3]),
    ])
    assert result['substitutions'] == 1
    assert result['deletions'] == 0
    assert result['insertions'] == 1
    assert result['num_ref_tokens'] == 6
    assert result['wer'] == pytest.approx(2 / 6)


def test_empty_reference():
    assert calculate_wer([], [])['wer'] == 0.0
    result = calculate_wer([], [1, 2])
    assert result['wer'] == 2.0
    assert result['insertions'] == 2
    assert result['num_ref_tokens'] == 0


def test_wer_metric():
    metric = METRICS.build(dict(type='WERMetric'))
    data_samples = []
    for gt_gloss, pred_gloss in [([1, 2, 3], [1, 4, 3]),
                                 ([1, 2, 3], [1, 3])]:
        sample = ActionDataSample()
        sample.set_field(torch.tensor(gt_gloss), 'gt_gloss')
        sample.set_field(pred_gloss, 'pred_gloss')
        data_samples.append(sample.to_dict())

    metric.process(None, data_samples)
    result = metric.compute_metrics(metric.results)
    assert result['substitutions'] == 1
    assert result['deletions'] == 1
    assert result['insertions'] == 0
    assert result['num_ref_tokens'] == 6
    assert result['wer'] == pytest.approx(2 / 6)


def test_wer_metric_with_action_data_sample():
    metric = WERMetric()
    sample = ActionDataSample()
    sample.set_field(torch.tensor([1, 2, 3]), 'gt_gloss')
    sample.set_field([1, 4, 3], 'pred_gloss')
    metric.process(None, [sample])
    result = metric.compute_metrics(metric.results)
    assert result['substitutions'] == 1
    assert result['wer'] == pytest.approx(1 / 3)


def test_wer_metric_blank_idx():
    metric = WERMetric(blank_idx=4)
    sample = ActionDataSample()
    sample.set_field(torch.tensor([1, 4, 2]), 'gt_gloss')
    sample.set_field([1, 4, 9, 2], 'pred_gloss')
    metric.process(None, [sample.to_dict()])
    result = metric.compute_metrics(metric.results)
    assert result['insertions'] == 1
    assert result['num_ref_tokens'] == 2
    assert result['wer'] == pytest.approx(1 / 2)
