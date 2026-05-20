# Copyright (c) OpenMMLab. All rights reserved.
import time
from typing import Sequence

import torch
from mmengine.runner import TestLoop, ValLoop, autocast
from mmengine.runner.loops import _parse_losses, _update_losses

from mmaction.registry import LOOPS


def _sync_cuda() -> None:
    """Synchronize CUDA kernels before reading wall-clock time."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class _TimedEvalMixin:
    """Mixin that accumulates model inference time during evaluation.

    The measured section is ``model.val_step``/``model.test_step`` only, so it
    excludes dataloader time and metric computation. It includes the model data
    preprocessor and CTC greedy decoding because those are part of MMEngine's
    prediction step.
    """

    def _reset_timing(self) -> None:
        self._inference_time_sec = 0.0
        self._num_inference_samples = 0

    def _record_timing(self, elapsed: float, outputs: Sequence) -> None:
        self._inference_time_sec += elapsed
        self._num_inference_samples += len(outputs)

    def _timing_metrics(self) -> dict:
        if self._num_inference_samples == 0:
            time_per_sample = 0.0
            fps = 0.0
        else:
            time_per_sample = (self._inference_time_sec /
                               self._num_inference_samples)
            fps = (self._num_inference_samples / self._inference_time_sec
                   if self._inference_time_sec > 0 else 0.0)

        return dict(
            inference_time=self._inference_time_sec,
            inference_time_per_sample=time_per_sample,
            inference_fps=fps,
            num_inference_samples=self._num_inference_samples)


@LOOPS.register_module()
class TimedValLoop(_TimedEvalMixin, ValLoop):
    """Validation loop that logs per-sample inference time."""

    def run(self) -> dict:
        """Launch validation with inference timing."""
        self._reset_timing()
        self.runner.call_hook('before_val')
        self.runner.call_hook('before_val_epoch')
        self.runner.model.eval()

        self.val_loss.clear()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        metrics.update(self._timing_metrics())

        if self.val_loss:
            loss_dict = _parse_losses(self.val_loss, 'val')
            metrics.update(loss_dict)

        self.runner.call_hook('after_val_epoch', metrics=metrics)
        self.runner.call_hook('after_val')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]):
        """Iterate one validation mini-batch."""
        self.runner.call_hook(
            'before_val_iter', batch_idx=idx, data_batch=data_batch)
        with autocast(enabled=self.fp16):
            _sync_cuda()
            start = time.perf_counter()
            outputs = self.runner.model.val_step(data_batch)
            _sync_cuda()
            elapsed = time.perf_counter() - start

        outputs, self.val_loss = _update_losses(outputs, self.val_loss)
        self._record_timing(elapsed, outputs)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_val_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)


@LOOPS.register_module()
class TimedTestLoop(_TimedEvalMixin, TestLoop):
    """Test loop that logs per-sample inference time."""

    def run(self) -> dict:
        """Launch test with inference timing."""
        self._reset_timing()
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        self.test_loss.clear()
        for idx, data_batch in enumerate(self.dataloader):
            self.run_iter(idx, data_batch)

        metrics = self.evaluator.evaluate(len(self.dataloader.dataset))
        metrics.update(self._timing_metrics())

        if self.test_loss:
            loss_dict = _parse_losses(self.test_loss, 'test')
            metrics.update(loss_dict)

        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

    @torch.no_grad()
    def run_iter(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one test mini-batch."""
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        with autocast(enabled=self.fp16):
            _sync_cuda()
            start = time.perf_counter()
            outputs = self.runner.model.test_step(data_batch)
            _sync_cuda()
            elapsed = time.perf_counter() - start

        outputs, self.test_loss = _update_losses(outputs, self.test_loss)
        self._record_timing(elapsed, outputs)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
