# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.registry import MODELS
from mmaction.utils import ForwardResults, SampleList
from .base import BaseHead


@MODELS.register_module()
class STGCNCTCHead(BaseHead):
    """CTC head for ST-GCN sequence recognition.

    The head preserves the temporal axis from the ST-GCN backbone, averages
    only person and joint axes, and predicts frame-wise gloss logits.

    Example:
        >>> head = STGCNCTCHead(num_classes=67, in_channels=256)
        >>> x = torch.randn(2, 1, 256, 25, 65)
        >>> logits = head(x)
        >>> assert logits.shape == (2, 25, 68)

    Args:
        num_classes (int): Number of gloss classes, excluding CTC blank.
        in_channels (int): Number of channels in input feature.
        hidden_size (int): Hidden size of the BiLSTM. Defaults to 256.
        num_layers (int): Number of BiLSTM layers. Defaults to 2.
        dropout (float): Dropout used between LSTM layers. Defaults to 0.
        blank_idx (int, optional): CTC blank index. Defaults to num_classes.
        init_cfg (dict or list[dict], optional): Initialization config.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 hidden_size: int = 256,
                 num_layers: int = 2,
                 dropout: float = 0.,
                 blank_idx: Optional[int] = None,
                 init_cfg: Optional[Union[Dict, List[Dict]]] = dict(
                     type='Normal', layer='Linear', std=0.01),
                 **kwargs) -> None:
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            init_cfg=init_cfg,
            **kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.blank_idx = num_classes if blank_idx is None else blank_idx
        assert 0 <= self.blank_idx <= num_classes, (
            '`blank_idx` must be in [0, num_classes].')

        lstm_dropout = dropout if num_layers > 1 else 0.
        self.lstm = nn.LSTM(
            input_size=in_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes + 1)
        self.ctc_loss = nn.CTCLoss(
            blank=self.blank_idx, zero_infinity=True, reduction='mean')

    def forward(self, x: torch.Tensor, **kwargs) -> ForwardResults:
        """Forward ST-GCN features.

        Args:
            x (torch.Tensor): ST-GCN features with shape ``(N, M, C, T, V)``
                or ``(N, C, T, V)``.

        Returns:
            torch.Tensor: Frame-wise logits with shape
                ``(N, T, num_classes + 1)``.
        """
        if x.dim() == 5:
            x = x.mean(dim=1)
        elif x.dim() != 4:
            raise ValueError('STGCNCTCHead expects input with shape '
                             '(N, M, C, T, V) or (N, C, T, V), '
                             f'but got {tuple(x.shape)}.')

        assert x.size(1) == self.in_channels, (
            f'Expected {self.in_channels} channels, but got {x.size(1)}.')

        x = x.mean(dim=-1)
        x = x.permute(0, 2, 1).contiguous()
        x, _ = self.lstm(x)
        logits = self.fc(x)
        return logits

    def loss(self, feats: Union[torch.Tensor, Tuple[torch.Tensor]],
             data_samples: SampleList, **kwargs) -> Dict:
        """Perform forward propagation and CTC loss calculation."""
        logits = self(feats, **kwargs)
        return self.loss_by_feat(logits, data_samples)

    def loss_by_feat(self, logits: torch.Tensor,
                     data_samples: SampleList) -> Dict:
        """Calculate CTC loss from frame-wise logits."""
        log_probs = F.log_softmax(logits, dim=-1).permute(1, 0, 2)
        batch_size, input_len = logits.size(0), logits.size(1)
        device = logits.device

        targets = []
        target_lengths = []
        for sample in data_samples:
            if not hasattr(sample, 'gt_gloss'):
                raise AttributeError(
                    'STGCNCTCHead requires each data sample to contain '
                    '`gt_gloss`.')
            target = sample.gt_gloss.to(device=device, dtype=torch.long)
            target = target.flatten()
            assert torch.all((target >= 0) & (target < self.num_classes)), (
                'CTC targets must be valid gloss ids in '
                f'[0, {self.num_classes - 1}] and must not contain the '
                f'blank index ({self.blank_idx}).')
            targets.append(target)
            target_lengths.append(target.numel())

        assert len(data_samples) == batch_size, (
            'The number of data samples must match the logits batch size.')
        target_lengths = torch.tensor(
            target_lengths, dtype=torch.long, device=device)
        input_lengths = torch.full(
            (batch_size, ), input_len, dtype=torch.long, device=device)
        assert torch.all(target_lengths <= input_lengths), (
            'Each CTC target length must be less than or equal to the '
            'corresponding input length. Consider preserving more frames or '
            'reducing ST-GCN temporal downsampling.')

        if targets:
            targets = torch.cat(targets)
        else:
            targets = torch.empty(0, dtype=torch.long, device=device)

        loss_ctc = self.ctc_loss(log_probs, targets, input_lengths,
                                 target_lengths)
        return dict(loss_ctc=loss_ctc)

    def predict(self, feats: Union[torch.Tensor, Tuple[torch.Tensor]],
                data_samples: SampleList, **kwargs) -> SampleList:
        """Perform forward propagation and CTC greedy decoding."""
        logits = self(feats, **kwargs)
        return self.predict_by_feat(logits, data_samples)

    def predict_by_feat(self, logits: torch.Tensor,
                        data_samples: SampleList) -> SampleList:
        """Decode frame-wise logits into gloss id sequences."""
        pred_ids = logits.argmax(dim=-1)
        for data_sample, frame_ids in zip(data_samples, pred_ids):
            decoded = self.ctc_greedy_decode(frame_ids)
            data_sample.set_field(decoded, 'pred_gloss')
        return data_samples

    def ctc_greedy_decode(self, frame_ids: torch.Tensor) -> List[int]:
        """Collapse repeats and remove blanks from a frame-wise prediction."""
        decoded = []
        prev = None
        for idx in frame_ids.detach().cpu().tolist():
            if idx != prev and idx != self.blank_idx:
                decoded.append(idx)
            prev = idx
        return decoded
