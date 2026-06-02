# Copyright (c) OpenMMLab. All rights reserved.
from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
from mmengine.model import BaseModule

from mmaction.registry import MODELS


@MODELS.register_module()
class SkeletonCNN1D(BaseModule):
    """1D-CNN backbone for skeleton sequences.

    The input follows the GCN skeleton format:
    ``(N, M, T, V, C)`` where ``M`` is the number of persons, ``T`` is the
    temporal length, ``V`` is the number of joints and ``C`` is the coordinate
    dimension. The backbone flattens ``V * C`` into temporal features and
    applies Conv1d over the time axis.
    """

    def __init__(self,
                 in_channels: int,
                 num_joints: int,
                 num_person: int = 1,
                 hidden_channels: Sequence[int] = (64, 128, 64),
                 kernel_size: int = 3,
                 pool_kernel_size: int = 2,
                 dropout: float = 0.0,
                 data_bn: bool = True,
                 init_cfg: Optional[Union[dict, list]] = None) -> None:
        super().__init__(init_cfg=init_cfg)

        self.in_channels = in_channels
        self.num_joints = num_joints
        self.num_person = num_person
        self.out_channels = hidden_channels[-1]

        input_channels = num_joints * in_channels
        self.data_bn = (nn.BatchNorm1d(input_channels)
                        if data_bn else nn.Identity())

        layers = []
        current_channels = input_channels
        padding = kernel_size // 2
        for out_channels in hidden_channels:
            layers.extend([
                nn.Conv1d(
                    current_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=pool_kernel_size)
            ])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_channels = out_channels

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward skeleton input.

        Args:
            x (torch.Tensor): Input tensor with shape ``(N, M, T, V, C)``.

        Returns:
            torch.Tensor: Feature tensor with shape ``(N, M, C_out, T_out, 1)``,
            which is compatible with ``GCNHead``.
        """
        N, M, T, V, C = x.shape
        assert V == self.num_joints, (
            f'Expected {self.num_joints} joints, but got {V}.')
        assert C == self.in_channels, (
            f'Expected {self.in_channels} channels, but got {C}.')

        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = self.conv(x)
        x = x.view(N, M, self.out_channels, x.size(-1), 1)
        return x
