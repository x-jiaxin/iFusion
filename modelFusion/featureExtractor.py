"""
@Author: 幻想
@Date: 2022/05/02 16:09
"""
import torch.nn as nn


class featureExtractor(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self):
        featureT = None
        featureS = None
        return featureT, featureS
