"""
Author: xjiaxin
Date: 2022-04-28 19:05:00
"""

import torch
import torch.nn as nn


class fusion(nn.Module):
    """
    input:[B,N,C]
    C:[C,C]
    """

    def __init__(self):
        super().__init__()

    def forward(self, *args):
        f1, f2 = args  # B,N,C
        B, N, C1 = f1.shape
        _, _, C2 = f2.shape
        Cm = torch.matmul(f1.transpose(2, 1), f2)  # [B,C1,C2]
        C_row = torch.sum(Cm, dim=2).reshape(B, -1, C1)
        alpha = torch.sigmoid(C_row)
        # f1_new = torch.mul(f1, alpha)  # 逐元素相乘
        f1_new = f1 + alpha
        # f1_new = torch.mul(f1, alpha)
        return f1_new


if __name__ == '__main__':
    device = torch.device("cuda")
    a = torch.rand(2, 1, 1024).to(device)
    b = torch.rand(2, 1, 1024).to(device)
    fblock = fusion().to(device)
    # a, b = fblock(a, b)
    a = fblock(a, b)
    print(a.shape)
    # print(b.shape)
