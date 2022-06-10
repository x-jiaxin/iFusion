"""
@Date: 2022/06/10 16:56
"""
import torch

from FPS import FPS_sample_pc, index_points, get_knn_twopc_index

a = torch.tensor([i for i in range(30)], dtype=torch.float).view(1, 10, -1)
a1 = FPS_sample_pc(a, 5)
b = get_knn_twopc_index(a1, a, 3)  # id
print(b.shape)
print(a)
print(a1)
print(index_points(a, b))
