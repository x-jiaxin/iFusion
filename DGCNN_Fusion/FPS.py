"""
@Date: 2022/06/07 15:53
"""
import torch


def get_knn_twopc_index(pc1, pc2, k):
    """
    pc1包含于pc2,pc1中的点在pc2中的k近邻点
    Args:
        pc1:
        pc2:
        k:

    Returns:pc1中的点在pc2中的k近邻点下标

    """
    B, N, _ = pc1.shape
    _, M, _ = pc2.shape
    inner = -2 * torch.matmul(pc1, pc2.transpose(2, 1))
    xx = torch.sum(pc1 ** 2, dim=2, keepdim=True).view(B, N, 1)
    yy = torch.sum(pc2 ** 2, dim=2, keepdim=True).view(B, 1, M)
    pairwise_distance = -xx - inner - yy
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def farthest_point_sample(xyz, npoint, RAN=True):
    """
    Input:
        xyz: pointcloud data, [B, N, C]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)  # 用来储存采样index[64,512]
    distance = torch.ones(B, N).to(device) * 1e10  # 用来储存距离[64,2048]
    if RAN:
        farthest = torch.randint(0, 1, (B,), dtype=torch.long).to(device)  # 表示上一次抽样的到点 [64]
    else:
        farthest = torch.randint(1, 2, (B,), dtype=torch.long).to(device)

    batch_indices = torch.arange(B, dtype=torch.long).to(device)  # 一个1-B的整数 [64]
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)  # 找出上一次采样的点
        # [64,1,3]
        dist = torch.sum((xyz - centroid) ** 2, -1)  # [64,2048]
        mask = dist < distance  # 更新每次最小距离
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]  # 求取最大距离 [64]
        # print(farthest)

    return centroids


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def FPS_sample_pc(pc, npoints):
    index = farthest_point_sample(pc, npoints)
    return index_points(pc, index)


if __name__ == '__main__':
    a = torch.rand(10, 100, 3).cuda()
    a = FPS_sample_pc(a, npoints=20)
    print(a.shape)
