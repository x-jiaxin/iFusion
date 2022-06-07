import torch.nn as nn
import torch
import torch.nn.functional as F

from modelfusion2.PointnetT import PointNetT
from utils.fusion import fusion


class PointNetS(nn.Module):
    def __init__(self, emb_dims=1024, input_shape="bnc"):
        # emb_dims:			Embedding Dimensions for PointNet.
        # input_shape:		Shape of Input Point Cloud (b: batch, n: no of points, c: channels)
        super(PointNetS, self).__init__()
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "Allowed shapes are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        self.input_shape = input_shape
        self.emb_dims = emb_dims
        self.fusion = fusion()

        # 待测试没有Batchnorm
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, self.emb_dims, 1)

    def forward(self, source, featureT):
        # input_data: 		Point Cloud having shape input_shape.
        # output:			PointNet features (Batch x emb_dims)
        if self.input_shape == "bnc":
            num_points = source.shape[1]
            source = source.permute(0, 2, 1)
        else:
            num_points = source.shape[2]

        output = F.relu(self.conv1(source))
        output = self.fusion(output, featureT[0])
        output = F.relu(self.conv2(output))
        output = self.fusion(output, featureT[1])
        output = F.relu(self.conv3(output))
        output = self.fusion(output, featureT[2])
        output = F.relu(self.conv4(output))
        output = self.fusion(output, featureT[3])
        output = F.relu(self.conv5(output))
        output = self.fusion(output, featureT[4])

        return output


if __name__ == "__main__":
    x = torch.rand((2, 10, 3))
    x1 = torch.rand((2, 10, 3))
    pnT = PointNetT()
    T = pnT(x1)
    pn = PointNetS(emb_dims=1024, input_shape="bnc")
    y, y1 = pn(x, T)
    print("Input Shape of PointNet: ", x.shape, "\nOutput Shape of PointNet: ", y.shape)
    print(y1.shape)
