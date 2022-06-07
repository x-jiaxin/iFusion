import torch.nn as nn
import torch
import torch.nn.functional as F

from DirectNetFusion.fusion import fusion


class PointNet(nn.Module):
    def __init__(self, emb_dims=1024, input_shape="bnc"):
        # emb_dims:			Embedding Dimensions for PointNet.
        # input_shape:		Shape of Input Point Cloud (b: batch, n: no of points, c: channels)
        super(PointNet, self).__init__()
        self.pointT = None
        self.pointS = None
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "Allowed shapes are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        self.input_shape = input_shape
        self.emb_dims = emb_dims

        # 待测试没有Batchnorm
        self.conv1T = nn.Conv1d(3, 64, 1)
        self.conv1S = nn.Conv1d(3, 64, 1)
        self.conv2T = nn.Conv1d(64, 64, 1)
        self.conv2S = nn.Conv1d(64, 64, 1)
        self.conv3T = nn.Conv1d(64, 64, 1)
        self.conv3S = nn.Conv1d(64, 64, 1)
        self.conv4T = nn.Conv1d(64, 128, 1)
        self.conv4S = nn.Conv1d(64, 128, 1)
        self.conv5T = nn.Conv1d(128, self.emb_dims, 1)
        self.conv5S = nn.Conv1d(128, self.emb_dims, 1)

    def forward(self, *args):
        self.pointS, self.pointT = args
        if self.input_shape == "bnc":
            num_points = self.pointS.shape[1]
            self.pointS = self.pointS.permute(0, 2, 1)
            self.pointT = self.pointT.permute(0, 2, 1)
        else:
            num_points = self.pointS.shape[2]
        featureS = F.relu(self.conv1S(self.pointS))  # 64
        featureT = F.relu(self.conv1T(self.pointT))
        featureS = fusion()(featureS, featureT)

        featureS = F.relu(self.conv2S(featureS))  # 64
        featureT = F.relu(self.conv2T(featureT))
        featureS = fusion()(featureS, featureT)

        featureS = F.relu(self.conv3S(featureS))  # 64
        featureT = F.relu(self.conv3T(featureT))
        featureS = fusion()(featureS, featureT)

        featureS = F.relu(self.conv4S(featureS))  # 128
        featureT = F.relu(self.conv4T(featureT))
        featureS = fusion()(featureS, featureT)

        featureS = F.relu(self.conv5S(featureS))  # 1024
        featureT = F.relu(self.conv5T(featureT))
        featureS = fusion()(featureS, featureT)

        return featureS, featureT


if __name__ == "__main__":
    device = torch.device("cuda")
    x = torch.rand((2, 2, 3))
    y = torch.rand((2, 2, 3))
    x = x.to(device)
    y = y.to(device)
    model = PointNet()
    model = model.to(device)
    outputS, outputT = model(x, y)
    print("S:", outputS.shape)
    print("T:", outputT.shape)
