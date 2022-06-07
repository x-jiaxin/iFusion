import torch.nn as nn
import torch
import torch.nn.functional as F

from utils.fusion import fusion


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
        self.outdim = 1024

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
        # self.w1 = nn.Parameter(torch.tensor([1.0]),requires_grad=True)
        # self.w2 = nn.Parameter(torch.tensor([1.0]),requires_grad=True)

    def forward(self, *args):
        self.pointS, self.pointT = args
        if self.input_shape == "bnc":
            num_points = self.pointS.shape[1]
            self.pointS = self.pointS.permute(0, 2, 1)
            self.pointT = self.pointT.permute(0, 2, 1)
        else:
            num_points = self.pointS.shape[2]
        featureS1 = F.relu(self.conv1S(self.pointS))  # 64
        featureT1 = F.relu(self.conv1T(self.pointT ))
        fusionS1, fusionT1 = fusion()(featureS1, featureT1)

        featureS2 = F.relu(self.conv2S(fusionS1))  # 64
        featureT2 = F.relu(self.conv2T(fusionT1))
        fusionS2, fusionT2 = fusion()(featureS2, featureT2)

        featureS3 = F.relu(self.conv3S(fusionS2))  # 64
        featureT3 = F.relu(self.conv3T(fusionT2))
        fusionS3, fusionT3 = fusion()(featureS3, featureT3)

        featureS4 = F.relu(self.conv4S(fusionS3))  # 128
        featureT4 = F.relu(self.conv4T(fusionT3))
        fusionS4, fusionT4 = fusion()(featureS4, featureT4)

        featureS5 = F.relu(self.conv5S(fusionS4))  # 1024
        featureT5 = F.relu(self.conv5T(fusionT4))
        fusionS5, fusionT5 = fusion()(featureS5, featureT5)

        # featureS = torch.cat([featureS1,featureS2,featureS3,featureS4,featureS5], dim=1)
        # featureT = torch.cat([featureT1,featureT2,featureT3,featureT4,featureT5], dim=1)

        # fusionS = torch.cat([fusionS1,fusionS2,fusionS3,fusionS4,fusionS5],dim=1)
        # fusionT = torch.cat([fusionT1,fusionT2,fusionT3,fusionT4,fusionT5],dim=1)

        # fusionS = torch.cat([fusionS3,fusionS5],dim=1)
        # fusionT = torch.cat([fusionT3,fusionT5],dim=1)
        # return fusion()(featureS, featureT)
        return fusionS5, fusionT5


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
