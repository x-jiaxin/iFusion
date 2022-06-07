import torch.nn as nn
import torch
import torch.nn.functional as F


class PointNetT(nn.Module):
    def __init__(self, emb_dims=1024, input_shape="bnc"):
        # emb_dims:			Embedding Dimensions for PointNet.
        # input_shape:		Shape of Input Point Cloud (b: batch, n: no of points, c: channels)
        super(PointNetT, self).__init__()
        if input_shape not in ["bcn", "bnc"]:
            raise ValueError(
                "Allowed shapes are 'bcn' (batch * channels * num_in_points), 'bnc' "
            )
        self.input_shape = input_shape
        self.emb_dims = emb_dims
        self.features = []

        # 待测试没有Batchnorm
        self.conv1 = nn.Conv1d(3, 64, 1)
        self.conv2 = nn.Conv1d(64, 64, 1)
        self.conv3 = nn.Conv1d(64, 64, 1)
        self.conv4 = nn.Conv1d(64, 128, 1)
        self.conv5 = nn.Conv1d(128, self.emb_dims, 1)

    def forward(self, input_data):
        # input_data: 		Point Cloud having shape input_shape.
        # output:			PointNet features (Batch x emb_dims)
        if self.input_shape == "bnc":
            num_points = input_data.shape[1]
            input_data = input_data.permute(0, 2, 1)
        else:
            num_points = input_data.shape[2]

        output1 = F.relu(self.conv1(input_data))
        self.features.append(output1)
        output2 = F.relu(self.conv2(output1))
        self.features.append(output2)
        output3 = F.relu(self.conv3(output2))
        self.features.append(output3)
        output4 = F.relu(self.conv4(output3))
        self.features.append(output4)
        output5 = F.relu(self.conv5(output4))
        self.features.append(output5)

        return self.features


if __name__ == "__main__":
    x = torch.rand((2, 10, 3))
    pn = PointNetT(emb_dims=1024, input_shape="bnc")
    y = pn(x)
    print(y[4].shape)
