import torch.nn as nn
import torch
import torch.nn.functional as F
from DirectNetFusion.fusion import fusion


class PointNetFusion(nn.Module):
    def __init__(self, emb_dims=1024, input_shape="bnc"):
        # emb_dims:			Embedding Dimensions for PointNet.
        # input_shape:		Shape of Input Point Cloud (b: batch, n: no of points, c: channels)
        super(PointNetFusion, self).__init__()
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

    def forward(self, input_data, tag):
        if self.input_shape == "bnc":
            num_points = input_data.shape[1]
            input_data = input_data.permute(0, 2, 1)
        else:
            num_points = input_data.shape[2]
        if tag == 'template':
            self.output1 = F.relu(self.conv1(input_data))

            self.output2 = F.relu(self.conv2(self.output1))

            self.output3 = F.relu(self.conv3(self.output2))

            self.output4 = F.relu(self.conv4(self.output3))

            self.output5 = F.relu(self.conv5(self.output4))

            return self.output5
        elif tag == 'source':
            output = F.relu(self.conv1(input_data))
            output = self.fusion(output, self.output1)
            output = F.relu(self.conv2(output))
            output = self.fusion(output, self.output2)
            output = F.relu(self.conv3(output))
            output = self.fusion(output, self.output3)
            output = F.relu(self.conv4(output))
            output = self.fusion(output, self.output4)
            output = F.relu(self.conv5(output))
            output = self.fusion(output, self.output5)
            return output


if __name__ == '__main__':
    a = torch.rand(2, 10, 3)
    b = torch.rand(2, 10, 3)
    pt = PointNetFusion()
    output1 = pt(a, tag='template')
    output2 = pt(b, tag='source')
    print(output1.shape)
    print(output2.shape)