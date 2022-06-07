"""
@Author: 幻想
@Date: 2022/05/02 14:50
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from modelfusion2.PointNet import PointNet
from operations.Pooling import Pooling
from operations.transform_functions import PCRNetTransform


class DirectnetF(nn.Module):
    def __init__(self, feature_model, droput=0.0, pooling='max'):
        super(DirectnetF, self).__init__()
        self.feature_model = feature_model
        self.pooling = Pooling(pooling)
        self.fc1 = nn.Linear(self.feature_model.emb_dims * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 6)

        # self.w1 = nn.Parameter(torch.tensor([1.0], requires_grad=True))
        # self.w2 = nn.Parameter(torch.tensor([1.0], requires_grad=True))

    def forward(self, template, source):
        # templateFeatures = self.feature_modelT(template)
        est_R, est_t = self.getPose(template, source)

        return {'est_R': est_R,  # source -> template 得到估计的旋转矩阵[B,3,3]
                'est_t': est_t,  # source -> template 得到估计的平移向量[B,1,3]
                'est_T': PCRNetTransform.convert2transformation(est_R, est_t)
                # source -> template   #得到估计的变换矩阵[B,4,4]
                }
    def getPose(self, template, source):
        sourceFeature, templateFeature = self.feature_model(source, template)
        sourceFeatures = self.pooling(sourceFeature)  # 源点云的全局特征
        templateFeature_s = self.pooling(templateFeature)  # 源点云的全局特征
        y = torch.cat([templateFeature_s, sourceFeatures], dim=1)
        pose_6d = F.relu(self.fc1(y))
        pose_6d = F.relu(self.fc2(pose_6d))
        pose_6d = F.relu(self.fc3(pose_6d))
        pose_6d = self.fc4(pose_6d)  # [B,6]
        # 得到R
        est_R, est_t = PCRNetTransform.create_pose_6d(pose_6d)  #
        # 返回列表：估计的旋转矩阵，估计的平移向量
        return est_R, est_t


if __name__ == '__main__':
    template, source = torch.rand(1, 2, 3), torch.rand(1, 2, 3)
    device = torch.device("cuda")
    template, source = template.to(device), source.to(device)
    # pt = PointNetT()
    ps = PointNet()
    model = DirectnetF(ps).to(device)
    result = model(template, source)
    print(result['est_R'].shape)
    print(result['est_t'].shape)
    print(result['est_T'].shape)
