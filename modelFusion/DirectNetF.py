"""
@Author: 幻想
@Date: 2022/05/02 14:50
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from Pointnet import PointNet
from fusion import fusion
from operations.Pooling import Pooling
from operations.transform_functions import PCRNetTransform


class DirectnetF1(nn.Module):
    def __init__(self, feature_model=PointNet(), droput=0.0, pooling='max'):
        super(DirectnetF1, self).__init__()
        self.source_features = None
        self.feature_model = feature_model
        self.pooling = Pooling(pooling)
        self.fc1 = nn.Linear(self.feature_model.outdim * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 6)
        # self.w1 = nn.Parameter(torch.tensor([1.0]),requires_grad=True)
        # self.w2 = nn.Parameter(torch.tensor([1.0]),requires_grad=True)

    def forward(self, template, source):
        est_R, est_t = self.getPose(template, source)

        return {'est_R': est_R,  # source -> template 得到估计的旋转矩阵[B,3,3]
                'est_t': est_t,  # source -> template 得到估计的平移向量[B,1,3]
                'est_T': PCRNetTransform.convert2transformation(est_R, est_t)
                # source -> template   #得到估计的变换矩阵[B,4,4]
                }

    def getPose(self, template, source):
        sourceFeature, templateFeature = self.feature_model(source, template)
        # sourceFeatures = self.pooling(sourceFeature)  # 源点云的全局特征
        # templateFeatures = self.pooling(templateFeature)  # 源点云的全局特征
        # y = torch.cat([self.w1 * templateFeatures, self.w2 * sourceFeatures], dim=1)
        # y = torch.cat([templateFeatures, sourceFeatures], dim=1)

        sourceFeatures = self.pooling(sourceFeature).unsqueeze(1)  # 源点云的全局特征
        templateFeatures = self.pooling(templateFeature).unsqueeze(1)  # 源点云的全局特征
        template_fusion, source_fusion = fusion()(templateFeatures, sourceFeatures)
        y = torch.cat([template_fusion.squeeze(1), source_fusion.squeeze(1)], dim=1)
        pose_6d = F.relu(self.fc1(y))
        pose_6d = F.relu(self.fc2(pose_6d))
        pose_6d = F.relu(self.fc3(pose_6d))
        pose_6d = self.fc4(pose_6d)  # [B,6]
        # 得到R
        est_R, est_t = PCRNetTransform.create_pose_6d(pose_6d)  #
        # 返回列表：估计的旋转矩阵，估计的平移向量
        return est_R, est_t


if __name__ == '__main__':
    template, source = torch.rand(2, 2, 3), torch.rand(2, 2, 3)
    device = torch.device("cuda")
    template, source = template.to(device), source.to(device)
    pn = PointNet()
    model = DirectnetF1(pn).to(device)
    result = model(template, source)
    print(result['est_R'].shape)
    print(result['est_t'].shape)
    print(result['est_T'].shape)
    # print(result['r'].shape)
    # print(result['transformed_source'].shape)
    # print('-'*50)
