import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Pointnet import PointNet
from operations.Pooling import Pooling
from operations.transform_functions import PCRNetTransform


class Directnet(nn.Module):
    def __init__(self, feature_model=PointNet(), droput=0.0, pooling='max'):
        super(Directnet, self).__init__()
        self.feature_model = feature_model
        self.pooling = Pooling(pooling)
        self.fc1 = nn.Linear(self.feature_model.emb_dims * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 6)

    def forward(self, template, source):
        # 估计的旋转矩阵
        est_R = torch.eye(3).to(template).view(1, 3, 3).expand(template.size(0), 3, 3).contiguous()  # (Bx3x3)
        # 估计的平移向量
        est_t = torch.zeros(1, 3).to(template).view(1, 1, 3).expand(template.size(0), 1, 3).contiguous()  # (Bx1x3)
        template_features = self.pooling(self.feature_model(template))  # 模板点云的全局特征
        est_R, est_t, = self.spam(template_features, source, est_R, est_t)

        result = {'est_R': est_R,  # source -> template 得到估计的旋转矩阵[B,3,3]
                  'est_t': est_t,  # source -> template 得到估计的平移向量[B,1,3]
                  'est_T': PCRNetTransform.convert2transformation(est_R, est_t),
                  # source -> template   #得到估计的变换矩阵[B,4,4]
                  'r': template_features - self.source_features}  # 得到两个全局特征的差值[B,feature_shape]
        return result

    def spam(self, template_features, source, est_R, est_t):
        self.source_features = self.pooling(self.feature_model(source))  # 源点云的全局特征
        y = torch.cat([template_features, self.source_features], dim=1)
        pose_6d = F.relu(self.fc1(y))
        pose_6d = F.relu(self.fc2(pose_6d))
        pose_6d = F.relu(self.fc3(pose_6d))
        pose_6d = self.fc4(pose_6d)  # [B,6]
        # 得到R
        est_R, est_t = PCRNetTransform.create_pose_6d(pose_6d)  #
        # 返回列表：估计的旋转矩阵，估计的平移向量
        return est_R, est_t


if __name__ == '__main__':
    template, source = torch.rand(10, 10, 3), torch.rand(10, 10, 3)
    pn = PointNet()
    net = Directnet(pn)
    result = net(template, source)
    print(result['est_R'].shape)
    print(result['est_t'].shape)
    print(result['est_T'].shape)
    # print(result['r'].shape)
    # print(result['transformed_source'].shape)
    # print('-'*50)
