import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Pointnet import PointNet
from operations.Pooling import Pooling
from operations.transform_functions import PCRNetTransform
from utils.fusion import fusion


class DirectnetFusion(nn.Module):
    def __init__(self, feature_model=PointNet(), droput=0.0, pooling='max'):
        super(DirectnetFusion, self).__init__()
        self.feature_model = feature_model
        self.pooling = Pooling(pooling)
        self.fc1 = nn.Linear(self.feature_model.emb_dims * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 6)

    def forward(self, template, source):
        template_features = self.pooling(self.feature_model(template))
        source_features = self.pooling(self.feature_model(source))
        est_R, est_t = self.spam(template_features, source_features)
        result = {'est_R': est_R,  # source -> template 得到估计的旋转矩阵[B,3,3]
                  'est_t': est_t,  # source -> template 得到估计的平移向量[B,1,3]
                  'est_T': PCRNetTransform.convert2transformation(est_R, est_t),
                  # source -> template   #得到估计的变换矩阵[B,4,4]
                  'r': template_features - source_features}  # 得到两个全局特征的差值[B,feature_shape]
        return result

    def spam(self, template_features, source_features):
        template_features = template_features.unsqueeze(1)
        source_features = source_features.unsqueeze(1)
        template_fusion, source_fusion = fusion()(template_features, source_features)
        y = torch.cat([template_fusion.squeeze(1), source_fusion.squeeze(1)], dim=1)
        pose_6d = F.relu(self.fc1(y))
        pose_6d = F.relu(self.fc2(pose_6d))
        pose_6d = F.relu(self.fc3(pose_6d))
        pose_6d = self.fc4(pose_6d)  # [B,6]
        # 得到R
        est_R, est_t = PCRNetTransform.create_pose_6d(pose_6d)
        return est_R, est_t


if __name__ == '__main__':
    template, source = torch.rand(2, 10, 3), torch.rand(2, 10, 3)
    pn = PointNet()
    net = DirectnetFusion(pn)
    result = net(template, source)
    print(result['est_R'].shape)
    print(result['est_t'].shape)
