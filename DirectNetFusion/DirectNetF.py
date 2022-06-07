import torch
import torch.nn as nn
import torch.nn.functional as F
from DirectNetFusion.PointnetFusion import PointNetFusion
from operations.Pooling import Pooling
from operations.transform_functions import PCRNetTransform


class DirectnetF(nn.Module):
    def __init__(self, feature_model=PointNetFusion(), droput=0.0, pooling='max'):
        super(DirectnetF, self).__init__()
        self.feature_model = feature_model
        self.pooling = Pooling(pooling)
        self.fc1 = nn.Linear(self.feature_model.emb_dims * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 6)
        # self.fc4 = nn.Linear(256, 7)

    def forward(self, template, source):
        # 估计的旋转矩阵
        est_R = torch.eye(3).to(template).view(1, 3, 3).expand(template.size(0), 3, 3).contiguous()  # (Bx3x3)
        # 估计的平移向量
        est_t = torch.zeros(1, 3).to(template).view(1, 1, 3).expand(template.size(0), 1, 3).contiguous()  # (Bx1x3)
        templateFeature = self.pooling(self.feature_model(template,tag='template'))
        est_R, est_t, = self.spam(templateFeature, source, est_R, est_t)

        result = {'est_R': est_R,  # source -> template 得到估计的旋转矩阵[B,3,3]
                  'est_t': est_t,  # source -> template 得到估计的平移向量[B,1,3]
                  'est_T': PCRNetTransform.convert2transformation(est_R, est_t)}
        # source -> template   #得到估计的变换矩阵[B,4,4]
        return result

    def spam(self, templateFeature, source, est_R, est_t):
        sourceFeature = self.pooling(self.feature_model(source,tag='source'))
        y = torch.cat([templateFeature, sourceFeature], dim=1)
        pose_6d = F.relu(self.fc1(y))
        pose_6d = F.relu(self.fc2(pose_6d))
        pose_6d = F.relu(self.fc3(pose_6d))
        pose_6d = self.fc4(pose_6d)  # [B,6]
        # pose_7d = self.fc4(pose_6d)  # [B,7]

        # pose_7d = PCRNetTransform.create_pose_7d(pose_7d)
        # # Find current rotation and translation.
        # batch_size = source.shape[0]
        # identity = torch.eye(3).to(source).view(1, 3, 3).expand(batch_size, 3, 3).contiguous()
        # est_R_temp = PCRNetTransform.quaternion_rotate(identity, pose_7d).permute(0, 2, 1)
        # est_t_temp = PCRNetTransform.get_translation(pose_7d).view(-1, 1, 3)
        # # update translation matrix.
        # est_t = torch.bmm(est_R_temp, est_t.permute(0, 2, 1)).permute(0, 2, 1) + est_t_temp
        # # update rotation matrix.
        # est_R = torch.bmm(est_R_temp, est_R)

        # 得到R
        est_R, est_t = PCRNetTransform.create_pose_6d(pose_6d) 
        # 返回列表：估计的旋转矩阵，估计的平移向量
        return est_R, est_t


if __name__ == '__main__':
    template, source = torch.rand(10, 10, 3), torch.rand(10, 10, 3)
    pn = PointNetFusion()
    net = DirectnetF(pn)
    result = net(template, source)
    print(result['est_R'].shape)
    print(result['est_t'].shape)
    print(result['est_T'].shape)
    # print(result['r'].shape)
    # print(result['transformed_source'].shape)
    # print('-'*50)
