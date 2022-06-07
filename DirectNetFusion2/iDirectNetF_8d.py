import torch
import torch.nn as nn
import torch.nn.functional as F
# from Pointnet import PointNet
from Pointnet import PointNet
from operations.Pooling import Pooling
from operations.transform_functions import PCRNetTransform
from operations.dual import dual_quat_to_extrinsic
from DirectNetFusion2.fusion import fusion


class iDirectnetF_8d(nn.Module):
    def __init__(self, feature_model=PointNet(), droput=0.0, pooling='max'):
        super(iDirectnetF_8d, self).__init__()
        self.feature_model = feature_model
        self.pooling = Pooling(pooling)
        # self.fc1 = nn.Linear(self.feature_model.emb_dims * 2, 1024)
        self.fc1 = nn.Linear(1088 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 8)
        # self.fc5 = nn.Linear(1088, 1024)

    def forward(self, template, source, maxIteration=5):
        # template_features = self.feature_model(template)  # [B,1024,N]
        template_features = self.pooling(self.feature_model(template))
        est_R = torch.eye(3).to(template).view(1, 3, 3).expand(template.size(0), 3, 3).contiguous()  # (Bx3x3)
        est_t = torch.zeros(1, 3).to(template).view(1, 1, 3).expand(template.size(0), 1, 3).contiguous()  # (Bx1x3)
        for i in range(maxIteration):
            est_R, est_t, source = self.spam(template_features, source, est_R, est_t)
        result = {'est_R': est_R,  # source -> template 得到估计的旋转矩阵[B,3,3]
                  'est_t': est_t,  # source -> template 得到估计的平移向量[B,1,3]
                  'est_T': PCRNetTransform.convert2transformation(est_R, est_t),
                  # source -> template   #得到估计的变换矩阵[B,4,4]
                  # 'r': template_features - self.source_features,
                  'transformed_source': source}  # 得到两个全局特征的差值[B,feature_shape]
        return result

    def spam(self, template_features, source, est_R, est_t):
        # self.source_feature_1024, self.source_feature_1088 = self.feature_model(source)  # [B,1024,N
        # self.source_feature_1024, self.source_feature_1088 = self.pooling(
        #     self.source_feature_1024), self.pooling(self.source_feature_1088)  # [B,1024]
        # template_feature_1024, template_feature_1088 = self.pooling(
        #     template_features[0]), self.pooling(template_features[1])  # [B,1024]
        # template_fusion, source_fusion = fusion()(
        #     template_feature_1024.unsqueeze(1),
        #     self.source_feature_1024.unsqueeze(1))  # [B,1,1024]
        # template_fusion, source_fusion = template_fusion.squeeze(1), source_fusion.squeeze(1)
        # t_1024 = F.relu(self.fc5(template_feature_1088))  # 1088 -> 1024
        # s_1024 = F.relu(self.fc5(self.source_feature_1088))
        # y = torch.cat([t_1024 + template_fusion, s_1024 + source_fusion], dim=1)  # 2048
        # self.source_features = self.pooling(self.feature_model(source)[0]) template_feature =
        # template_features.unsqueeze(1)  # B,1,1024 source_features = self.source_features.unsqueeze(1)  # B,1,
        # 1024 template_fusion, source_fusion = fusion()(template_feature, source_features)  # B,1,
        # 1024 y = torch.cat([template_fusion.squeeze(1), source_fusion.squeeze(1)], dim=1) y1 =
        # template_fusion.squeeze(1) + source_fusion.squeeze(1) y = torch.cat([template_features,
        # self.source_features, y1], dim=1) y = torch.cat([template_features, self.source_features], dim=1) y =
        # torch.cat([template_features + template_fusion.squeeze(1), self.source_features + source_fusion.squeeze(
        # 1)], dim=1)
        self.source_features = self.pooling(self.feature_model(source))
        template_fusion, source_fusion = fusion()(template_features.unsqueeze(1), self.source_features.unsqueeze(1))
        # y = torch.cat([template_features, self.source_features], dim=1)
        y = torch.cat([template_features + template_fusion.squeeze(1), self.source_features + source_fusion.squeeze(1)],
                      dim=1)
        pose_8d = F.relu(self.fc1(y))
        pose_8d = F.relu(self.fc2(pose_8d))
        pose_8d = F.relu(self.fc3(pose_8d))
        pose_8d = self.fc4(pose_8d)  # [B,8]
        # 得到R
        pose_8d = PCRNetTransform.create_pose_8d(pose_8d)
        R_qe = pose_8d[:, 0:4]
        D_qe = pose_8d[:, 4:]
        # get current R and t
        est_R_temp, est_t_temp = dual_quat_to_extrinsic(R_qe, D_qe)
        # update rotation matrix.
        est_t = torch.bmm(est_R_temp, est_t.permute(0, 2, 1)).permute(0, 2, 1) + est_t_temp
        est_R = torch.bmm(est_R_temp, est_R)
        # Ps' = est_R * Ps + est_t
        source = PCRNetTransform.quaternion_transform2(source, pose_8d, est_t_temp)
        return est_R, est_t, source


if __name__ == '__main__':
    template, source = torch.rand(2, 10, 3).cuda(), torch.rand(2, 10, 3).cuda()
    pn = PointNet(emb_dims=1024)
    net = iDirectnetF_8d(pn)
    net = net.cuda()
    result = net(template, source)
    print(result['est_R'].shape)
    print(result['est_t'].shape)
