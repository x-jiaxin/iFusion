import torch
import torch.nn as nn
import torch.nn.functional as F
from Pointnet import PointNet
from localDGCNN import DGCNN
from operations.Pooling import Pooling
from operations.transform_functions import PCRNetTransform
from operations.dual import dual_quat_to_extrinsic
from fusion import fusion


# from FPS import FPS_process_pc


class iFusionPlus(nn.Module):
    def __init__(self, feature_model=PointNet(), dgcnn=DGCNN(), droput=0.0, pooling='max'):
        super(iFusionPlus, self).__init__()
        self.feature_model = feature_model
        self.dgcnn = dgcnn
        self.pooling = Pooling(pooling)
        self.fc1 = nn.Linear(self.feature_model.emb_dims * 2, 1024)
        # self.fc1 = nn.Linear(1088 * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 8)
        # self.fc5 = nn.Linear(2048, 1024)

    def forward(self, template, source, maxIteration=2):
        template_pt_features = self.pooling(self.feature_model(template))  # [B,1024]
        template_struct_features = self.pooling(self.dgcnn(template))
        # template_struct_features = self.pooling(self.dgcnn(FPS_process_pc(template, npoints=512)))

        # cat->2048
        # template_features = torch.cat([template_pt_features, template_struct_features], dim=1)

        # MLP
        # template_features = F.relu(self.fc5(template_features))

        # add
        template_features = template_pt_features + template_struct_features

        # fusion
        # template_features = template_pt_features + \
        #                     fusion()(template_pt_features.unsqueeze(1), template_struct_features.unsqueeze(1))[
        #                         0].squeeze(1)
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
        source_pt_features = self.pooling(self.feature_model(source))
        source_struct_features = self.pooling(self.dgcnn(source))
        # source_struct_features = self.pooling(self.dgcnn(FPS_process_pc(source, npoints=512)))

        # cat->2048        # 2048->1024
        # self.source_features = torch.cat([source_pt_features, source_struct_features], dim=1)
        # self.source_features = F.relu(self.fc5(self.source_features))

        # add->1024
        self.source_features = source_pt_features + source_struct_features

        # fusion
        # self.source_features = source_pt_features + \
        #                        fusion()(source_pt_features.unsqueeze(1), source_struct_features.unsqueeze(1))[
        #                            0].squeeze(1)

        # fusion
        # template_fusion, source_fusion = fusion()(template_features.unsqueeze(1), self.source_features.unsqueeze(1))
        # y = torch.cat([template_features + template_fusion.squeeze(1), self.source_features +
        #                source_fusion.squeeze(1)], dim=1)

        # no fusion
        y = torch.cat([template_features, self.source_features], dim=1)

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
    template, source = torch.rand(2, 100, 3).cuda(), torch.rand(2, 100, 3).cuda()
    pn = PointNet(emb_dims=1024)
    dg = DGCNN()
    net = iFusionPlus(pn, dgcnn=dg)
    net = net.cuda()
    result = net(template, source)
    print(result['est_R'].shape)
    print(result['est_t'].shape)
