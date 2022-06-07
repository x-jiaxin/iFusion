"""
@Author: xjiaxin
@Date: 2022/04/18 14:07
"""
import torch
import torch.nn as nn
from modelFusion.Pointnet import PointNet
from operations.Pooling import Pooling
from operations.transform_functions import PCRNetTransform


# from utils.fusion import fusion


class iPCRNetF(nn.Module):
    def __init__(self, feature_model=PointNet(), droput=0.0, pooling='max'):
        super(iPCRNetF, self).__init__()
        self.template_feature = None
        self.source_features = None
        self.feature_model = feature_model
        self.pooling = Pooling(pooling)

        self.linear = [nn.Linear(self.feature_model.emb_dims * 2, 1024), nn.ReLU(),
                       nn.Linear(1024, 1024), nn.ReLU(),
                       nn.Linear(1024, 512), nn.ReLU(),
                       nn.Linear(512, 512), nn.ReLU(),
                       nn.Linear(512, 256), nn.ReLU()]

        if droput > 0.0:
            self.linear.append(nn.Dropout(droput))
        self.linear.append(nn.Linear(256, 7))

        self.linear = nn.Sequential(*self.linear)
        # self.w1 = nn.Parameter(torch.tensor([1.0]),requires_grad=True)
        # self.w2 = nn.Parameter(torch.tensor([1.0]),requires_grad=True)

    def forward(self, template, source, maxIteration=8):
        est_R = torch.eye(3).to(template).view(1, 3, 3).expand(template.size(0), 3, 3).contiguous()  # (Bx3x3)
        est_t = torch.zeros(1, 3).to(template).view(1, 1, 3).expand(template.size(0), 1, 3).contiguous()  # (Bx1x3)
        self.template_feature = self.pooling(self.feature_model(template))
        if maxIteration == 1:
            est_R, est_t, source = self.getPose(template, source, est_R, est_t)
        else:
            for i in range(maxIteration):
                est_R, est_t, source = self.getPose(template, source, est_R, est_t)
        result = {'est_R': est_R,
                  'est_t': est_t,
                  'est_T': PCRNetTransform.convert2transformation(est_R, est_t),
                  'transformed_source': source}
        return result

    def getPose(self, template, source, est_R, est_t):
        batch_size = source.size(0)
        sourceFeature, templateFeature = self.feature_model(source, template)
        sourceFeatures = self.pooling(sourceFeature)  # 源点云的全局特征
        templateFeatures = self.pooling(templateFeature)  # 源点云的全局特征
        y = torch.cat([templateFeatures, sourceFeatures], dim=1)
        # y = torch.cat([self.w1 * templateFeatures, self.w2 * sourceFeatures], dim=1)
        pose_7d = self.linear(y)
        pose_7d = PCRNetTransform.create_pose_7d(pose_7d)
        # Find current rotation and translation.
        identity = torch.eye(3).to(source).view(1, 3, 3).expand(batch_size, 3, 3).contiguous()
        est_R_temp = PCRNetTransform.quaternion_rotate(identity, pose_7d).permute(0, 2, 1)
        est_t_temp = PCRNetTransform.get_translation(pose_7d).view(-1, 1, 3)
        # update translation matrix.
        est_t = torch.bmm(est_R_temp, est_t.permute(0, 2, 1)).permute(0, 2, 1) + est_t_temp
        # update rotation matrix.
        est_R = torch.bmm(est_R_temp, est_R)
        source = PCRNetTransform.quaternion_transform(source, pose_7d)  # Ps' = est_R*Ps + est_t
        return est_R, est_t, source


if __name__ == '__main__':
    template, source = torch.rand(2, 10, 3), torch.rand(2, 10, 3)
    pn = PointNet()
    net = iPCRNetF(pn)
    result = net(template, source)
    print(result['est_R'].shape)
    print(result['est_t'].shape)
