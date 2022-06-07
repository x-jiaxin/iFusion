"""
@Author: 幻想
@Date: 2022/05/03 00:29
"""
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils.mse import compute_metrics, summary_metrics
from train import exp_name, MAX_EPOCHS, getModel,PCRNetTransform,ModelNet40,RegistrationData,FrobeniusNormLoss

BATCH_SIZE = 32
EVAL = False
START_EPOCH = 0
pretrained = f'../checkpointspcr/{exp_name}/models/best_model.t7'  # 使用最好的模型参数测试


def rmse(pts, T, ptt, T_gt):
    pts_pred = pts @ T[:, :3, :3].transpose(1, 2) + T[:, :3, 3].unsqueeze(1)
    pts_gt = pts @ T_gt[:, :3,
                   :3].transpose(1, 2) + T_gt[:, :3, 3].unsqueeze(1)
    return torch.norm(pts_pred - ptt, dim=2).mean(dim=1)


def test_one_epoch(device, model, test_loader):
    model.eval()
    test_loss = 0.0
    count = 0
    errors = []
    r_mse, t_mse = [], []
    rmses = 0

    for i, data in enumerate(tqdm(test_loader)):
        template, source, gtT, gtR, gtt = data
        template = template.to(device)
        source = source.to(device)
        gtT = gtT.to(device)
        gtR = gtR.to(device)
        gtt = gtt.to(device)
        gtt = gtt - torch.mean(source, dim=1).unsqueeze(1)
        source = source - torch.mean(source, dim=1, keepdim=True)
        template = template - torch.mean(template, dim=1, keepdim=True)

        output = model(template, source)
        est_R = output['est_R']  # B*3*3
        est_t = output['est_t']  # B*1*3
        gtT = PCRNetTransform.convert2transformation(gtR, gtt)
        cur_r_mse, cur_t_mse = compute_metrics(est_R, est_t, gtR, gtt)
        T_rmse = rmse(source, output['est_T'], template, gtT)
        r_mse.append(cur_r_mse)
        t_mse.append(cur_t_mse)
        rmses += T_rmse.sum().item()

        loss_val = FrobeniusNormLoss()(output['est_T'], gtT)
        test_loss += loss_val.item()
        count += 1
    r_mse, t_mse = summary_metrics(r_mse, t_mse)
    r_rmse = np.sqrt(r_mse)
    t_rmse = np.sqrt(t_mse)
    rmses = rmses / count
    test_loss = float(test_loss) / count

    print(exp_name, 'finished!')

    return test_loss, t_rmse, r_rmse, rmses


if __name__ == '__main__':
    testset = RegistrationData(
        'PCRNet', ModelNet40(train=False), is_testing=True)
    testloader = DataLoader(testset, batch_size=BATCH_SIZE,
                            shuffle=False, drop_last=False, num_workers=4)
    device = torch.device('cpu')
    model = getModel()
    model = model.to(device)
    if pretrained:
        model.load_state_dict(torch.load(pretrained, map_location='cpu'))
    # model.to(device)
    test_loss, translation_error, rotation_error, Transformation_error = test_one_epoch(
        device, model, testloader)
    print("Test Loss: {}, Rotation Error: {} & Translation Error: {} & Transformation Error: {}".format(test_loss,
                                                                                                        rotation_error,
                                                                                                        translation_error,
                                                                                                        Transformation_error))
