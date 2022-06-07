import os
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset.ModelNet40 import ModelNet40, RegistrationData
from losses.chamfer_distance import ChamferDistanceLoss
from Pointnet import PointNet
from iPCRNetF import iPCRNetF
from operations.transform_functions import PCRNetTransform
from mse import compute_metrics, summary_metrics

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# torch.cuda.set_device(1)
BATCH_SIZE = 32
START_EPOCH = 0
MAX_EPOCHS = 200
device = torch.device("cuda:0")
pretrained = ""  # 是否有训练过的模型可用
resume = ""  # 最新的检查点文件
exp_name = "ipcr8_FusionGlobalAdd_1-alpha"

dirName = os.path.join(
    os.path.dirname(__file__), os.pardir, "checkpoints3", exp_name, "models"
)
if not os.path.exists(dirName):
    os.makedirs(dirName)


def train_one_epoch(device, model, train_loader, optimizer):
    model.train()
    train_loss = 0.0
    count = 0
    for i, data in enumerate(tqdm(train_loader)):
        template, source, igt, gtR, gtt = data
        template = template.to(device)  # [B,N,3]
        source = source.to(device)  # [B,N,3]
        igt = igt.to(device)
        gtR = gtR.to(device)
        gtt = gtt.to(device)
        source = source - torch.mean(source, dim=1, keepdim=True)
        template = template - torch.mean(template, dim=1, keepdim=True)
        gtT = PCRNetTransform.convert2transformation(gtR, gtt)
        output = model(template, source)
        loss_val = ChamferDistanceLoss()(template,output['transformed_source'])
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        train_loss += loss_val.item()
        # 批次加一
        count += 1
    train_loss = float(train_loss) / count
    return train_loss


def test_one_epoch(device, model, test_loader):
    model.eval()
    test_loss = 0.0
    r_mse, t_mse = [], []
    count = 0
    for i, data in enumerate(tqdm(test_loader)):
        template, source, igt, gtR, gtt = data
        template = template.to(device)
        source = source.to(device)
        igt = igt.to(device)
        gtR = gtR.to(device)
        gtt = gtt.to(device)
        source = source - torch.mean(source, dim=1, keepdim=True)
        template = template - torch.mean(template, dim=1, keepdim=True)
        output = model(template, source)
        gtT = PCRNetTransform.convert2transformation(gtR, gtt)
        loss_val = ChamferDistanceLoss()(template,output['transformed_source'])
        test_loss += loss_val.item()

        est_R = output['est_R']  # B*3*3
        est_t = output['est_t']  # B*1*3
        cur_r_mse, cur_t_mse = compute_metrics(est_R, est_t, gtR, gtt)
        r_mse.append(cur_r_mse)
        t_mse.append(cur_t_mse)
        count += 1

    r_mse, t_mse = summary_metrics(r_mse, t_mse)
    r_rmse = np.sqrt(r_mse)
    t_rmse = np.sqrt(t_mse)
    print(f"RMSE(R):{r_rmse},RMSE(t):{t_rmse}")
    test_loss = float(test_loss) / count
    return test_loss


def train(model, train_loader, test_loader):
    startTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    learnable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(learnable_params)
    if checkpoint is not None:
        min_loss = checkpoint["min_loss"]
        optimizer.load_state_dict(checkpoint["optimizer"])
    best_test_loss = np.inf
    for epoch in range(START_EPOCH, MAX_EPOCHS):
        train_loss = train_one_epoch(device, model, train_loader, optimizer)
        test_loss = test_one_epoch(device, model, test_loader)

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            snap = {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "min_loss": best_test_loss,
                "optimizer": optimizer.state_dict(),
            }
            torch.save(
                snap, os.path.join(dirName, "best_model_snap.t7"))
            torch.save(
                model.module.state_dict(),
                os.path.join(dirName, "best_model.t7"),
            )
            torch.save(
                model.module.feature_model.state_dict(),
                os.path.join(dirName, "best_ptnet_model.t7"),
            )

        torch.save(
            snap,
            os.path.join(dirName, "model_snap.t7")
        )
        torch.save(
            model.module.state_dict(),
            os.path.join(dirName, "model.t7")
        )
        torch.save(
            model.module.feature_model.state_dict(),
            os.path.join(dirName, "ptnet_model.t7"),
        )

        print(
            "EPOCH:{},Training Loss:{},Testing Loss:{},Best Loss:{}".format(epoch + 1, train_loss, test_loss,
                                                                            best_test_loss))
    endTime = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
    print(f"Over!\nStart time:{startTime}\nEnd time:{endTime}")


def getModel():
    feature = PointNet(emb_dims=1024)
    return iPCRNetF(feature_model=feature)


if __name__ == "__main__":
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    np.random.seed(1234)
    trainset = RegistrationData("PCRNet", ModelNet40(train=True))
    testset = RegistrationData("PCRNet", ModelNet40(train=False))
    trainloader = DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4
    )
    testloader = DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False, num_workers=4
    )
    model = getModel()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)
    checkpoint = None
    if resume:
        checkpoint = torch.load(resume)
        START_EPOCH = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model"])
    if pretrained:
        model.load_state_dict(torch.load(pretrained, map_location="cpu"))
    train(model, trainloader, testloader)
