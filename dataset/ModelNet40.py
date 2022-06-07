import torch
from torch.utils.data import Dataset
import numpy as np
import os
import h5py
import glob
from torch.utils.data import DataLoader
from operations.transform_functions import PCRNetTransform

# import open3d as o3d

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# DATA_DIR = os.path.join(BASE_DIR, os.pardir, 'data')
# if not os.path.exists(DATA_DIR):
#     os.mkdir(DATA_DIR)


DATA_DIR = '/media/newamax/94d146aa-e21d-4f2d-ae9d-1f5444870820/xjx/PointCloudData'


def load_data(train):
    if train:
        partition = 'train'
    else:
        partition = 'test'

    # 存放数据和对应标签
    Data = []
    Label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_{}*.h5'.format(partition))):
        with h5py.File(h5_name, 'r') as file:
            data = file['data'][:].astype('float32')
            label = file['label'][:].astype('int64')
            Data.append(data)
            Label.append(label)

    Data = np.concatenate(Data, axis=0)  # (9840, 2048, 3)  9840个样本，每个样本2048个点，每个点3维
    Label = np.concatenate(Label, axis=0)  # (9840, 1)
    return Data, Label


def read_classed():
    # 读取所有类的类名
    with open(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'shape_names.txt'), 'r') as file:
        shape_name = file.read()
        shape_name = np.array(shape_name.split('\n')[:-1])
        return shape_name


class ModelNet40(Dataset):
    def __init__(self, train=True, num_points=1024, randomize_data=False):
        super(ModelNet40, self).__init__()
        self.data, self.labels = load_data(train)
        self.shapes = read_classed()
        self.num_points = num_points
        self.randomize_data = randomize_data

    def __getitem__(self, index):
        if self.randomize_data:
            current_points = self.randomize(index)  # 从该实例2048个点随机采样了1024个点
        else:
            current_points = self.data[index].copy()  # 直接使用该实例2048个点

        current_points = torch.from_numpy(current_points).float()
        label = torch.from_numpy(self.labels[index]).type(torch.LongTensor)
        return current_points, label  # 返回该实例（实例从2048个点随机采样了1024个点）以及标签

    def __len__(self):
        return self.data.shape[0]

    def get_shape(self, label):
        return self.shapes[label]

    def randomize(self, index):
        point_index = np.arange(0, self.num_points)  # 在0~num_points范围内生成索引
        np.random.shuffle(point_index)  # 打乱索引
        return self.data[index, point_index].copy()


class RegistrationData(Dataset):
    def __init__(self, algorithm='iPCRNet', data_class=ModelNet40(), is_testing=False):
        super(RegistrationData, self).__init__()
        self.algorithm = algorithm
        self.is_testing = is_testing
        self.data_class = data_class
        if self.algorithm == 'PCRNet' or self.algorithm == 'iPCRNet':
            self.transforms = PCRNetTransform(len(data_class), angle_range=45, translation_range=1)

    def __getitem__(self, index):
        # template_pc:模版点云 source_pc:源点云
        template_pc, label = self.data_class[index]
        self.transforms.index = index
        # 调用__call__对模版点云变换后获得源点云
        source_pc = self.transforms(template_pc)
        igt = self.transforms.igt
        if self.is_testing:
            # 返回列表：模板点云，源点云，真实的变换矩阵7d，真实的旋转矩阵，真实的平移向量
            return template_pc, source_pc, igt, self.transforms.igt_rotation, self.transforms.igt_translation
        else:
            # 返回列表：模板点云，源点云，真实的变换矩阵7d
            return template_pc, source_pc, igt, self.transforms.igt_rotation, self.transforms.igt_translation

    def __len__(self):
        return len(self.data_class)


if __name__ == '__main__':
    data = RegistrationData('PCRNet', ModelNet40(train=False))
    test_loader = DataLoader(data, batch_size=1, shuffle=False)
    for i, data in enumerate(test_loader):
        print(data[0].shape)
        print(data[1].shape)
        print(data[2].shape)
        break
