"""
@Date: 2022/05/27 15:17
"""
import time

import numpy as np
import torch
from visdom import Visdom

if __name__ == '__main__':
    # 实例化一个窗口
    wind = Visdom()
    # 初始化窗口信息
    wind.line([0.],  # Y的第一个点的坐标
              [0.],  # X的第一个点的坐标
              win='train_loss',  # 窗口的名称
              opts=dict(title='train_loss')  # 图像的标例
              )
    # 更新数
    for step in range(10):
        # 随机获取loss,这里只是模拟实现
        loss = np.random.randn() * 0.5 + 2
        wind.line([loss], [step], win='train_loss', update='append')
        time.sleep(5)
