"""
@Date: 2022/05/27 16:08
"""
import matplotlib.pyplot as plt
import torch

x = torch.arange(0, 10, 0.5)
train_loss = torch.sigmoid(x)
test_loss = torch.rand(20)
fig, ax = plt.subplots()
ax.plot(x, train_loss, label='train')
ax.plot(x, test_loss, label='test')
ax.plot(x, test_loss + 1, label='three')

ax.legend(title='')
ax.set(xlabel='epoch')
ax.set(ylabel='CDLoss')
ax.autoscale(tight=True)
fig.savefig('demo1.jpg', dpi=500)
