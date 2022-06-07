import torch
import torch.nn as nn
import math


class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.x2h = nn.Linear(input_size, 3 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 3 * hidden_size)
        self.reset_parameters()
        self.h2o = nn.Linear(hidden_size, input_size)

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

    def forward(self, x, hidden):
        x = x.permute(0, 2, 1)
        hidden = hidden.permute(0, 2, 1)

        b, n, d = x.size()
        x = x.reshape(-1, d)
        hidden = hidden.reshape(-1, hidden.size(2))

        gate_x = self.x2h(x)
        gate_h = self.h2h(hidden)

        gate_x = gate_x.squeeze()
        gate_h = gate_h.squeeze()

        i_r, i_i, i_n = gate_x.chunk(3, 1)
        h_r, h_i, h_n = gate_h.chunk(3, 1)

        # 公式1
        resetgate = torch.sigmoid(i_r + h_r)
        # 公式2
        inputgate = torch.sigmoid(i_i + h_i)
        # 公式3
        newgate = torch.tanh(i_n + (resetgate * h_n))
        # 公式4，不过稍微调整了一下公式形式
        hy = newgate + inputgate * (hidden - newgate)
        out = self.h2o(hy)
        out = out.reshape(b, d, n)
        return out


if __name__ == '__main__':
    h = torch.rand(10, 3, 1024)
    x = torch.rand(10, 64, 1024)
    gru = GRUCell(64, 3)
    res = gru(x, h)
    print(res.size())
