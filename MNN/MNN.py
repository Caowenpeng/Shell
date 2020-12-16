import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim


class MNNs(nn.Module):

    def __init__(self, input_size=912, mlp_hsize=100, rnn_hsize=200):
        super(MNNs, self).__init__()
        # MLP
        self.mlp = nn.Sequential(
            #nn.Dropout2d(0.2),
            nn.Linear(input_size, mlp_hsize), #输入数据的个数
            nn.Dropout2d(0.5),
            nn.ReLU(),
            #nn.Dropout2d(0.3),
            nn.Linear(mlp_hsize, rnn_hsize),
            nn.Dropout2d(0.5),
            nn.ReLU(),
            #nn.Dropout2d(0.5),
            nn.Dropout2d(0.7),
        )
        # LSTM
        self.rnn = nn.LSTM(
            input_size=40,
            hidden_size=rnn_hsize,  # rnn hidden unit
            num_layers=1,  # 有几层 RNN layers
            batch_first=True,  # input & output 会是以 batch size 为第一维度的特征集 e.g. (batch, time_step, input_size)

        )
        self.out = nn.Linear(rnn_hsize, 5)  # 输出层

    def forward(self, x, time_step=5):
        mlp_out = self.mlp(x)
        mlp_out = mlp_out.view(mlp_out.shape[0], time_step, -1)
        r_out, (h_n, h_c) = self.rnn(mlp_out, None)  # None 表示 hidden state 会用全0的 state
        out = self.out(r_out[:, -1, :])  # (batch, time step, input)  -1代表是最后的out put

        #pred_y = F.log_softmax(out, dim=1)
        # pred_y = torch.max(F.softmax(out), 1)[1].squeeze()
        # pred_y=pred_y.view(-1, 1).type(torch.LongTensor)
        return out






