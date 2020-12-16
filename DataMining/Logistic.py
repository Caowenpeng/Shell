import torch.nn as nn
import torch


class LogisticRegression(nn.Module):
    def __init__(self):
        super(LogisticRegression,self).__init__()
        self.lr = nn.Linear(5, 1)   #相当于通过线性变换y=x*T(A)+b可以得到对应的各个系数
        self.sm = nn.Sigmoid()  #相当于通过激活函数的变换

    def forward(self, x):

        x = self.lr(x)
        x = self.sm(x)
        return x