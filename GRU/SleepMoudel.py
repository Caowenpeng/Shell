import torch
from TorchCRF import CRF
import torch.nn as nn
import torch.nn.functional as F
                
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(        
            nn.Conv1d(
                in_channels=6,              
                out_channels=64,            
                kernel_size=5,              
                stride=5,                   
                padding=2,                 
            ),                              
            nn.ReLU(),
            nn.Dropout(0.2),                      
            nn.BatchNorm1d(64),   
        )
        self.conv2 = nn.Sequential(         
            nn.Conv1d(64, 128, 5, 1),     
            nn.ReLU(),
            nn.MaxPool1d(2,2),                
            nn.Dropout(0.2),                      
            nn.BatchNorm1d(128),
        )
        self.conv3 = nn.Sequential(         
            # nn.Conv2d(128,128,1)
            nn.Conv1d(128, 256, 6, 2),     
            nn.ReLU(),
            nn.MaxPool1d(2,2),               
            nn.Dropout(0.2),                     
            nn.BatchNorm1d(256),
        )
        self.conv4 = nn.Sequential(         
            # nn.Conv2d(128,128,1)
            nn.Conv1d(256, 512, 6, 2),     
            nn.ReLU(),
            nn.MaxPool1d(2,2),               
            nn.Dropout(0.2),                     
            nn.BatchNorm1d(512),
        )
        self.conv5 = nn.Sequential(         
            # nn.Conv2d(128,128,1)
            nn.Conv1d(512, 1024, 6, 2),     
            nn.ReLU(),
            nn.MaxPool1d(2,2),               
            nn.Dropout(0.2),                     
            nn.BatchNorm1d(1024),
        )
        self.firstout = nn.Sequential(         
            # nn.Conv2d(128,128,1)
            # nn.Linear(23296,1500),
            nn.Linear(10240,1500),   
            nn.ReLU(),
            nn.Dropout(0.5),                   
            nn.BatchNorm1d(1500),
            nn.Linear(1500,500),
            nn.ReLU(),
            nn.Dropout(0.5),                      
            nn.BatchNorm1d(500),
            # nn.Linear(500,200),
            # nn.ReLU(),
            # nn.Dropout(0.5),                      
            # nn.BatchNorm1d(200),
            # nn.Linear(200,100),
            # nn.ReLU(),
            # nn.Dropout(0.5),                      
            # nn.BatchNorm1d(100),
            
        )
        self.fc=nn.Linear(500,5)
        # self.fc= nn.Linear(1000,5)

    def forward(self, x):
        
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)         
        x=self.firstout(x)
        # print(x.shape)
        out=self.fc(x)
        # x=F.softmax(x, dim=1)
        return x,out   

class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        self.conv1 = nn.Sequential(        
            nn.Conv2d(
                in_channels=1,              
                out_channels=128,            
                kernel_size=(3,50),              
                stride=(1,10),                   
                padding=(1,1),                 
            ),                              
            nn.ReLU(),
            nn.Dropout(0.2),  
            nn.MaxPool2d(2,2),                    
            nn.BatchNorm2d(128),   
        )
        self.conv2 = nn.Sequential(         
            nn.Conv2d(128, 256, (3,5), stride=(1, 1), padding=(1, 1)),     
            nn.ReLU(),
            nn.Dropout(0.2),  
            nn.MaxPool2d(2,2),                      
            nn.BatchNorm2d(256),
        )
        self.conv3 = nn.Sequential(         
            nn.Conv2d(256, 512, (3,5), stride=(1, 1), padding=(1, 1)),     
            nn.ReLU(),
            nn.Dropout(0.2),  
            nn.MaxPool2d((1, 2), stride=(1, 2)),                     
            nn.BatchNorm2d(512),
        )
        self.conv4 = nn.Sequential(         
            nn.Conv2d(512, 512, (3,5), stride=(2, 2), padding=(1, 1)),     
            nn.ReLU(),
            nn.Dropout(0.2),  
            nn.MaxPool2d((1, 2), stride=(1, 2)),                     
            nn.BatchNorm2d(512),
        )
        self.firstout = nn.Sequential(         
            # nn.Conv2d(128,128,1)
            # nn.Linear(23296,1500),
            nn.Linear(11264,4096),   
            nn.ReLU(),
            nn.Dropout(0.5),                   
            nn.BatchNorm1d(4096),
            nn.Linear(4096,1500),
            nn.ReLU(),
            nn.Dropout(0.5),                      
            nn.BatchNorm1d(1500),
            nn.Linear(1500,500),
            nn.ReLU(),
            nn.Dropout(0.5),                      
            nn.BatchNorm1d(500),
        )
        self.fc=nn.Linear(500,5)
        # self.fc= nn.Linear(1000,5)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        x = self.conv4(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)         
        x=self.firstout(x)
        # print(x.shape)
        out=self.fc(x)
        # x=F.softmax(x, dim=1)
        return x,out  

class CNN3(nn.Module):
    def __init__(self):
        super(CNN3, self).__init__()
        self.conv1 = nn.Sequential(        
            nn.Conv1d(
                in_channels=1,
                out_channels=128,            
                kernel_size=7,              
                stride=2,
            ),                              
            nn.ReLU(),
            nn.Dropout(0.1),  
            nn.BatchNorm1d(128),                    
        )
        self.conv2 = nn.Sequential(         
            # nn.Conv1d(128, 128, 7, 2),
            # nn.ReLU(),
            # nn.Dropout(0.1),
            # nn.BatchNorm1d(128),
            nn.Conv1d(128, 128, 7, 2),
            nn.ReLU(),              
            nn.Dropout(0.1),
            nn.BatchNorm1d(128),  
            nn.Conv1d(128, 128, 7, 2),
            nn.ReLU(),              
            nn.Dropout(0.1),
            nn.BatchNorm1d(128),  
            nn.Conv1d(128, 128, 7, 2),
            nn.ReLU(),              
            nn.Dropout(0.1),
            nn.BatchNorm1d(128),  
            nn.Conv1d(128, 128, 7, 2),
            nn.ReLU(),              
            nn.Dropout(0.1),  
            nn.BatchNorm1d(128),                     
        )
        self.conv3 = nn.Sequential(         
            # nn.Conv2d(128,128,1)
            nn.Conv1d(128, 256, 7, 2),
            nn.ReLU(),               
            nn.Dropout(0.1),
            nn.BatchNorm1d(256),  
            nn.Conv1d(256, 256, 5, 2),
            nn.ReLU(),               
            nn.Dropout(0.1), 
            nn.BatchNorm1d(256),  
            nn.Conv1d(256, 256, 5, 2),
            nn.ReLU(),               
            nn.Dropout(0.1),  
            nn.BatchNorm1d(256),                     
           
        )
        self.conv4 = nn.Sequential(         
            nn.Conv1d(256, 256, 5, 2),
            nn.ReLU(),               
            nn.Dropout(0.1), 
            nn.BatchNorm1d(256),  
            nn.Conv1d(256, 256,3, 2),     
            nn.ReLU(),               
            nn.Dropout(0.1), 
            nn.BatchNorm1d(256),  
            nn.Conv1d(256, 256,3, 2),     
            nn.ReLU(),               
            nn.Dropout(0.1),
            nn.BatchNorm1d(256),  
        )
        self.firstout = nn.Sequential(         
            # nn.Conv2d(128,128,1)
            # nn.Linear(23296,1500),
            nn.Linear(512,100),   
            nn.ReLU(),
            nn.Dropout(0.1),                   
        )
        self.fc=nn.Linear(100,5)
        # self.fc= nn.Linear(1000,5)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        # print(x.shape)
        x = self.conv3(x)
        # print(x.shape)
        x = self.conv4(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)         
        x=self.firstout(x)
        # print(x.shape)
        out=self.fc(x)
        # x=F.softmax(x, dim=1)
        return x,out 
'''
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn=nn.LSTM(
            input_size=1412,
            hidden_size=100,         # rnn hidden unit
            num_layers=2,           # number of rnn layer
            batch_first=True,     
            bidirectional=True
            )
        self.out=nn.Linear(200,5)
    def forward(self,x):
        r_out, (h_n, h_c) = self.rnn(x, None)   # None represents zero initial hidden state
        out=self.out(r_out[:,-1,:])
        return out
'''
class RNN(nn.Module):
    def __init__(self):
        super(RNN, self).__init__()
        self.rnn1=nn.GRU(
            input_size=912,
            hidden_size=200,         # rnn hidden unit
            num_layers=2,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            dropout=0.5,
            bidirectional=True,
            
            )
        # self.fe = nn.Linear(400,400)
        self.fc = nn.Linear(400,5)
        # output = nn.Softmax(fc)
        self.crf = CRF(5)

    def forward(self, x):
        r_out, h_n = self.rnn1(x, None)
        # print(r_out.shape)
        # fea=self.fe()
        out=self.fc(r_out)
        out=out.view(-1,5)
        # print(out.shape)
        #out=self.out(r_out)
        fea=r_out.reshape(-1,400)      
        return fea,out

class RNN2(nn.Module):
    def __init__(self):
        super(RNN2, self).__init__()
        self.rnn1=nn.GRU(
            input_size=912,
            hidden_size=100,         # rnn hidden unit
            num_layers=2,           # number of rnn layer
            batch_first=True,       # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
            dropout=0.3,
            bidirectional=True
            )

        self.fc = nn.Linear(200, 5)
        # output = nn.Softmax(fc)
        self.crf = CRF(5)
        
    def forward(self, x):
        r_out, h_n1 = self.rnn1(x, None)
        # r_out1, h_n1 = self.rnn1(x,None)
        # F.log_softmax(self.l5(x), dim=1)
        # print(r_out.shape)
        out=self.fc(r_out)
        out = out.view(-1, 1, 5)
        return out