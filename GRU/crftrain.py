import matplotlib as mpl
mpl.use('Agg')
import torch
import torch.nn as nn
import os
import numpy as np
import pandas as pd
import torch.nn.functional as F
from sklearn import preprocessing
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import SleepMoudel
import random
from h5py import File as h5file
import buf

save_path = "D:/WorkSpace/Shell/GRU/Result/"
def train(LR,batchsize,epochs,list_train,list_test,knum):
    
    rnn=SleepMoudel.RNN2()
    # print(cnn)
    optimizer = torch.optim.Adam(rnn.parameters(), lr=LR,weight_decay=0.01)   # optimize all cnn parameters
    # optimizer1 = torch.optim.Adam(cnn.parameters(), lr=0.0001)   # optimize all cnn parameters
    # loss_func = nn.CrossEntropyLoss()
    total_loss = []
    iteration = []
    total_accuracy = []
    total_train_accuracy = []
    test_total_loss = []
    i=0
    best_accuracy= 0.0
    last_improved = 0
    require_improvement = 2000
    flag = False
    loader = buf.pre_train(list_train, batchsize)
    x_test, y_test = buf.pre_test(list_test, batchsize)
    x_test=x_test.unsqueeze(dim=1)

    y_test=y_test.unsqueeze(1)

    for epoch in range(epochs):
        # output = np.empty(shape=[0,1000])
        # stratnum=0
        for load in range(1):
            # endnum=stratnum+10
            # loader=SleepEEG2.pre_train(list_train[stratnum:endnum],batchsize)
            for b_x, b_y in loader:
                b_x = b_x.unsqueeze(dim=1)
                b_y = b_y.unsqueeze(1)
                if len(b_x) != 1:
                    # b_x=b_x.view(6,-1,400)
                    if len(b_x) % 6 == 0:
                        b_x = b_x.view(6, -1, 912)
                    out = rnn(b_x)
                    # pred_y = torch.max(out, 1)[1].data.numpy()           
                    # prediction = torch.max(F.softmax(out,dim=1), 1)[1]
                    # # print(prediction,prediction.shape)
                    # pred_y = prediction.data.numpy().squeeze()
                    # print(out.shape)
                    loss = - rnn.crf(out, b_y, torch.ones((out.shape[0], out.shape[1]), dtype=torch.bool))
                    # loss = loss_func(out, b_y)   # cross entropy loss

                    total_loss.append(loss)
                    i = i + 1
                    iteration.append(i)
                    optimizer.zero_grad()           # clear gradients for this training step
                    loss.backward(loss)                 # backpropagation, compute gradients
                    optimizer.step()   
                    
                    pred_y= torch.argmax(out, 2).data.numpy()
                    # print(pred_y)
                    train_accuracy = float((pred_y == b_y.data.numpy()).astype(int).sum()) / float(len(b_y))
            
                    # train_accuracy = float((pred_y == np.array(b_y)).astype(int).sum()) / float(len(b_y))
                    total_train_accuracy.append(train_accuracy)
                    # # output=np.concatenate((output, outrnn.data.numpy()), axis = 0)
                    if (epoch+1) % 50 == 0:
                        #newfile = h5file(save_path+'train_pred/result'+str(i)+'.h5', 'w')
                        newfile['/pred']=pred_y
                        newfile['/lable']=b_y.data.numpy()
                        newfile.close()

                    if i % 100 == 0:

                        print('Epoch: ', epoch+1, 'Step: ', i, '| train loss: %.4f' % loss.data.numpy()[-1], '| train accuracy: %.2f'%train_accuracy )
                    
        test_loss = 0
        accuracy = 0
                        
        for testload in range(1):
            # testloader=SleepEEG2.pre_train(list_test[testload:testload+1],batchsize)
            # for testnum,(x_test,y_test) in enumerate(testloader):
            if len(x_test) !=1: 
                # x_test=x_test[:18138,:,:]
                # y_test=y_test[:18138,:]
                
                if len(x_test) !=1:
                    x_test=x_test.view(6,-1,912)
                   
                    test_output = rnn(x_test)
                    test_loss=-rnn.crf(test_output, y_test, torch.zeros((test_output.shape[0],1),dtype=torch.bool))
                    
                    test_total_loss.append(test_loss)
                    # pred_y = torch.max(F.softmax(test_output,dim=1), 1)[1]
                    # pred_y = pred_y.data.numpy().squeeze()
                    
                    pred_y= torch.argmax(test_output, 2).data.numpy()
                    accuracy = float((pred_y == y_test.data.numpy()).astype(int).sum()) / float(len(y_test))
            
                    # torch.save(pred_y,'pred_y2'+str(knum)+str(testload)+'_'+str(testnum)+'.txt')
                    # torch.save(y_test,'y_test2'+str(knum)+str(testload)+'_'+str(testnum)+'.txt')
                    # accuracy =float((pred_y == np.array(y_test)).astype(int).sum()) / float(len(y_test))
                    total_accuracy.append(accuracy)
                    print('Epoch: ', epoch+1, '| test loss: ', test_loss.data.numpy()[-1], '| test accuracy: %.4f'%accuracy)
                    print(pred_y.shape)
                    if (epoch+1) % 2 == 0:
                        newfile = h5file(save_path+'test_pred/testresult'+str(knum)+'.h5', 'w')

                        newfile['/pred'] = pred_y
                        newfile['/lable'] = y_test.data.numpy()
                        newfile.close()
                    torch.cuda.empty_cache()
                            
                    # test_loss /= len(list_test)
                    # accuracy /= len(list_test)

                    # print('Epoch: ', epoch,'| train loss: ' , test_loss, '| test accuracy: ',accuracy)
                    # total_accuracy.append(accuracy) 
                    # test_total_loss.append(test_loss)
                    
                # b_x = b_x.cpu()
                # b_y = b_y.cpu()
            torch.cuda.empty_cache()
            # stratnum=stratnum+10
        
        # torch.save(output, 'knum'+str(knum)+'outputdata'+str(epoch)+'.txt')
        if (epoch+1) % 50 == 0:
           
            # print(total_train_accuracy,now_accuracy,now_loss)
            torch.save(rnn, save_path+'result/gru_Epoch213'+str(knum)+str(epoch)+'.pkl')  # save entire net
            torch.save(total_loss, save_path+'result/gru_Epoch_total_loss213'+str(knum)+str(epoch)+'.txt')
            torch.save(test_total_loss, save_path+'result/gru_loss_Epoch213'+str(knum)+str(epoch)+'.txt')
            torch.save(total_train_accuracy, save_path+'result/gru_Epoch_tracc213'+str(knum)+str(epoch)+'.txt')
            torch.save(total_accuracy, save_path+'result/gru_Epoch_teacc213'+str(knum)+str(epoch)+'.txt')