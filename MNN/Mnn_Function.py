
import os

import torch.utils.data as Data
import torch.optim as optim
from MNN import *
import pyedflib
import numpy as np
import h5py
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix as cm
import  matplotlib.pyplot as plt
from tkinter import _flatten
from sklearn.model_selection import KFold
import gc
from ResultCode import cm_plot,kappa

##迭代次数为50
EPOCH = 50
# batch size 为500
BATCH_SIZE = 256
##学习率为0.01
LEARNING_RATE = 0.01
##动量因子为0.9
MOMENTUM = 0.9
##mlp隐藏单元
mlp_hsize = 300
##rnn隐藏单元
rnn_hsize =300
row = 0
use_gpu = False
h5file = "D:/WorkSpace/Shell/data/alldata-90/h5data/"
files = os.listdir(h5file)  # 得到文件夹下的所有文件名称
files_len = len(files)
channels = ['handfea', 'lable']
Max_acc = []


# 保存  训练测试结果
def saveResult(testresult, loss_train, loss_test, acc, koldi, mnn):
    np.savetxt('./result/test_matrix/testresult' + koldi + '.txt', testresult)
    np.savetxt('./result/loss/loss_train' + koldi + '.txt', loss_train)
    np.savetxt('./result/acc/acc_' + koldi + '.txt', acc)
    np.savetxt('./result/loss/loss_test' + koldi + '.txt', loss_test)
    torch.save(mnn, '/net/net' + koldi + '.pkl')  # entire net


# 训练MNN
def mnnTrain(train_x, train_y, loss_all, mnn, loss_func,optimizer):
    train_y_shape = train_y.shape[0]
    train_y = train_y.view(train_y.shape[0]).type(torch.LongTensor)

    if (use_gpu):
        train_x, train_y = train_x.cuda(), train_y.cuda()
    # mnn output
    output = mnn(train_x.type(torch.FloatTensor))
    loss = loss_func(output, train_y)

    # cross entropy loss
    loss_all = loss_all + loss
    optimizer.zero_grad()  # clear gradients for this training step
    loss.backward()  # backpropagation, compute gradients
    optimizer.step()  # apply gradients


# 画图  损失  准确率
def plot_acc_loss(x1, x2, train_acc, loss_train, loss_test, foidi):
    plt.plot(x1, train_acc, 'b.-', label='Train accuracy')
    plt.title('accuracy vs epoches')
    plt.xlabel('epoches')
    plt.ylabel('accuracy')
    plt.legend(["Test accuracy", "Train accuracy"])
    plt.savefig("./result/picture/accuracy" + foidi + ".jpg")
    plt.show()

    plt.subplot(2, 1, 1)
    plt.plot(x2, loss_train, 'r.-', label=u'loss_train')
    plt.subplot(2, 1, 1)
    plt.plot(x2, loss_test, 'b.-', label=u'loss_test')
    plt.title('loss vs epoches')
    plt.ylabel('loss')
    plt.savefig("./result/picture/loss" + foidi + ".jpg")
    plt.show()


def del_collect(del_var):
    del del_var
    gc.collect()


def Set_Dataset(h5file, train_index, type=0):
    global eeg_data, labels
    flag = 0
    ##训练集
    for filei in train_index:
        fileh5 = h5py.File(h5file + files[filei], 'r')
        #    print(fileh5[channels[5]].value.shape)


        if flag == 0:
            eeg_data = torch.cat((torch.from_numpy(fileh5["F4-A1"].value), torch.from_numpy(fileh5["EOG-L"].value)), dim=1)
            labels = torch.from_numpy(fileh5["labels"].value)
            flag = 1

        else:
            eeg_data_new = torch.cat((torch.from_numpy(fileh5["F4-A1"].value), torch.from_numpy(fileh5["EOG-L"].value)), dim=1)
            eeg_data = torch.cat((eeg_data, eeg_data_new), dim=0)
            labels = torch.cat((labels, torch.from_numpy(fileh5["labels"].value)), dim=0)

    if type == 1:
        torch_dataset = Data.TensorDataset(eeg_data, labels.squeeze())
        train_loader = Data.DataLoader(dataset=torch_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=4)  ##num_workers是加载文件的核心数
        return train_loader
    else:
        return eeg_data, labels.squeeze()

if __name__ =='__main__':
    ##提取训练数据
    Sam = np.array(range(files_len))  #
    New_sam = KFold(n_splits=5)
    foldi = 0

    print("开始")

    for train_index, test_index in New_sam.split(Sam):

        foldi = foldi + 1
        # test_eeg_data, test_labels = Set_Dataset(h5file, test_index, type=0)

        mnn = MNNs()
        optimizer = torch.optim.SGD(mnn.parameters(), lr=LEARNING_RATE)
        loss_func = nn.CrossEntropyLoss()
        if (use_gpu):
            mnn = mnn.cuda()
            loss_func = loss_func.cuda()

        # 定义数组
        Loss_list_train = []
        Loss_list_test = []
        Accuracy_list = []
        Kappa = []
        Loss_list2 = []
        Accuracy_list2 = []
        Matrix_all = []
        # 我这里迭代了80次，所以x的取值范围为(0，80)，然后再将每次相对应的准确率以及损失率附在x上
        x1 = range(0, EPOCH)
        x2 = range(0, EPOCH)
        y1 = Accuracy_list
        y2 = Loss_list_train
        y3 = Loss_list_test
        best_acc = 0

        print("开始训练------")
        # 训练MNN网络

        max_accuracy = 0

        for epoch in range(EPOCH):
            loss_all = 0
            loss_test_all = 0
            acc_train_all = 0
            step_all = 0
            test_step_all = 0
            global test_pred_all
            global test_label_all
            print("开始训练模型---------")
            part_len = 5
            train_part_len = len(train_index) // part_len
            test_part_len = len(test_index) // part_len
            for train_part in range(part_len):
                if train_part == 0:
                    train_part_index = train_index[train_part_len * train_part:train_part_len * (train_part + 1)]
                elif train_part == (part_len-1):
                    train_part_index = train_index[train_part_len * train_part:len(train_index)]
                else:
                    train_part_index = train_index[train_part_len * train_part:train_part_len * (train_part + 1)]
                for step, (train_x, train_y) in enumerate(Set_Dataset(h5file, train_part_index, type=1)):  # 设定训练数据

                    train_y_shape = train_y.shape[0]
                    train_y = train_y.view(train_y.shape[0])

                    if (use_gpu):
                        train_x, train_y = train_x.cuda(), train_y.cuda()
                    # mnn output
                    output = mnn(train_x.type(torch.FloatTensor))
                    loss_train = loss_func(output, train_y.type(torch.LongTensor))
                    train_pred_y = torch.max(F.softmax(output), 1)[1].numpy().squeeze()

                    # cross entropy loss
                    loss_all = loss_all + loss_train
                    optimizer.zero_grad()  # clear gradients for this training step
                    loss_train.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients
                step_all = step + 1 + step_all
            print("训练模型完毕")
            ##测试集切割
            for test_part in range(part_len):
                if test_part == 0:
                    test_part_index = test_index[test_part_len * test_part:test_part_len * (test_part + 1)]
                elif test_part == part_len-1:
                    test_part_index = test_index[test_part_len * test_part:len(test_index)]
                else:
                    test_part_index = test_index[test_part_len * test_part:test_part_len * (test_part + 1)]

                for step, (test_x, test_y) in enumerate(Set_Dataset(h5file, test_part_index, type=1)):  # 设定训练数据
                    if 'test_label_all' not in vars():
                        test_label_all = test_y
                    else:
                        test_label_all = torch.cat((test_label_all, test_y), dim=0)
                    test_y = test_y.view(test_y.shape[0])
                    if (use_gpu):
                        test_x, test_y = test_x.cuda(), test_y.cuda()

                    # mnn output
                    output = mnn(test_x.type(torch.FloatTensor))
                    test_pred_y = torch.max(F.softmax(output), 1)[1]
                    if 'test_pred_all' not in vars():
                        test_pred_all = test_pred_y
                    else:
                        test_pred_all = torch.cat((test_pred_all, test_pred_y), dim=0)
                    loss_test = loss_func(output, test_y.type(torch.LongTensor))
                    # cross entropy loss
                    loss_test_all = loss_test_all + loss_test
                test_step_all = step + 1 + test_step_all

            # 准确率
            train_accuracy = accuracy_score(train_y.numpy().squeeze(), train_pred_y)
            loss_train_avg = loss_all / step_all
            Loss_list_train.append(loss_train_avg)

            # #测试集

            test_accuracy = accuracy_score(test_label_all.numpy().squeeze(), test_pred_all.numpy().squeeze())
            loss_test_avg = loss_test_all / test_step_all
            Loss_list_test.append(loss_test_avg)

            Accuracy_list.append(test_accuracy)
            # accuracy = sum(test_labels1 == pred_y)/ test_labels.numpy().squeeze().size
            print("Kold_" + str(foldi) + '_Epoch: ', epoch + 1, '| train loss: %.4f' % loss_train_avg,
                  '| train accuracy: %.4f' % train_accuracy,
                  '| test loss: %.4f' % loss_test_avg, '| test accuracy: %.4f' % test_accuracy)

            gc.collect()

            Matrix_all.append(cm(np.array(test_label_all).reshape(-1).squeeze(), np.array(test_pred_all).reshape(-1).squeeze()))
            Kappa.append(kappa(cm(np.array(test_label_all).reshape(-1).squeeze(), np.array(test_pred_all).reshape(-1).squeeze())))
        # Max_acc.append(accuracy)
        # #混淆矩阵

        print("结束训练网络-------")

        # 保存
        result1 = np.array(Matrix_all).reshape(-1, 25)
        # result4 = np.array(Loss_list_test)
        result2 = np.array(Loss_list_train)
        result3 = np.array(Loss_list_test)
        result4 = np.array(Accuracy_list)
        saveResult(result1, result2, result3, result4, "_kold" + str(foldi).zfill(2), mnn)
        # 画图及保存
        plot_acc_loss(x1, x2, y1, y2, y3, "_kold" + str(foldi).zfill(2))
        result5 = np.array(Kappa)
        cm_plot(np.array(test_label_all).reshape(-1).squeeze(), np.array(test_pred_all).reshape(-1).squeeze(),
                "_foid" + str(foldi))
        np.savetxt('./result/kappa/kappa_flod' + str(foldi) + '.txt', result5)
    print("结束")



