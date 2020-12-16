import matplotlib as plt

import torch
import torch.nn as nn
import os
import numpy as np
import h5py 
from sklearn import preprocessing
import torch.utils.data as Data
import pandas as pd  
import pickle  
from sklearn.metrics import roc_auc_score  
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, f1_score, recall_score
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文

filepath = "D:/WorkSpace/Shell/GRU/Result/test_pred/"  #图片保存路径
savepath = "D:/WorkSpace/Shell/GRU/Result/picture/"  #图片保存路径

def kappa(matrix):
    n = np.sum(matrix)
    sum_po = 0
    sum_pe = 0
    for i in range(len(matrix[0])):
        sum_po += matrix[i][i]
        row = np.sum(matrix[i, :])
        col = np.sum(matrix[:, i])
        sum_pe += row * col
    po = sum_po / n
    pe = sum_pe / (n * n)
    # print(po, pe)
    return (po - pe) / (1 - pe)
def cm_plot(original_label, predict_label,kunm,pic=None):
    cm = confusion_matrix(original_label, predict_label)  
    print('kappa:', kappa(cm))
    plt.figure()
    plt.matshow(cm, cmap=plt.cm.Blues)     
    plt.colorbar()    
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
            
    plt.ylabel('True label')  
    plt.xlabel('Predicted label')  
    plt.title('confusion matrix')
    
    
    if pic is not None:
        plt.savefig(str(pic) + '.svg')
    # plt.xticks(('Wake','N1','N2','N3','REM'))
    # plt.yticks(('Wake','N1','N2','N3','REM'))
    plt.savefig(savepath + "cnnmatrix"+str(kunm)+".svg")
    plt.show()
    # plt.savefig("/home/data_new/zhangyongqing/flx/pythoncode/"+str(knum)+"matrix.jpg")

def train_pre_data(num):
    filename1 = filepath+'testresult'+str(num)+'.h5'
    f1 = h5py.File(filename1, 'r')
    pred=f1['/pred']
    lable=f1['/lable']
    return pred, lable

def pre_data():
    
    # global pred
    # global lable
    for index in range(1, 6):

        pred, lable = train_pre_data(index)
        cm_plot(lable, pred,index)
        #pred1, lable1 = train_pre_data(j)
            # pred = np.concatenate((pred, pred1))
            # lable = np.concatenate((lable, lable1))

    return pred, lable
def pre_data2(num):
    filename2 = '/opt/data/private/new/handfea/traindata/newfea/num1/lastresult/Last_SLP'+str(num)+'.h5'
    f2 = h5py.File(filename2, 'r')
    time1 = f2['/time'][:]
    time2 = [n.decode("ascii", "ignore") for n in time1]
    pred = f2['/result'][:]
    lable = f2['/lable'][:]

    return pred,lable,time2
if __name__ == '__main__':

    list1 = [1080, 1082, 1086, 1088, 1090]

    pre_data()



    for num in list1:


        #pred,lable,time=pre_data2(num)
        pred,lable=pre_data()

        # accuracy = accuracy_score(lable, pred)
        # print(accuracy)
        # pred=pred[:794]
        # lable=lable[:794]
        # print(classification_report(pred,lable))
        # accuracy = accuracy_score(pred,lable)
        # print(accuracy)
        # cm_plot(lable,pred)

        x = range(0, len(pred), 100)
        #x_axis_data = time

        # print(x_axis_data)
        y_axis_data1 = pred
        y = [0, 1, 2, 3, 4]
        # plt.figure(dpi=200)
        plt.figure(figsize=(26, 8))
        # plt.scatter(x_axis_data, pred, marker='x', color='red', s=40, label='predict')
        # plt.scatter(x_axis_data, lable, marker = 'x', color = 'blue', s = 40, label = 'Second')
        # plt.legend(loc='best')
        # plt.show()
        # plt.savefig("/opt/data/private/new/pythoncode/picture/sdtgru1082.svg")

        # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
        # plt.plot(x_axis_data, y_axis_data1, 'r-', color='blue', alpha=1, linewidth=1,label='predict')
        # plt.plot(x_axis_data, lable, 'r-', color='red', alpha=1, linewidth=1, label='lable')

        # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
        plt.legend(loc="upper right")
        plt.xlabel('Time')
        # plt.ylabel('accuracy')
        plt.xticks(x)
        plt.yticks(y,('Wake','N1','N2','N3','REM'))
        plt.tick_params(labelsize=16)
        ax=plt.gca()
        ax.spines['bottom'].set_linewidth(2)
        ax.spines['left'].set_linewidth(2)
        accuracy = accuracy_score(lable, pred)
        print(accuracy)
        plt.show()
        plt.savefig(savepath+"grucrf"+str(num)+".svg")
