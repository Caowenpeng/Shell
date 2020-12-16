import torch
import torch.nn as nn
import os
import  matplotlib.pyplot as plt
import numpy as np
import h5py
import random
import pandas as pd
from sklearn import preprocessing
import torch.utils.data as Data
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import random
import crftrain
from random import choice
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report



datapath = "E:/Data/datafeature/"  #数据路径
savepath = "E:/Result/"  #结果保存路径

##kappa系数计算
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
##混淆矩阵
def cm_plot(original_label, predict_label, knum,pic=None):
    cm = confusion_matrix(original_label, predict_label)
    print('kappa:',kappa(cm))
    plt.figure()
    plt.matshow(cm, cmap=plt.cm.Blues)
    plt.colorbar()
    for x in range(len(cm)):
        for y in range(len(cm)):
            plt.annotate(cm[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center')
    # plt.yticks(cm,('Wake','N1','N2','N3','REM'))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('confusion matrix')
    if pic is not None:
        plt.savefig(str(pic) + '.jpg')
    plt.show()
    # plt.savefig("/home/data_new/zhangyongqing/flx/pythoncode/"+str(knum)+"matrix.jpg")
    plt.savefig(savepath+"picture/jgmatrix"+str(knum)+".jpg")
'''
def bu_data():
    filename='/opt/data/private/new/handfea/traindata/newfea/SLP'+str(1099)+'.h5'
    # filename='/home/data_new/zhangyongqing/flx/sleepdata/shhs1-'+str(filenum)+'.h5'
    f = h5py.File(filename, 'r')
    lable=f['/lable'][1018:1140]
    handfea=f['/handfea'][1018:1140]
    
    fea=np.vstack((handfea,handfea,handfea,handfea))
    lable1=np.vstack((lable,lable,lable,lable))
    return fea,lable1
def pre_data(filenum):
    bfea,blable=bu_data()
    filename='/opt/data/private/new/handfea/traindata/O2A1/SLP'+str(filenum)+'.h5'
    # filename='/home/data_new/zhangyongqing/flx/sleepdata/shhs1-'+str(filenum)+'.h5'
    f = h5py.File(filename, 'r')
    lable=f['/lable'][:]
    O2A1=f['/O2-A1'][:]
    f.close()
    filename='/opt/data/private/new/handfea/traindata/F4A1/SLP'+str(filenum)+'.h5'
    # filename='/home/data_new/zhangyongqing/flx/sleepdata/shhs1-'+str(filenum)+'.h5'
    f = h5py.File(filename, 'r')
    F4A1=f['/F4-A1'][:]
    f.close()
    filename='/opt/data/private/new/handfea/traindata/C4A1/SLP'+str(filenum)+'.h5'
    # filename='/home/data_new/zhangyongqing/flx/sleepdata/shhs1-'+str(filenum)+'.h5'
    f = h5py.File(filename, 'r')
    C4A1=f['/C4-A1'][:]
    f.close()
    filename='/opt/data/private/new/handfea/traindata/EOGR/SLP'+str(filenum)+'.h5'
    # filename='/home/data_new/zhangyongqing/flx/sleepdata/shhs1-'+str(filenum)+'.h5'
    f = h5py.File(filename, 'r')
    EOGR=f['/EOG-R'][:]
    f.close()
    filename='/opt/data/private/new/handfea/traindata/EOGL/SLP'+str(filenum)+'.h5'
    # filename='/home/data_new/zhangyongqing/flx/sleepdata/shhs1-'+str(filenum)+'.h5'
    f = h5py.File(filename, 'r')
    EOGL=f['/EOG-L'][:]
    f.close()
    filename='/opt/data/private/new/handfea/traindata/EMG/SLP'+str(filenum)+'.h5'
    # filename='/home/data_new/zhangyongqing/flx/sleepdata/shhs1-'+str(filenum)+'.h5'
    f = h5py.File(filename, 'r')
    EMG=f['/EMG'][:]
    f.close()
    fea=np.hstack((C4A1,EMG,EOGL,EOGR,F4A1,O2A1))
    
    fea=pd.DataFrame(fea)
    fea=fea.fillna(0)
    
    fea=np.array(fea)
    min_max_scaler = preprocessing.MinMaxScaler()
    fea = min_max_scaler.fit_transform(fea)
    fea=np.vstack((fea,bfea[:1140-len(lable)]))
    lable=np.vstack((lable,blable[:1140-len(lable)]))
    fea=(torch.from_numpy(fea)).type('torch.FloatTensor')
    newfile1 = h5py.File('/opt/data/private/new/handfea/traindata/newfea/SLP'+str(filenum)+'.h5','w')
    newfile1['/lable']=lable
    newfile1['/handfea']=fea
    print(len(lable),len(fea))
    # return fea,lable
'''
def Log_Train_Test(LR,batchsize,epochs,knum):
    '''
    list1=[1006,1007,1008,1009,1035,1101,1103]
    list2=[1002,1011,1013,1015,1017,1019]
    list3=[1001,1012,1014,1016,1018,1020]
    list4=[1092,1094,1095,1100,1102]
    list5=[1021,1022,1023,1024,1025,1026]
    list6=[1027,1028,1029,1030,1032,1034]
    list7=[1040,1042,1044,1046,1048]
    list8=[1039,1041,1043,1045,1047,1049,1050]
    list9=[1053,1055,1057,1059,1061,1063]
    list10=[1065,1069,1071,1073,1075]
    list11=[1077,1079,1081,1085,1087,1089]
    list12=[1080,1082,1086,1088,1090]
    list13=[1068,1070,1072,1076]
    list14=[1052,1096,1098]
    list15=[1091,1093]
    list16=[1060,1064]
    list17=[1097,1099]
    
    a1=choice(list1)
    a2=choice(list2)
    a3=choice(list3)
    a4=choice(list4)
    a5=choice(list5)
    a6=choice(list6)
    a7=choice(list7)
    a8=choice(list8)
    a9=choice(list9)
    a10=choice(list10)
    a11=choice(list11)
    a12=choice(list12)
    a13=choice(list13)
    a14=choice(list14)
    a15=choice(list15)
    a16=choice(list16)
    a17=choice(list17)
    list_test=[a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15,a16,a17]
    '''
    # list_test=[1101, 1017, 1001, 1100, 1024, 1030, 1048, 1045, 1063, 1065, 1079, 1082, 1076, 1052, 1093, 1064, 1097]
    # list_test=[1008, 1019, 1001, 1092, 1026, 1032, 1042, 1043, 1063, 1071, 1079, 1086, 1068, 1096, 1093, 1064, 1099]
    # list_test=[1007, 1019, 1012, 1102, 1026, 1034, 1048, 1045, 1061, 1065, 1085, 1086, 1068, 1098, 1093, 1060, 1099]
    # list_test=[1103, 1019, 1018, 1100, 1023, 1034, 1048, 1041, 1057, 1071, 1081, 1090, 1076, 1052, 1091, 1060, 1097]
    # list_test=[1103, 1017, 1014, 1102, 1022, 1032, 1040, 1043, 1057, 1075, 1085, 1080, 1072, 1098, 1093, 1064, 1099]
    # list_test=[1101, 1013, 1016, 1102, 1024, 1030, 1046, 1041, 1063, 1071, 1085, 1090, 1076, 1052, 1093, 1064, 1099]
    # list_test=[1103, 1011, 1014, 1095, 1024, 1027, 1040, 1049, 1063, 1075, 1087, 1088, 1070, 1096, 1093, 1064, 1099]
    # list_test=[1008, 1015, 1014, 1102, 1026, 1029, 1046, 1043, 1061, 1073, 1079, 1082, 1072, 1052, 1093, 1060, 1097]

    # list_test=[1009, 1015, 1012, 1095, 1025, 1034, 1040, 1050, 1061, 1065, 1089, 1088, 1072, 1052, 1093, 1060, 1099]
    # list_test=[1103, 1017, 1012, 1100, 1024, 1029, 1046, 1049, 1059, 1071, 1079, 1086, 1072, 1098, 1091, 1064, 1097]

    # list_test=[1029, 1023, 1032, 1088, 1096, 1013, 1011, 1018, 1097, 1045, 1025, 1052, 1028, 1087, 1082, 1065, 1007]
    # list_test=[1065, 1069, 1071, 1073, 1075, 1080, 1082, 1086, 1088, 1090, 1068, 1070, 1072, 1076, 1052, 1096, 1098]
    # list_test=[1001, 1002, 1006, 1007, 1008, 1009, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021]
    # list_test = [1006, 1007, 1008, 1009, 1035, 1101, 1103, 1002, 1011, 1013, 1015, 1017, 1019, 1097, 1099, 1060, 1064]
    list_test =[1021,1022,1023,1024,1025,1026,1027,1028,1029,1030,1032,1034,1040,1042,1044,1046,1048]
    no_list=[1003,1004,1005,1010,1031,1033,1036,1037,1038,1051,1054,1056,1058,1062,1066,1067,1074,1078,1083,1084]

    list_train=[]
    for i in range(1001,1104,1):
        if i not in no_list and i not in list_test:
            list_train.append(i)


    # random.shuffle(list_train)
    print(list_train,list_test)
    # print(list_train[17:-1],list_train[:17])
    crftrain.train(LR,batchsize,epochs,list_train,list_test,knum)

def pre_data(filenum):
    filename1=datapath+"SLP"+str(filenum)+'.h5'
    f1 = h5py.File(filename1, 'r')
    lable=f1['/lable'][720:1080]
    x_data=f1['/handfea'][720:1080]
    # print(len(lable),len(x_data))
    return x_data,lable

def pre_train(tlist,batchsize):
    x_train = np.empty(shape=[0,912])
    y_train=np.empty(shape=[0,1])
    for i in tlist:
        x_train1, y_train1=pre_data(i)
        x_train=np.concatenate((x_train,x_train1), axis = 0) 
        y_train=np.concatenate((y_train,y_train1),axis = 0)
        
    x_train=(torch.from_numpy(x_train)).type('torch.FloatTensor')
    y_train=(torch.from_numpy(y_train)).type('torch.LongTensor')
    y_train=y_train.squeeze(dim=1)
    print(x_train.shape,y_train.shape)
    torch_dataset = Data.TensorDataset(x_train, y_train)
    loader = Data.DataLoader(
        dataset=torch_dataset,      # torch TensorDataset format
        batch_size=batchsize,      # mini batch size
        shuffle=True,               # random shuffle for training
        num_workers=1,              # subprocesses for loading data
        drop_last=False,
    )
    return loader

def pre_test(tlist,batchsize):
    x_train = np.empty(shape=[0,912])
    y_train=np.empty(shape=[0,1])
    for i in tlist:
        x_train1, y_train1=pre_data(i)
        
        # trainlen.append(len(y_train1))
        x_train=np.concatenate((x_train,x_train1), axis = 0) 
        y_train=np.concatenate((y_train,y_train1),axis = 0)
        
    x_train=(torch.from_numpy(x_train)).type('torch.FloatTensor')
    y_train=(torch.from_numpy(y_train)).type('torch.LongTensor')
    y_train=y_train.squeeze(dim=1)
    print(x_train.shape,y_train.shape)
    # torch_dataset = Data.TensorDataset(x_train, y_train)
    # loader = Data.DataLoader(
    #     dataset=torch_dataset,      # torch TensorDataset format
    #     batch_size=batchsize,      # mini batch size
    #     shuffle=False,               # random shuffle for training
    #     num_workers=1,              # subprocesses for loading data
    #     drop_last=False,
    # )
    return x_train, y_train

if __name__ == '__main__':

    # list_test=[1103, 1019, 1018, 1100, 1023, 1034, 1048, 1041, 1057, 1071, 1081, 1090, 1076, 1052, 1091, 1060, 1097]
    # # list_test=[1103, 1017, 1014, 1102, 1022, 1032, 1040, 1043, 1057, 1075, 1085, 1080, 1072, 1098, 1093, 1064, 1099]
    # # list_test=[1101, 1013, 1016, 1102, 1024, 1030, 1046, 1041, 1063, 1071, 1085, 1090, 1076, 1052, 1093, 1064, 1099]
    # # list_test=[1103, 1011, 1014, 1095, 1024, 1027, 1040, 1049, 1063, 1075, 1087, 1088, 1070, 1096, 1093, 1064, 1099]
    #
    # # list_test = [1007, 1019, 1012, 1102, 1026, 1034, 1048, 1045, 1061, 1065, 1085, 1086, 1068, 1098, 1093, 1060, 1099]
    # list_train = []
    # for i in range(1001, 1104, 1):
    #     if i not in list_test:
    #         list_train.append(i)
    # print(list_train, list_test)
    # clf = SVC(C=1, kernel='rbf', gamma=2**(-1), decision_function_shape='ovr')
    # x_train, y_train=pre_train(list_train,0)
    # x_test, y_test=pre_test(list_test,0)
    # clf.fit(x_train, y_train)
    #
    # train_acc = clf.score(x_train, y_train)
    # test_acc = clf.score(x_test, y_test)
    # y_pre = clf.predict(x_test)
    # print(train_acc,test_acc)
    # cm_plot(y_test, y_pre, 9)
    # print(classification_report(y_test, y_pre))

    epochs = 50
    batchsize= 128
    LR = 0.0001     
    Log_Train_Test(LR,batchsize,epochs,5)
    # Log_Train_Test()
    # for filenum in range(1001,1104):
    #     if filenum not in list1:
    #         pre_data(filenum)

    # j=0
    # for filenum in range(1001,1104):
    #     filename='/opt/data/private/new/handfea/traindata/O2A1/SLP'+str(filenum)+'.h5'
    #     f = h5py.File(filename, 'r')
    #     # O2A1=f['/O2-A1']
    #     lable=f['/lable'][:]
    #     # print(filenum,":",len(lable))
    #     if len(lable)<=1140:
    #         j=j+1
    #         print(filenum,":",len(lable),1140-len(lable))
    # print(j)
