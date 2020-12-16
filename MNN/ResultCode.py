from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import numpy as np
from copy import deepcopy
import matplotlib as plt
import os
import h5py
from sklearn import preprocessing
import torch.utils.data as Data
import pandas as pd
import pickle
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score, recall_score, precision_score
# mpl.rcParams['font.sans-serif'] = ['SimHei']  # 添加这条可以让图形显示中文
from sklearn.model_selection import KFold
import copy
import seaborn as sns
kappapath = "D:/WorkSpace/Shell/MNN/result_new20/kappa/"
datapath = "D:/WorkSpace/Shell/datafeature/"  #数据路径
filepath = "D:/WorkSpace/Shell/MNN/result_new20/label/"  #预测结果保存路径
savepath = "D:/WorkSpace/Shell/MNN/result_new20/picture/matrix/"  #图片保存路径



time_filepath = "D:/WorkSpace/Shell/MNN/lastresult/"  #保存有时间戳的文件

time_files = os.listdir(time_filepath)  # 得到文件夹下的所有文件名称
timefiles_len = len(time_files)

files = os.listdir(datapath)  # 得到文件夹下的所有文件名称
files_len = len(files)

datapath_read_path = "E:/cao/datafeature/"  #数据路径
datapath_read =os.listdir(time_filepath)
files_read = os.listdir(time_filepath)  # 得到文件夹下的所有文件名称

files_result = os.listdir(filepath)  # 得到文件夹下的所有文件名称

sleepPicture_path = "D:/WorkSpace/Shell/MNN/result_new20/picture/sleep_picture/"


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


def cm_plot(original_label, predict_label, kunm, pic=None):

    prec_score = precision_score(original_label, predict_label, average=None)
    recall = recall_score(original_label, predict_label, average=None)
    f1 = f1_score(original_label, predict_label, average=None)
    cm = confusion_matrix(original_label, predict_label)
    cm_new = np.empty(shape=[5,5])
    for x in range(5):
        t=cm.sum(axis=1)[x]
        for y in range(5):
            cm_new[x][y] = round(cm[x][y]/t * 100, 2)
    plt.figure()
    plt.matshow(cm_new, cmap=plt.cm.Blues)
    plt.colorbar()
    x_numbers = []
    y_numbers = []
    cm_percent = []
    for x in range(5):
        y_numbers.append(cm.sum(axis=1)[x])
        x_numbers.append(cm.sum(axis=0)[x])
        for y in range(5):
            percent = format(cm_new[x, y] * 100 / cm_new.sum(axis=1)[x], ".2f")
            cm_percent.append(str(percent))
            plt.annotate(format(cm_new[x, y] * 100/cm_new.sum(axis=1)[x], ".2f"), xy=(y, x), horizontalalignment='center',
                         verticalalignment='center', fontsize=10)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('confusion matrix')

    y_stage = ["W\n(" + str(y_numbers[0]) + ")", "N1\n(" + str(y_numbers[1]) + ")", "N2\n(" + str(y_numbers[2]) + ")", "N3\n(" + str(y_numbers[3]) + ")", "REM\n("+ str(y_numbers[4]) + ")"]
    x_stage = ["W\n(" + str(x_numbers[0]) + ")", "N1\n(" + str(x_numbers[1]) + ")", "N2\n(" + str(x_numbers[2]) + ")",
               "N3\n(" + str(x_numbers[3]) + ")", "REM\n(" + str(x_numbers[4]) + ")"]
    y = [0, 1, 2, 3, 4]
    plt.xticks(y, x_stage)
    plt.yticks(y, y_stage)
    #sns.heatmap(cm_percent, fmt='g', cmap="Blues", annot=True, cbar=False, xticklabels=x_stage, yticklabels=y_stage)  # 画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒



    plt.savefig(savepath + "matrix" + str(kunm) + ".svg")
    #plt.show()
    plt.show()
    plt.close()
    # plt.savefig("/home/data_new/zhangyongqing/flx/pythoncode/"+str(knum)+"matrix.jpg")
    return kappa(cm), classification_report(original_label, predict_label)

def train_pre_data(num):
    filename1 = filepath + 'kfold_' + str(num) + '.h5'
    f1 = h5py.File(filename1, 'r')

    pred = f1['label']
    lable = f1['pred']
    return pred, lable


def pre_data():
    Kappa = []
    Prec_score = []
    for index in range(1, 21):

        print(index)
        pred, lable = train_pre_data(index)
        kappa_index, prec_score= cm_plot(lable, pred, index)
        Kappa.append(kappa_index)
        Prec_score.append(prec_score)
        np.savetxt(kappapath+"/kappa/kappa_" + str(index) +".txt", np.array(Kappa))
        np.savetxt(kappapath+"/report/report_" + str(index) +".txt", np.array(Prec_score))


def pre_data2(num):
    #filename2 = '/opt/data/private/new/handfea/traindata/newfea/num1/lastresult/SLP' + str(num) + '.h5'
    for index in range(timefiles_len):
        file_dat = h5py.File(datapath + files[index], 'r')

        time1 = file_dat['/time'][:]
        time2 = [n.decode("ascii", "ignore") for n in time1]
        pred = file_dat['/result'][:]
        lable = file_dat['/label'][:]

    return pred, lable, time2


##给原始数据文件加上时间戳数据
def add_time():
    #filename2 = '/opt/data/private/new/handfea/traindata/newfea/num1/lastresult/SLP' + str(num) + '.h5'

    for index in range(timefiles_len):
        file_time = h5py.File(time_filepath+time_files[index], 'r')
        file_read = h5py.File(datapath_read+files_read[index], 'r')

        if len(file_time.keys()) == 2  :
            file_time.close()
            file_read.close()
            continue
        else:
            file_dat = h5py.File(datapath + files[index], 'w')
            file_dat['time'] = file_time['time'][:]
            file_dat['handfea'] = file_read['handfea'][:]
            file_dat['label'] = file_read['lable'][:]
            file_time.close()
            file_read.close()
            file_dat.close()

#将所有测试集的睡眠特征图画出
def plot_sleepPicture():
    print(files_len)
    files_kflod = np.array(range(files_len))
    kf = KFold(n_splits=5)

    for index in range(1, 6):


        file_name = filepath + files_result[index - 1]
        file_result = h5py.File(file_name, 'r')
        if index == 1:
            print(file_result['label'][1:100])
        knum = 0
        for train_index, test_index in kf.split(files_kflod):

            ##五折分别对应五个restult文件
            if index == (knum + 1):
                last_time = 0
                print("折数",index, knum + 1)
                for test_index_knum in test_index:
                    files_time = h5py.File(time_filepath + time_files[test_index_knum], 'r')  #读取文件中的时间戳信息 预测值和真实标签值
                    file_time = files_time['time'][:]
                      #从总的测试集结果文件中读取预测数据和label数据
                    print(last_time)
                    label = file_result['label'][last_time:last_time + file_time.shape[0]]
                    result = file_result['pred'][last_time:last_time + file_time.shape[0]]
                    last_time = file_time.shape[0] + last_time
                    ##替换为正确的stage
                    label_new = replace_stage(copy.deepcopy(label))
                    result_new = replace_stage(copy.deepcopy(result))
                    plot_sleep(label_new, result_new, file_time,"kflod"+str(knum + 1),files[test_index_knum][0:7])  #画出睡眠结构图
                    files_time.close()
                knum = knum + 1
            else:
                knum = knum + 1
                continue
        file_result.close()

##将睡眠分期对应的数字替换正确
def replace_stage(old_array):
    old_array[old_array == 4] = 5
    old_array[old_array == 3] = 4
    old_array[old_array == 2] = 3
    old_array[old_array == 1] = 2
    old_array[old_array == 5] = 1
    return old_array


#画出睡眠特征图
def plot_sleep(label, pred, file_time, kfold, name):
    x = range(0, len(label))
    y = np.arange(5)
    y_stage = ["W", "REM", "N1", "N2", "N3"]
    plt.figure(figsize=(24, 8))
    plt.ylabel("Sleep Stage")
    plt.xlabel("Sleep Time")
    time_choice = np.array([s.decode('UTF-8') for s in file_time[0:len(file_time):int(len(file_time) / 10)]])
    plt.xticks(range(0, len(file_time), int(len(file_time)/10)), time_choice)  ##为了将坐标刻度设为字
    plt.yticks(y, y_stage)  ##为了将坐标刻度设为字
    # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    plt.plot(x, pred, 'r-', color='blue', alpha=1, linewidth=1, label='predict')
    plt.plot(x, label, 'r-', color='red', alpha=1, linewidth=1, label='label')
    plt.legend(loc='best')
    plt.savefig(sleepPicture_path +str(kfold)+"sdt"+str(name)+".svg")   ##保存睡眠模型图文件
    plt.close()






#

if __name__ == '__main__':
#     # pred, lable, time2 = pre_data2(1002)
#     # print(time2)
#
    #plot_sleepPicture()


    pre_data()

    # for num in list1:
    #
    #     # accuracy = accuracy_score(lable, pred)
    #     # print(accuracy)
    #     # pred=pred[:794]
    #     # lable=lable[:794]
    #     # print(classification_report(pred,lable))
    #     # accuracy = accuracy_score(pred,lable)
    #     # print(accuracy)
    #     # cm_plot(lable,pred)
    #
    #     x = range(0,len(pred),100)
    #     x_axis_data = time
    #     # print(x_axis_data)
    #     y_axis_data1 = pred
    #     y=[0,1,2,3,4]
    #     # plt.figure(dpi=200)
    #     plt.figure(figsize=(26,8))
    #     # plt.scatter(x_axis_data, pred, marker='x', color='red', s=40, label='predict')
    #     # plt.scatter(x_axis_data, lable, marker = 'x', color = 'blue', s = 40, label = 'Second')
    #     # plt.legend(loc='best')
    #     # plt.show()
    #     # plt.savefig("/opt/data/private/new/pythoncode/picture/sdtgru1082.svg")
    #
    #     # plot中参数的含义分别是横轴值，纵轴值，线的形状，颜色，透明度,线的宽度和标签
    #     plt.plot(x_axis_data, y_axis_data1, 'r-', color='blue', alpha=1, linewidth=1,label='predict')
    #     plt.plot(x_axis_data, lable, 'r-', color='red', alpha=1, linewidth=1, label='lable')
    #
    #     # 显示标签，如果不加这句，即使在plot中加了label='一些数字'的参数，最终还是不会显示标签
    #     plt.legend(loc="upper right")
    #     plt.xlabel('Time')
    #     # plt.ylabel('accuracy')
    #     plt.xticks(x)
    #     plt.yticks(y,('Wake','N1','N2','N3','REM'))
    #     plt.tick_params(labelsize=16)
    #     ax=plt.gca()
    #     ax.spines['bottom'].set_linewidth(2)
    #     ax.spines['left'].set_linewidth(2)
    #     accuracy = accuracy_score(lable,pred)
    #     print(accuracy)
    #     plt.show()
