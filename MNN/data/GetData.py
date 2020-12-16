import pyedflib
import numpy as np
import h5py
import csv
import math
import pandas as pd
import os
import torch
'''
data_path = 'D:/WorkSoftwareData/alldata-90/'   # 存放数据的具体位置，需要改成自己数据存放的地方
signal_name  = 'C4A1'     # 所选的通道名称
raw_data = mne.io.read_raw_edf(dataset_path, preload=True)
# preload: 如果为True，则数据将被预加载到内存中(这样可以加快数据的索引), 默认为False
raw.pick_channels([signal_name])
eeg = raw.to_data_frame()   # 将读取的数据转换成pandas的DataFrame数据格式
eeg = list(eeg.values[:,1])  #转换成numpy的特有数据格式
'''

data_path = 'D:/WorkSpace/Shell/data/alldata-90/data/'   # 存放数据的具体位置，需要改成自己数据存放的地方
data_files = os.listdir(data_path) #得到文件夹下的所有文件名称
lable_path = 'D:/WorkSpace/Shell/data/alldata-90/lable/'
lable_files = os.listdir(lable_path) #得到文件夹下的所有文件名称
h5file = "D:/WorkSpace/Shell/data/alldata-90/h5data/"
channels = ['ECG', 'EMG', 'EOG-R', 'EOG-L', 'O2-A1', 'C4-A1', 'F4-A1']
##迭代次数为50
EPOCH = 50
#batch size 为500
BATCH_SIZE = 500
##学习率为0.01
LEARNING_RATE = 0.01
##动量因子为0.9
MOMENTUM = 0.9
##mlp隐藏单元
mlp_hsize = 300
##rnn隐藏单元
rnn_hsize =300
# 降采样
def downsample(signals, signal_size, oldfreq, newfreq):
    index = pd.date_range('1/1/2000', periods=signal_size, freq=oldfreq)  # 这个起始时间任意指定，freq为其频率
    data = pd.Series(signals, index=index)
    data_obj = data.resample(rule=newfreq, kind='period').mean().to_numpy()  # 第一个为抽样频率，label表示左右开闭区间
    print(len(data_obj))
    return data_obj


# 转换文件格式
def loadData(data_files, h5file, lable_files):
    # 对所有文件进行转换
    row = 0
    for filei in range(len(data_files)):
        print("文件：", filei+1)
        if (filei+1) in [3,4,5,10,31,33,36,37,38,51,54,56,58,62,66,67,74,78,83,84]:
            continue
        print("wenjian"+data_path+data_files[filei])
        fileedf = pyedflib.EdfReader(data_path+data_files[filei])
        signal_headers = fileedf.getSignalHeaders()
        fileh5 = h5py.File(h5file+str(filei+1).zfill(4)+".h5", 'w')
        #将采样率和文件
        fileh5['sample_rate'] = [np.array(signal_headers[0]['sample_rate'], dtype=int)]
        fileh5['digital_max'] = [np.array(signal_headers[0]['digital_max'], dtype=int)]
        fileh5['digital_min'] = [np.array(signal_headers[0]['digital_min'], dtype=int)]
        #    print(fileh5[channels[5]].value.shape)
        sample_rate = signal_headers[0]['sample_rate']
        column = 30 * sample_rate  ##30秒epochs  列数f
        size = 0
        row = 0
        newSignal = 0
        for signi in [3,6]:

            ##为了后续的第二个通道进行降采样对应label
            column = 30 * sample_rate
            size = fileedf.readSignal(signi, 0, None, False).size  # 获取数据点总数
            row = size // column  ##行数
            oldsignals = fileedf.readSignal(signi, 0, size, False)
            if(fileh5['sample_rate'][0] != 250):
                #删除已经赋予的其他采样率

                column = 30 * 250
                print("旧的采样点总数",size)
                ##用来处理采样率  2L为原始采样率：1/采样率   4L为新采样率：1/采样率
                newsignals = downsample(oldsignals,size,'2L','4L')
                # print("采样率异常" + data_path + data_files[filei])
            else:
                newsignals = oldsignals
            newSignal = len(newsignals) // column * column  ##epochs总数

            fileh5[str(channels[signi])] = newsignals[0:newSignal].reshape(-1, column)  # 存储数据点
            # 将标签加入数据文件中
        del fileh5['sample_rate']
        fileh5['sample_rate'] = [250]
        with open(lable_path+lable_files[filei], "r") as f:
            csv_read = csv.reader(f)
            labels = []
            # print("行数",row)
            for labelsum, label in enumerate(csv_read):
                if labelsum != row:
                    labels.append(label[0])  # 或许对应数量的标签
                else:
                    break
            labels = np.array(labels, dtype='int32').reshape(-1, 1)  # 转换类型
            fileh5['labels'] = labels
        fileedf.close()
        #patchsize = int(math.sqrt(column))  # 确定每个patch的长和宽
        # 短时傅里叶变换  n_fft=5s*采样率  窗口重叠为70%即窗口滑动hop_length=n_fft*30%
        ##返回值为一个tensor,其中第一个维度为输入数据的batch size，
        # 第二个维度为STFT应用的频数，
        # 第三个维度为帧总数，
        # 最后一个维度包含了返回的复数值中的实部和虚部部分
        # 拼接文件的数据
        # if filei == 0:
        #     stft_data = torch.stft(torch.from_numpy(fileh5[channels[5]].value),n_fft=5*sample_rate,hop_length=int(5*sample_rate*0.25))
        #     labels = torch.from_numpy(fileh5["labels"].value)
        # else:
        #     stft_data = torch.cat((stft_data,torch.stft(torch.from_numpy(fileh5[channels[5]].value),n_fft=5*sample_rate,hop_length=int(5*sample_rate*0.25))),dim = 0)
        #     labels = torch.cat((labels,torch.from_numpy(fileh5["labels"].value)),dim = 0)
        #


    #     if filei == 0:
    #         eeg_data_channel1 = torch.from_numpy(fileh5[channels[3]].value)
    #         eeg_data_channel2 = torch.from_numpy(fileh5[channels[5]].value)
    #         eeg_data = torch.cat((eeg_data_channel1, eeg_data_channel2), dim=1)
    #         labels = torch.from_numpy(fileh5["labels"].value)
    #     else:
    #         eeg_data_channel1 = torch.from_numpy(fileh5[channels[3]].value)
    #         eeg_data_channel2 = torch.from_numpy(fileh5[channels[5]].value)
    #         eeg_data_new = torch.cat((eeg_data_channel1, eeg_data_channel2), dim=1)
    #         eeg_data = torch.cat((eeg_data, eeg_data_new), dim=0)
    #         labels = torch.cat((labels, torch.from_numpy(fileh5["labels"].value)), dim=0)
    #
    # fileh5 = h5py.File(data_path + h5file[filei], 'r')
    # row += fileh5[channels[6]].value.shape[0]  # 行数
    # sample_rate = fileh5['sample_rate'][0]
    # column = 30 * fileh5['sample_rate'][0]  ##30秒epochs  列数f
    # patchsize = int(math.sqrt(column))  # 确定每个patch的长和宽
    # eeg_data_channel1 = torch.from_numpy(fileh5[channels[3]].value)
    # eeg_data_channel2 = torch.from_numpy(fileh5[channels[5]].value)
    # test_eeg_data = torch.cat((eeg_data_channel1, eeg_data_channel2), dim=1)
    # test_labels = torch.from_numpy(fileh5["labels"].value)





print("开始")
loadData(data_files, h5file, lable_files)
print("结束")


def getData(complexd_h5file):
    filekey=set()
    for filei in range(5):
        fileh5 = h5py.File(complexd_h5file[filei], 'r')
        for key in fileh5.keys():
            filekey.add(key)
        print(type(fileh5["EMG_cos"].value), fileh5["EMG_cos"].value)
        print("------------------------")
        fileh5.close()
    return

# print("开始")
# getData(complexd_h5file)
# print("结束")


def pcaWhite(X):
    X -= np.mean(X, axis=0)  # 减去均值，使得以0为中心
    cov = np.dot(X.T, X) / X.shape[0]  # 计算协方差矩阵
    U, S, V = np.linalg.svd(cov)  # 矩阵的奇异值分解
    Xrot = np.dot(X, U)
    Xwhite = Xrot / np.sqrt(S + 1e-5)  # 加上1e-5是为了防止出现分母为0的异常
    return X


##将实数转化为复数
# def complexedValue(complexd_h5file, channels, white):
#     for filei in range(5):
#         fileedf = pyedflib.EdfReader(edffile[filei])
#         complexd_fileh5 = h5py.File(complexd_h5file[filei], 'w')
#         fileh5 = h5py.File(h5file[filei], 'r')
#         signal_headers = fileedf.getSignalHeaders()
#         # 将信息放入新文件中
#         complexd_fileh5['sample_rate'] = np.array(signal_headers[filei]['sample_rate'], dtype=int)
#         complexd_fileh5['digital_max'] = np.array(signal_headers[filei]['digital_max'], dtype=int)
#         complexd_fileh5['digital_min'] = np.array(signal_headers[filei]['digital_min'], dtype=int)
#         complexd_fileh5['labels'] = fileh5['labels'].value
#         # print(complexd_fileh5['labels'].value[1:10,:],fileh5['labels'].value[1:10,:])
#         # z(i) = cosx + i*sinx  分别放在数组中
#         digital_max = fileh5["digital_max"].value
#         digital_min = fileh5["digital_min"].value
#
#         # 遍历所有通道，并将数据转换为复值形式
#         for channel in channels:
#             channel_data = fileh5[channel].value
#             angle = (channel_data - np.array(digital_min)) / np.array((digital_max - digital_min) * math.pi)
#             column = len(angle[0])
#             anglecos = np.cos(angle)  # reshape
#             anglesin = np.sin(angle)
#             print("白化开始")
#             if white == 1:
#                 # 白化处理
#                 complexd_fileh5[channel + "_cos"] = pcaWhite(anglecos)
#                 complexd_fileh5[channel + "_sin"] = pcaWhite(anglesin)
#
#             else:
#                 complexd_fileh5[channel + "_cos"] = anglecos
#                 complexd_fileh5[channel + "_sin"] = anglesin
#
#             # print('白化-------------')
#             #
#             # complexd_fileh5[channel+"_cos"] = pcaWhite(np.cos(angle))    #白化
#             # complexd_fileh5[channel+"_sin"] = pcaWhite(np.sin(angle))    #白化
#             print('白化结束---------')
# print("开始")
# complexedValue(complexd_h5file, channels,white=0)
# print("结束")


