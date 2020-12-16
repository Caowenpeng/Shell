from random import choice

from scipy.spatial.distance import cdist
import numpy as np
import os
import seaborn as sns
import pandas as pd
import h5py
import matplotlib.pyplot as plt
channels = ['handfea', 'lable']

h5file = "D:/WorkSpace/Shell/datafeature/"
feature_savepath = "D:/WorkSpace/Shell/GRU/Result/featureHeatmap/"

files = os.listdir(h5file)  # 得到文件夹下的所有文件名称
files_len = len(files)
sleep_label_num = [0, 1, 2, 3, 4]
sleep_label = ['W', 'REM', 'N1', 'N2', 'N3']

list1 = [1006, 1007, 1008, 1009, 1035, 1101, 1103]
list2=[1002,1011,1013,1015, 1017, 1019]
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

##获取两个特征文件的特征相似性
def featureHeatmap(feature1, feature2):

    feature1_np = np.array(feature1)
    feature2_np = np.array(feature2)
    dis = cdist(feature1_np,feature2_np,metric='euclidean')
    return dis

#画出同一个特征文件中特征的相似性热力图
def draw(df):
    dfData = df.corr()
    plt.subplots(figsize=(4, 4))  # 设置画面大小
    sns.heatmap(dfData, annot=True, vmax=1, square=True, cmap="Blues")
    plt.savefig('./BluesStateRelation.png')
    plt.show()
##根据热力图矩阵画出热力图
def plot_Heatmap(heatmap_matrix,heatmap_savepath):

    plt.figure()
    plt.matshow(heatmap_matrix, cmap=plt.cm.Blues_r)
    plt.colorbar()
    for x in range(heatmap_matrix.shape[0]):
        for y in range(heatmap_matrix.shape[0]):
            plt.annotate(heatmap_matrix[x, y], xy=(x, y), horizontalalignment='center', verticalalignment='center', fontsize=15)

    y = np.arange(5)
    y_stage = ["W", "REM", "N1", "N2", "N3"]
    plt.xticks(y, y_stage)
    plt.yticks(y, y_stage)

    plt.legend(fontsize=15)
    plt.ylabel('Night One', fontsize=20)
    plt.xlabel('Night Two', fontsize=20)
    plt.title('Heatmap', fontsize=20)
    plt.savefig(heatmap_savepath+ "heatmap"+".svg")
    plt.show()

##将睡眠分期对应的数字替换正确
def replace_stage(old_array):
    old_array[old_array == 4] = 5
    old_array[old_array == 3] = 4
    old_array[old_array == 2] = 3
    old_array[old_array == 1] = 2
    old_array[old_array == 5] = 1
    return old_array

##用来获取分期特征
def getFeature(file_1,file_2):
    sleep_label_feature = []
    sleep_label_feature_2 = []
    fileh5 = h5py.File(file_1, 'r') #第一个文件用于提取分期特征
    fileh5_2 = h5py.File(file_2, 'r') #第二个文件用于提取分期特征
    eeg_data = fileh5[channels[0]][:]
    label = replace_stage(fileh5[channels[1]][:])
    eeg_data_2 = fileh5_2[channels[0]][:]
    label_2 = replace_stage(fileh5_2[channels[1]][:])

    for index in sleep_label_num:
        ##从文件中获取五个分期的首次出现的下标位置
        label_index = np.argwhere(np.array(label).reshape(-1) == sleep_label_num[index])[2]
        label_index_2 = np.argwhere(np.array(label_2).reshape(-1) == sleep_label_num[index])[2]
        sleep_label_feature.append(np.array(eeg_data)[label_index].reshape(-1))  #获取对应标签的特征
        sleep_label_feature_2.append(np.array(eeg_data_2)[label_index_2].reshape(-1)) #获取对应标签的特征

    fileh5.close()
    fileh5_2.close()

    return sleep_label_feature, sleep_label_feature_2


def main():
    file_index1 = choice(list1)
    file_index2 = choice(list1)
    file_index3 = choice(list13)
    if file_index1 == file_index2:
        file_index2 = choice(list1)
    print(file_index1,file_index2,file_index3)
    ##获取文件中的五个分期对应的特征
    sleep_label_feature, sleep_label_feature_2 = getFeature(h5file + "SLP" + str(file_index1) + ".h5", h5file + "SLP" + str(file_index2) + ".h5")
    #获取两个特征文件的特征相似性矩阵
    heatmap_matrix = featureHeatmap(np.array(sleep_label_feature), np.array(sleep_label_feature_2))
    #保留两位小数
    heatmap_matrix = np.round(heatmap_matrix, 2)
    #画出热力图矩阵
    plot_Heatmap(heatmap_matrix, feature_savepath)

    ##获取文件中的五个分期对应的特征
    sleep_label_feature_diff, sleep_label_feature_3_diff = getFeature(h5file + "SLP" + str(file_index1) + ".h5", h5file + "SLP" + str(file_index3) + ".h5")
    #获取两个特征文件的特征相似性矩阵
    heatmap_matrix_diff = featureHeatmap(np.array(sleep_label_feature_diff), np.array(sleep_label_feature_3_diff))
    #保留两位小数
    heatmap_matrix_diff = np.round(heatmap_matrix_diff, 2)
    #画出热力图矩阵
    plot_Heatmap(heatmap_matrix_diff, feature_savepath+"diff_")

if __name__ == '__main__':
    main()

