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


def cm_plot(original_label, predict_label, kunm, savepath):

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

    plt.savefig(savepath + "matrix" + str(kunm) + ".svg")
    plt.show()
    plt.close()
    return kappa(cm), classification_report(original_label, predict_label)