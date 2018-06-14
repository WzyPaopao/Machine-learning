# -*-coding: UTF-8-*-

import numpy as np
import matplotlib.pyplot as plt

'''加载文件，返回数据集和标签集'''
def openFile(fileName):
    dataSet = np.loadtxt(fileName, np.str, delimiter=',')
    data = dataSet[1:, 0:len(dataSet[0])-1].astype(np.float)
    label = dataSet[1:, len(dataSet[0])-1].astype(np.float)
    # print(dataSet)
    return data, label

def viewData():
    data, label = openFile("train.csv")
    # 放大标签之间的视觉差异
    label[label == 0] = 0.1
    label[label == 1] = 3

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 2], data[:, 17], 15.0*label, 15.0*label)
    plt.show()

if __name__ == '__main__':
    print("hello world")
    # trainData, trainLabel = openFile("train.csv")
    # print(trainLabel)
    viewData()
