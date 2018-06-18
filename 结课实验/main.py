# -*-coding: UTF-8-*-

from numpy import *
import matplotlib.pyplot as plt
import random

'''加载文件，返回数据集和标签集'''
def openFile(fileName):
    dataSet = loadtxt(fileName, str, delimiter=',')
    data = dataSet[1:, 0:len(dataSet[0])-1].astype(float)
    label = dataSet[1:, len(dataSet[0])-1].astype(float)
    # print(dataSet)
    return data, label

'''选取两列数据进行数据可视化'''
def viewData():
    data, label = openFile("train.csv")
    # 放大标签之间的视觉差异
    label[label == 0] = 0.1
    label[label == 1] = 3

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data[:, 2], data[:, 17], 15.0*label, 15.0*label)
    plt.show()

'''求两个点之间的欧氏距离'''
def getDistance(aimerArr, baseArr):
    l = len(aimerArr)
    distance = 0
    for i in range(l):
        distance += (aimerArr[i] - baseArr[i])**2
        if i == l-1:
            distance **= 0.5
    return distance

'''   KNN   '''
def KNN(k, dataSet, labelSet, aimData):
    row, line = shape(dataSet)
    distance = zeros((2, row))
    for i in range(row):
        distance[0][i] = i
        distance[1][i] = getDistance(aimData, dataSet[i])
    distance = distance.T[distance.T[:, 1].argsort()].T

    counter_normal = 0
    counter_liar = 0
    # print(labelSet)
    for i in range(k):
        if labelSet[distance[0][i]] == 0:
            counter_normal += 1
        else:
            counter_liar += 1
    if counter_normal > counter_liar:
        flag = 0
    else:
        flag = 1
    return flag

'''主函数'''
if __name__ == '__main__':
    print("hello world")
    trainData, trainLabel = openFile("train.csv")
    testData = loadtxt("test.csv", str, delimiter=',')
    testData= testData[1:, :].astype(float)
    # print(trainLabel)
    # viewData()
    k = 51
    random_num = 200
    row, line = shape(trainData)
    # trainData = trainData[0:(row/200), :]
    # trainLabel = trainLabel[0:(row/200)]
    randomTrainData = zeros((random_num, line))
    randomTrainLabel = zeros(random_num)
    '''随机抽取5000条数据'''
    last_index = -1
    for i in range(random_num):
        index = random.randint(0, row - 1)
        while index == last_index:
            index = random.randint(0, row - 1)
        last_index = index
        for j in range(line):
            randomTrainData[i][j] = trainData[index][j]
        randomTrainLabel[i] = trainLabel[index]

    row, line = shape(trainData)
    n, m = shape(testData)
    # prediction = zeros(row)
    # counter = 0.0001
    prediciton = zeros(n)
    for i in range(n):
        prediciton[i] = KNN(k, randomTrainData, randomTrainLabel, testData[i])
    #     if trainLabel[i] == KNN(k, trainData, trainLabel, testData[i]):
    #         counter += 1
    #
    # print(counter/row)
    # print(dis)
    # print(prediciton)

    myCsvFile = zeros((n, 2), dtype=int)
    # myCsvFile[0][0] = 'user_id'
    # myCsvFile[0][1] = 'label'
    for i in range(n):
        myCsvFile[i][0] = testData[i][0]
        myCsvFile[i][1] = prediciton[i]
        if prediciton[i] == 1:
            print("-----------------------")

    savetxt('201531060235.csv', myCsvFile, fmt='%d', delimiter=',', header='user_id,label')

    print("hello world")