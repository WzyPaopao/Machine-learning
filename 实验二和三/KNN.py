# -*- coding: UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

'''
    将文件中的数据导入矩阵
    :return 字符串矩阵，矩阵长度
'''
def openFile():
    file_object = open('lenses.txt')
    fileContext = file_object.readlines()
    tempData = np.array(fileContext)
    # print tempData
    size = tempData.size
    # print size
    data = np.zeros([size, 5], dtype=basestring)
    for i in range(0, size):
         for j in range(0, 5):
             data[i][j] = tempData[i].split('\t')[j]
             if j == 4:
                 data[i][j] = data[i][j].strip('\n')
    # print data
    return data, size

'''将矩阵的文本属性数据量化，并归一化'''
def turnTheMatrix(data, size):
    myData = np.zeros([size, 5])
    operator = [{'young':0, 'pre':0.5, 'presbyopic': 1}, {'myope': 0, 'hyper': 1}, {'yes': 0, 'no': 1},
                {'reduced': 0, 'normal': 1}, {'no lenses':0, 'soft':0.5, 'hard':1}]
    for j in range(0, 5):
        for i in range(0, size):
            myData[i][j] = operator[j].get(data[i][j])

    return myData
    # print myData

'''绘制散点图，并划分类别'''
def createPlot(data_array, dataLabels):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(data_array[:, 0], data_array[:, 2], 15.0*dataLabels, 15.0*dataLabels)
    plt.show()

'''
    分类器测试
    :parameter 
        data, 训练数据
        dataLabels，分类标签
        testData，需要分类的预测数据
    
'''
def classifierTest(data, dataLabels, testData_num):
    length = np.zeros([data.size / 5, 2])
    # '''分割测试数据，并数据量化'''
    # testDataArray = testData.strip('\n').split('\t')
    # operator = [{'young': 0, 'pre': 0.5, 'presbyopic': 1}, {'myope': 0, 'hyper': 1}, {'yes': 0, 'no': 1},
    #             {'reduced': 0, 'normal': 1}, {'no lenses': 0, 'soft': 0.5, 'hard': 1}]
    # testData_num = np.zeros(4)
    # for i in range(0, 4):
    #     testData_num[i] = operator[i].get(testData[i])

    '''计算距离'''
    for i in range(0, data.size / 5):
        for j in range(0, 4):
            length[i][0] = length[i][0] + (testData_num[j] - data[i][j])**2
            if j == 3:
                length[i][0] = length[i][0] ** 0.5    #开根号
                length[i][1] = dataLabels[i]

    '''排序'''
    for i in range(0, len(length)-1):
        for j in range(0, len(length)-i-1):
            if length[j][0] > length[j+1][0]:
                temp = length[j][0]
                length[j][0] = length[j+1][0]
                length[j+1][0] = temp
                temp = length[j][1]
                length[j][1] = length[j + 1][1]
                length[j + 1][1] = temp

    return length

def classify(length, k):
    Len = len(length)
    kind_00 = 0
    kind_05 = 0
    kind_10 = 0
    for i in range(0, k):
        if length[Len-i-1][1] == 0:
            kind_00 = kind_00 + 1
        elif length[Len-i-1][1] == 0.5:
            kind_05 = kind_05 + 1
        elif length[Len-i-1][1] == 1:
            kind_10 = kind_10 + 1

    '''求最大值'''
    if kind_00 >= kind_05:
        max = kind_00
        max_id = 0
    else:
        max = kind_05
        max_id = 0.5
    if max <= kind_10:
        max = kind_10
        max_id = 1

    '''分类'''
    dic = {0: 'no lenses', 0.5: 'soft', 1: 'hard'}
    pre_class = dic.get(max_id)

    # print 'the predict class is: ' + pre_class
    return pre_class


if __name__ == '__main__':
    # testData = 'young	hyper	yes	normal	hard'
    data, size = openFile()    # data为字符串矩阵
    myData = turnTheMatrix(data, size)       # myData为归一化后的矩阵
    # print myData
    dataLabels = myData[:, 4]
    # createPlot(myData, dataLabels)
    # print data
    counter = 0.0
    k = 16           # K取值
    for testData in myData:
        length = classifierTest(myData, dataLabels, testData)
        dic = {0: 'no lenses', 0.5: 'soft', 1: 'hard'}
        rightClass = dic.get(testData[4])
        pre_class = classify(length, k)
        print 'predict: ' + pre_class + ' --- right: ' + rightClass

        if pre_class == rightClass:
            counter = counter + 1

    print '\n--------- the prediction accuracy is ' + str( round(counter/len(myData), 4)*100 ) + '%'