# -*-coding: utf-8-*-

from numpy import *
import random
import math

'''打开文件，返回特征集和标签集'''
def openFile(fileName):
    data = loadtxt(fileName, delimiter='\t')
    dataSet = data[:, 0:len(data[0])-1]
    labelSet = data[:, len(data[0])-1]
    return dataSet, labelSet

'''随机初始化中心点'''
def initTheCenter(class_num, dataSet):
    row, line = shape(dataSet)
    # print(row, line)
    maxArr = zeros(line)
    minArr = zeros(line)
    for i in range(line):
        maxArr[i] = max(dataSet[:, i])
        minArr[i] = min(dataSet[:, i])
        # print(maxArr[i])

    center = zeros((class_num, line))
    # print(shape(center))
    for i in range(class_num):
        for j in range(line):
            # print("i:", i)
            # print("j:", j)
            center[i][j] = random.randrange(math.ceil(minArr[j]), int(maxArr[j]))
    return center

'''标记数据'''
def markThePoint(dataSet, initCenter):
    row, line = shape(dataSet)
    row_center, line_center = shape(initCenter)
    distination = zeros((row, row_center))

    for i in range(row_center):
        for j in range(row):
            for k in range(line):
                distination[j][i] += (dataSet[j][k] - initCenter[i][k])**2

    for i in range(row):
        for j in range(row_center):
            distination[i][j] **= 0.5

    prediction_label = zeros(row)
    for i in range(row):
        for j in range(row_center):
            if min(distination[i]) == distination[i][j]:
                prediction_label[i] = j

    return prediction_label

'''对同类数据的同种特征取平均值，得出新的中点'''
def getTheNewCenter(markedList, dataSet):
    l = len(markedList)
    row, line = shape(dataSet)
    markList = zeros((2, l))
    for i in range(l):
        markList[0][i] = i
        markList[1][i] = markedList[i]

    newCenter = zeros((2, l), dtype=float)
    for i in range(l):
        for j in range(line):
            if markList[1][i] == 1:
                newCenter[0][i] += dataSet[i][j]
            newCenter[0][i] /= l

if __name__ == '__main__':
    print("hello world")
    dataSet, labelSet = openFile("horseColicTraining.txt")
    tempSet = dataSet
    # tempSetk = 77
    for k in range(200):
        for i in range(k+1):
            initCenter = initTheCenter(2, tempSet)
            prediction = markThePoint(dataSet, initCenter)
            getTheNewCenter(prediction, dataSet)
            # print(prediction)
        l = len(prediction)
        label_1 = zeros((2, l))
        label_2 = zeros((2, l))

        count_1 = 0
        count_2 = 0
        for i in range(l):
            if prediction[i] == 1:
                label_1[0][count_1] = prediction[i]
                label_1[1][count_1] = labelSet[i]
                count_1 += 1
            elif prediction[i] == 0:
                label_2[0][count_2] = prediction[i]
                label_2[1][count_2] = labelSet[i]
                count_2 += 1

        plabel_1 = zeros((2, count_1))
        plabel_2 = zeros((2, count_2))

        counter = 0.0
        for i in range(count_1):
            plabel_1[0][i] = label_1[0][i]
            plabel_1[1][i] = label_1[1][i]
            if label_1[1][i] == 0:
                counter += 1

        if count_1 == 0: continue
        a = counter / count_1

        counter = 0.0
        for i in range(count_2):
            plabel_2[0][i] = label_2[0][i]
            plabel_2[1][i] = label_2[1][i]
            if label_2[1][i] == 1:
                counter += 1

        if count_2 == 0: continue
        b = counter / count_2
        # print("----------------------------------------")

        if a + b > 1.5 or (a > 0.75 and b > 0.75):
            print(a, b)
            break

    # print(plabel_1)
    # print(plabel_2)