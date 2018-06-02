# -*-coding: utf-8-*-

import numpy as np

'''将文件中的数据读取至矩阵'''
def openFile(filename):
    file_object = open(filename)
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

def counter(lineNum, aimer_1, data, size, aimer_2 = None):
    if aimer_2 == None:
        temp = 0
        for i in range(0, size):
            if data[i][lineNum] == aimer_1:
                temp = temp + 1
    else:
        temp = 0
        for i in range(0, size):
            if data[i][lineNum] == aimer_1 and data[i][4] == aimer_2:
                temp = temp + 1

    return temp

def pOfClass(data, size):
    c = np.zeros(3, dtype=float)
    p = np.zeros(3, dtype=float)
    for i in range(size):
        if data[i][4] == 0:
            c[0] = c[0] + 1
        elif data[i][4] == 0.5:
            c[1] = c[1] + 1
        elif data[i][4] == 1:
            c[2] = c[2] + 1

    for i in range(3):
        p[i] = c[i] / size

    return c, p

def classifier(data, testData, size):
    # print '\n' + str(testData)
    probability = np.ones(3, dtype=float)   # 预测概率数组
    condition = testData   # 目标条件数组

    # 综合条件的概率
    temp = 0
    for i in range(size):
        flag = True
        for j in range(0, 4):
            if data[i][j] != condition[j]:
                flag = False
                break
        if flag == True:
            temp = temp + 1
    # print 'temp = ' + str(temp)

    # 分类目标的概率
    num_result, pResult = pOfClass(data, size)

    # P(条件 | 分类目标类型)
    p_condition = np.zeros([3, 4], dtype=float)
    feature = 0.0
    for j in range(3):
        for i in range(0, 4):
            count = counter(i, condition[i], data, size, feature)
            p_condition[j][i] = count / num_result[j] * 1.0
            # print '-----' + str(count) + '--' +  str(num_result[j])
        feature = feature + 0.5
    # print '=====' + str(p_condition)

    #各种情况的概率
    for q in range(3):
        for j in range(0, 4):
            probability[q] = probability[q] * p_condition[q][j]
        probability[q] = probability[q] * pResult[q] / (temp / size + 0.0000001)   # 分母加0.0000001来避免分母为零

    pred = 0
    if probability[0] >= probability[1] and probability[0] >= probability[2]:
        pred = 0
    elif probability[1] >= probability[0] and probability[1] >= probability[2]:
        pred = 0.5
    elif probability[2] >= probability[0] and probability[2] >= probability[1]:
        pred = 1


    return pred

if __name__ == '__main__':
    sourceData, size = openFile('lenses.txt')
    data = turnTheMatrix(sourceData, size)
    #print data
    prediction = np.zeros(size, dtype=float)
    temp = 0.0
    for i in range(size):
        prediction[i] = classifier(data, data[i], size)
        if prediction[i] == data[i][4]:
            temp = temp + 1
    print 'prediction:\n' + str(prediction)
    print 'class:\n' + str(data[:, 4])
    print 'the precision: ' + str(round(temp / size * 100, 2)) + '%'