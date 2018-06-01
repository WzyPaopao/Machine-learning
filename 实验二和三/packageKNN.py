# -*- coding: UTF-8 -*-

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

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

if __name__ == '__main__':
    data, size = openFile()
    myData = turnTheMatrix(data, size)
    featureSet = myData[:, 0:3]
    classSet = myData[:, 4]
    # print featureSet, classSet
    knn = KNeighborsClassifier()
    knn.fit(featureSet, classSet)
    prediction = knn.predict(featureSet)

    counter = 0.0
    for i in range(0, len(classSet)):
        if prediction[i] == classSet[i]:
            counter = counter + 1

    print 'prediction:'
    print prediction
    print 'class'
    print classSet
    print str(counter / len(classSet) * 100) + '%'
