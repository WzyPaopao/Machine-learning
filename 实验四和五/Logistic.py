# -*- coding: UTF-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression

'''
    因为实验报告手册上没有数据说明，就粗略认为数据的最后一列中:
    1 代表马存活
    0 代表马死亡
'''

'''数据读取'''
def openFile(fileName):
    data = np.loadtxt(fileName, delimiter='\t')
    return data

if __name__ == '__main__':
    trainingData = openFile('horseColicTraining.txt')     # 训练数据读取
    testData = openFile('horseColicTest.txt')             # 测试数据读取
    train_feature = trainingData[:, [0, len(trainingData[0])-1]]
    train_result = trainingData[:, len(trainingData[0])-1]
    test_feature = testData[:, [0, len(testData[0])-1]]
    test_result = testData[:, len(testData[0])-1]

    logistic_model = LogisticRegression()
    logistic_model.fit(train_feature, train_result)

    fitted_test = logistic_model.predict_proba(test_feature)[:, 1]
    for i in range(len(fitted_test)):
        if fitted_test[i] >= 0.5:
            fitted_test[i] = 1
        else:
            fitted_test[i] = 0

    counter = 0
    for i in range(len(fitted_test)):
        if fitted_test[i] == test_result[i]:
            counter = counter + 1

    print('prediction' + str(fitted_test))
    print('class' + str(test_result))
    print(str(counter / len(fitted_test) * 100)) + '%'

    # 选取训练数据集的第三列和第四列分别作为x，y轴，绘制散点图
    # viewData_x = trainingData[:, 2]
    # viewData_y = trainingData[:, 3]
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.scatter(viewData_x, viewData_y)
    # plt.show()

    # 选取测试数据集的第三列和第四列分别作为x，y轴，绘制散点图
    viewData_x = testData[:, 2]
    viewData_y = testData[:, 3]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(viewData_x, viewData_y)
    plt.show()