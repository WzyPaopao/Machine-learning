# -*-coding: UTF-8-*-

import numpy as np
from numpy import *
import matplotlib.pyplot as plt

def openFile(fileName):
    data = np.loadtxt(fileName, delimiter='\t')
    return data

def stumpClassify(dataMatrix, dimen, threshVal, threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr, classLabels, D):
    dataMatrix = mat(dataArr); labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0; bestStump = {}; bestClasEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min(); rangeMax = dataMatrix[:,i].max();
        stepSize = (rangeMax-rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt', 'gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals.copy()
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr, classLabels, numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        # print "D:", D.T
        alpha = float(0.5*log((1.0-error)/max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        # print "classEst: ", classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        # print "aggClassEst: ", aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        # print "total error: ", errorRate
        if errorRate == 0.0: break
    return weakClassArr

def adaClassify(datToClass,classifierArr):
    dataMatrix = mat(datToClass)#
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix, classifierArr[i]['dim'], classifierArr[i]['thresh'], classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        # print aggClassEst
    return sign(aggClassEst)

def viewData(array_1, array_2, labelSet):
    # print("hello world")
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(array_1, array_2, 15.0 * labelSet, 15.0 * labelSet)
    plt.show()


if __name__ == '__main__':
    trainData = openFile("horseColicTraining.txt")
    featureData = trainData[:, 0:len(trainData[0])-1]
    classLabels = trainData[:, len(trainData[0])-1]

    testData = openFile("horseColicTest.txt")
    featureData_test = testData[:, 0:len(testData[0])-1]
    classLabels_test = testData[:, len(testData[0])-1]

    # viewData(featureData[:, 1], featureData[:, 3], classLabels)
    viewData(featureData_test[:, 1], featureData_test[:, 3], classLabels_test)

    classifierArr = adaBoostTrainDS(featureData, classLabels, 150)
    prediction = adaClassify(featureData_test, classifierArr)
    # print(prediction)
    # print(classLabels_test)
    counter = 0.0
    for i in range(0, len(classLabels_test)):
        if prediction[i] == classLabels_test[i]:
            counter += 1
    print(str(counter/len(classLabels_test)*100) + '%')