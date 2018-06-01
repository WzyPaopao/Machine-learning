#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    '''
        用numpy的包读取数据为一个数组
        属性：花萼长度，花萼宽度，花瓣长度，花瓣宽度
    '''
    data_array = np.loadtxt("iris.data.txt", delimiter=',')
    '''选择花瓣长度'''
    data = data_array[:, 2]

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签

    '''散点图，以第一列数据和第二列数据分别作为x，y轴，用蓝色和*标记'''
    # plt.plot(data_array[:, 0], data_array[:, 1], "b*")
    # plt.xlabel(u"花萼长度")
    # plt.ylabel(u"花萼宽度")

    '''直方图，以第一列数据及其分布分别作为x，y轴'''
    # plt.hist(data_array[:, 0], 50)
    # plt.xlabel(u'花萼长度')

    '''箱型图，一第一列数据绘制'''
    # plt.boxplot([data_array[:, 0]], labels=[u'长度'])
    # plt.title(u'花萼长度')
    # plt.show()

    '''茎叶图'''
    plt.stem(data_array[:, 0], data_array[:, 1])
    plt.show()

    plt.show()