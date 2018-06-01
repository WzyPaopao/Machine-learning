#!/usr/bin/env python
#-*- coding: utf-8 -*-

# @Time    : 2017/10/18 14:50
# @Author  : WenMin
# @Email    : < wenmin593734264@gmial.com >
# @File    : test.py
# @Software: PyCharm Community Edition

import numpy as np
import pandas as pd
from scipy.stats import mode
import matplotlib.pyplot as plt


# 计算均值
def get_mean_value(array_data):
    """
    :param array_data:
    :return: 样本均值
    """
    return np.mean(array_data)


# 计算众数
def get_mode_value(array_data):
    """
    :param array_data:
    :return: 众数
    """
    return mode(array_data)


# 计算方差
def get_var_value(array_data):
    """
    :param array_data:
    :return: 方差
    """
    return np.var(array_data)


# 计算标准差
def get_std_value(array_data):
    """
    :param array_data:
    :return:标准差
    """
    return np.std(array_data)


# 计算中位数
def get_median_value(array_data):
    """
    :param array_data:
    :return: 中位数
    """
    return np.median(array_data)


# 计算极差
def get_ptp_value(array_data):
    """
    :param array_data:
    :return:极差
    """
    return np.ptp(array_data)


# 计算Z-分数_（偏差程度）
def get_z_value(array_data, location):
    """
    :param array_data:
    :param location:数组某个位置索引
    :return: 索引位置当前数值的偏差程度
    """
    return (array_data[location]-np.mean(array_data))/np.std(array_data)


# 计算变异系数
def get_cv(array_data):
    """
    :param array_data:
    :return: 变异系数
    """
    return np.mean(array_data)/np.std(array_data)


'''定义
类型A：值域为[1,2)的所有数；
类型B：值域为[2,3)的所有数；
类型C：值域为[3,4)的所有数；
类型D：值域为[4,5)的所有数；
类型E：值域为[5,6)的所有数；
'''

# 绘制柱状图
def draw_bar(array_data):
    """
    绘制柱状图的前提是要做频率统计
    :param array_data: pandas DataFrame type
    :return:
    """
    # 以pandas读取为dataframe的形式，对每个范围里的数据进行统计
    a = array_data[(array_data['column'] >= 1) & (array_data['column'] < 2)].count()
    b = array_data[(array_data['column'] >= 2) & (array_data['column'] < 3.5)].count()
    c = array_data[(array_data['column'] >= 3.5) & (array_data['column'] < 4)].count()
    d = array_data[(array_data['column'] >= 4) & (array_data['column'] < 5)].count()
    e = array_data[(array_data['column'] >= 5) & (array_data['column'] < 7)].count()
    xticks = ['a', 'b', 'c', 'd', 'e']
    # 第一个参数为柱的横坐标， 第二个参数为柱的高度， 地上那个参数为对齐方式
    plt.bar(range(5), [list(a)[0], list(b)[0], list(c)[0], list(d)[0], list(e)[0]], align='center')
    # 设置柱的文字说明;第一个参数为文字说明的横坐标,第二个参数为文字说明的内容
    plt.xticks(range(5), xticks)
    # 设置横坐标的文字说明
    plt.xlabel('flower_wide')
    # 设置纵坐标的文字说明
    plt.ylabel('Frequency')
    # 设置标题
    plt.title('Flower wide count')
    plt.show()

# 绘制饼形图
def draw_pie(array_data):
    """
    以pandas读取为dataframe的形式，对每个范围里的数据进行统计
    :param array_data:
    :return:
    """
    a = array_data[(array_data['column'] >= 1) & (array_data['column'] < 2)].count()
    b = array_data[(array_data['column'] >= 2) & (array_data['column'] < 3.5)].count()
    c = array_data[(array_data['column'] >= 3.5) & (array_data['column'] < 4)].count()
    d = array_data[(array_data['column'] >= 4) & (array_data['column'] < 5)].count()
    e = array_data[(array_data['column'] >= 5) & (array_data['column'] < 7)].count()
    labels = ['a', 'b', 'c', 'd', 'e']
    # 第一个参数为柱的横坐标， 第二个参数为柱的高度， 地上那个参数为对齐方式
    plt.pie([list(a)[0], list(b)[0], list(c)[0], list(d)[0], list(e)[0]], labels=labels, autopct='%1.1f%%')
    plt.title('Flower wide count')
    plt.show()


# 绘制直方图
def draw_hist(array_data):
    """
    :param array_data: 一般的列表形式，不是dataFrame形式
    :return:
    """
    # 第一个参数为待绘制的定量数据，不同于定性数据，这里并没有事先进行频数统计
    # 第二个参数为划分的区间个数
    plt.hist(array_data, 50)
    plt.xlabel('Heights')
    plt.ylabel('Frequency')
    plt.title('Heights Of Male Students')
    plt.show()


#  绘制累计曲线
def draw_cumulative_hist(array_data):
    """
    :param array_data:
    :return:
    """
    # 创建累积曲线
    # 第一个参数为待绘制的定量数据,第二个参数为划分的区间个数
    # normed参数为是否无量纲化
    # histtype参数为'step'，绘制阶梯状的曲线, cumulative参数为是否累积
    plt.hist(array_data, 20, normed=True, histtype='step', cumulative=True)
    plt.xlabel('wide')
    plt.ylabel('Frequency')
    plt.title('wide Of Flowers')
    plt.show()


# 绘制散点图
def draw_scatter(array_data):
    """
    :param array_data:
    :return:
    """
    # 创建散点,第一个参数为点的横坐标, 第二个参数为点的纵坐标
    # 因为数据是单变量的，我们主要关心第二个参数：花瓣的宽度值的分布情况，第一个参数以长度占位
    plt.scatter(range(len(array_data)), array_data)
    plt.ylabel('wide')
    plt.show()


# 绘制箱形图
def draw_box(array_data):
    """
    :param array_data:
    :return:
    """
    # 创建箱形图, 第一个参数为待绘制的定量数据,第二个参数为数据的文字说明
    plt.boxplot([array_data], labels=['Wide'])
    plt.title('wide of Flowers')
    plt.show()


if __name__ == '__main__':
    '''用numpy的包读取数据为一个数组'''
    data_array = np.loadtxt("iris.data.txt", delimiter=',')
    '''这里是单变量分析，我们只考虑其中某一列的数据，所以选择数组的第三列为一个列表'''
    data = data_array[:, 2]

    '''将列表数据转换为pandas能够处理的形式'''
    da = pd.DataFrame(data, columns=['column'])
    ''' 一个快速得到当前数据的概况方法，计算统计数据长度、样本均值、方差、最小值、最大值、四分位数'''
    # print (da['column'].describe())

    # 绘制柱状图
    #draw_bar(da)
    # 绘制饼形图
    #draw_pie(da)
    # 绘制直方图
    #draw_hist(data)
    # 绘制累计曲线图
    #draw_cumulative_hist(data)
    # 绘制散点图
    #draw_scatter(data)
    # 绘制箱形图
    draw_box(data)

    # 计算均值
    #mean_value = get_mean_value(data)
    #print (mean_value)
    # 计算中位数
    #median_value = get_median_value(data)
    #print (median_value)
    # 计算众数
    #mode_value = get_mode_value(data)
    #print (mode_value)
    # 计算样本方差
    #var_value = get_var_value(data)
    #print (var_value)
    # 计算标准差
    #std_value = get_std_value(data)
    #print (std_value)
    # 计算极差
    #ptp_value = get_ptp_value(data)
    #print (ptp_value)
    # 计算变异系数
    #cv_value = get_cv(data)
    #print (cv_value)
    # 计算Z分数
    #z_0_value = get_z_value(data, 0)
    #print (z_0_value)