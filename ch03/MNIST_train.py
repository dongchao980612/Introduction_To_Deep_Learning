#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2025/4/4 18:13
# File    : MNIST_train.py
# Software: PyCharm
import os
import sys
import  numpy as np
from ch03.MNIST_funs import get_data, init_network, predict
# from ch03.funs import init_network
from ch03.mnist import load_mnist

sys.path.append(os.pardir)  # 为了导入父目录中的文件而进行的设定
if __name__ == '__main__':

    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,
                                                      normalize=False)
    # 输出各个数据的形状
    print(x_train.shape)  # (60000, 784)
    print(t_train.shape)  # (60000,)
    print(x_test.shape)  # (10000, 784)
    print(t_test.shape)  # (10000,)

    img = x_train[0]
    label = t_train[0]
    print(label)  # 5
    print(img.shape)  # (784,)
    img = img.reshape(28, 28)  # 把图像的形状变成原来的尺寸
    print(img.shape)  # (28, 28)
    # img_show(img)

    x, t = get_data()
    network = init_network()
    accuracy_cnt = 0
    for i in range(len(x)):
        y = predict(network, x[i])
        p = np.argmax(y)  # 获取概率最高的元素的索引
        if p == t[i]:
            accuracy_cnt += 1
        print("Accuracy:" + str(float(accuracy_cnt) / len(x)))