#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2025/4/4 17:45
# File    : main.py
# Software: PyCharm
import matplotlib.pylab as plt
import numpy as np

from ch03.funs import step_function, sigmoid, relu, identity_function, init_network, forward

if __name__ == '__main__':
    x = np.array([-1.0, 1.0, 2.0])
    print(step_function(x))

    # x = np.arange(-5.0, 5.0, 0.1)
    # y = step_function(x)
    # plt.plot(x, y)
    # plt.ylim(-0.1, 1.1)  # 指定y轴的范围
    # plt.savefig("step_function.png")
    # plt.show()

    x = np.array([-1.0, 1.0, 2.0])
    print(sigmoid(x))

    # x = np.arange(-5.0, 5.0, 0.1)
    # y = sigmoid(x)
    # plt.plot(x, y)
    # plt.ylim(-0.1, 1.1)  # 指定y轴的范围
    # plt.savefig("sigmoid.png")
    # plt.show()

    x = np.array([-1.0, 1.0, 2.0])
    print(sigmoid(x))

    # x = np.arange(-5.0, 5.0, 0.1)
    # y = relu(x)
    # plt.plot(x, y)
    # plt.ylim(-0.1, 1.1)  # 指定y轴的范围
    # plt.savefig("relu.png")
    # plt.show()

    # A = np.array([1, 2, 3, 4])
    # print(A)
    # print(np.ndim(A))
    # print(A.shape)
    # print(A.shape[0])
    #
    # B = np.array([[1, 2], [3, 4], [5, 6]])
    # print(B)
    # print(B.ndim)
    # print(B.shape)
    # print(B.shape[0])

    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    print(A.shape, B.shape)  # (2, 2) (2, 2)
    print(np.dot(A, B).shape)  # (2, 2)

    A = np.array([[1, 2, 3], [4, 5, 6]])
    B = np.array([[1, 2], [3, 4], [5, 6]])
    print(A.shape, B.shape)  # (2, 3) (3, 2)
    print(np.dot(A, B).shape)  # (2, 2)

    C = np.array([[1, 2], [3, 4]])
    try:
        print(np.dot(A, C).shape)
    except Exception as  e:
        print(e)

    A = np.array([[1, 2], [3, 4], [5, 6]])
    B = np.array([7, 8])
    print(A.shape, B.shape)  # (3, 2) (2,)
    print(np.dot(A, B).shape)  # (3,)

    X = np.array([[1, 2]])
    W = np.array([[1, 3, 5], [2, 4, 6]])
    print(X.shape, W.shape)  # (1, 2) (2, 3)
    y = np.dot(X, W)
    print(y)  # [[ 5 11 17]]

    print("3 层神经网络的实现:")
    X = np.array([1.0, 0.5])
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    B1 = np.array([0.1, 0.2, 0.3])

    # print(X.shape)  # (2,)
    # print(W1.shape)  # (2, 3)
    # print(B1.shape)  # (3,)

    A1 = np.dot(X, W1) + B1
    print(A1)  # [0.3, 0.7, 1.1]

    Z1 = sigmoid(A1)
    print(Z1)  # [0.57444252, 0.66818777, 0.75026011

    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])

    # print(Z1.shape)  # (3,)
    # print(W2.shape)  # (3, 2)
    # print(B2.shape)  # (2,)

    A2 = np.dot(Z1, W2) + B2
    Z2 = sigmoid(A2)

    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])
    A3 = np.dot(Z2, W3) + B3
    Y = identity_function(A3)  # 或者Y = A3
    print(Y)  # [0.31682708 0.69627909]

    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y)  # [ 0.31682708 0.69627909]
