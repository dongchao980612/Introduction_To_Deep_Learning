#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2025/4/4 18:29
# File    : MNIST_train_batch.py
# Software: PyCharm
from ch03.MNIST_funs import get_data, init_network, predict
import  numpy as np
if __name__ == '__main__':
    x, t = get_data()
    network = init_network()
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    batch_size = 100  # 批数量
    accuracy_cnt = 0
    for i in range(0, len(x), batch_size):
        x_batch = x[i:i + batch_size]
        y_batch = predict(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i + batch_size])
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))