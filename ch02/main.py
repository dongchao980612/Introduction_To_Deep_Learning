#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# Time    : 2025/4/4 17:37
# File    : main.py
# Software: PyCharm
from ch02.perceptron import AND, XOR

if __name__ == '__main__':
    print(AND(0, 0))  # 输出0
    print(AND(1, 0))  # 输出0
    print(AND(0, 1))  # 输出0
    print(AND(1, 1))  # 输出1

    print(XOR(0, 0))  # 输出0
    print(XOR(1, 0))  # 输出1
    print(XOR(0, 1))  # 输出1
    print(XOR(1, 1))  # 输出0

