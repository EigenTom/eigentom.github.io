---
layout:     post
title:      从零开始的深度学习 Ch5
subtitle:   Introduction to Convolutional Neural Network
date:       2021-01-10
author:     R1NG
header-img: img/post-bg-deeplearning.jpg
description: 卷积神经网络
catalog: true
tags:
    - 扩展自习
    - 2020
---


# 卷积神经网络<br>

卷积神经网络的结构和之前所介绍的神经网络相似, 都可以通过对不同层的组合实现网络的构建. 在卷积神经网络中, 我们新引入了 **卷积 (`Convolution`)** 层和 **池化 (`Pooling`)** 层. 

在详细介绍卷积层和池化层的原理与实现之前, 我们首先简述通过组装层构建 `CNN` 的方法. 对于常规的神经网络而言, 连接层可以用 `Affine` 层实现, 而激活函数层中可以使用 `ReLU` 函数, 在最后的输出层中, 激活函数可以选用 `SoftMax` 函数. 

而对卷积神经网络而言, 其层的连接顺序是: 卷积层 - 激活函数层 (如 `ReLU` 层) - 可省略的池化层, 如下图所示:

![20210131231833](https://cdn.jsdelivr.net/gh/KirisameMarisaa/KirisameMarisaa.github.io/img/blogpost_images/20210131231833.png)

使用全连接层的神经网络会丢弃输入数据的维度信息. 无论作为数据集的原始数据先天具有何种维度 (如三维的图像) , 它都将被统一转换为一维数据, 由此丢失了一些可能对学习而言至关重要的空间信息. 相比之下, 卷积层可以保持输入数据的维度信息不变. 当输入数据为具有三个维度的图像时, 卷积层会以三维数据的形式接收它, 并以同样的维度输出至下一层. 因此, 卷积神经网络相比常规的全连接神经网络更有可能正确理解和提炼出图像等多维数据中隐藏的信息. 

在卷积神经网络中, 卷积层的输入输出数据又称为 **特征图 (`Feature Map`)**, 卷积层的输入数据称为 **输入特征图 (`Input Feature Map`)**, 其输出数据称为 **输出特征图 (`Output Feature Map`)**. 

<br>

## 1. 卷积层和池化层


<br>

## 2. 卷积层与池化层的实现


<br>

## 3. 卷积神经网络的实现和可视化