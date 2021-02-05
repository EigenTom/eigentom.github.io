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

卷积层中所进行的运算即为 **卷积运算**. 

卷积运算涉及两个变量. 其一是作为输入的原始数据, 一般是一个较大的高维矩阵. 其二是卷积核 (也称 “滤波器” ), 即一个小于原始数据的矩阵. 在计算过程中, 我们将从原始数据中剥离出和卷积核维度一致的子矩阵 (我们也可以将卷积核看作叠加在原始数据之上的 “窗口”, 将剥离过程视为 “窗口的滑动”), 并将其和卷积核进行矩阵乘法, 再将所得矩阵中所有元素的值相加, 即完成一次卷积计算, 得到一个值, 它将作为输出矩阵中的一个元素. 按照给定的 “滑动步长”, 将卷积核从原始数据矩阵的左上角继续移动, 每一次移动均进行一次卷积计算, 直到卷积核矩阵的右下角到达原始数据矩阵的右下角, 计算终止. 而通过计算卷积核横向移动的步长和卷积核在滑动过程中 “换行” 的次数, 我们可以得出输出矩阵的维度.

![cnn-calculation](https://cdn.jsdelivr.net/gh/KirisameMarisaa/KirisameMarisaa.github.io/img/blogpost_images/cnn-calculation.gif)

在计算卷积运算的偏置时, 只需将偏置分别加到输出矩阵的每一个元素上即可:

![20210204221940](https://cdn.jsdelivr.net/gh/KirisameMarisaa/KirisameMarisaa.github.io/img/blogpost_images/20210204221940.png)

<br>

## 2. 卷积层与池化层的实现


<br>

## 3. 卷积神经网络的实现和可视化