---
layout:     post
title:      从零开始的深度学习 Ch1
subtitle:   Perceptron & Neural Network 
date:       2021-01-08
author:     R1NG
header-img: img/post-bg-algorithm.jpg
description: 感知机, 神经网络的定义, 实现与扩展
catalog: true
tags:
    - 扩展自习
    - 2020
---

# 从零开始的深度学习: 感知机与神经网络
## 1. 感知机的定义
动物的神经系统中最基础的组成单元为神经元 (神经细胞), 用于接受刺激, 产生兴奋并传导兴奋. 神经元有且只有激活态和非激活态两种状态, 并且只有神经元处于激活态时, 传入的兴奋才会由它传出. 

感知机是神经网络最基本的组成单元, 其本质是以特征向量为自变量的分段函数, 能够完整地模拟神经元的逻辑功能. 

一个标准的感知机包含以下三个组成部分:
1. 输入 (`Input`)
2. 与每一个输入所对应的权值 (`Weights`)
3. 激活函数 (`Function`)

![perceptron](https://cdn.jsdelivr.net/gh/KirisameMarisaa/KirisameMarisaa.github.io@master/img/blogpost_images/perceptron.jpg)

我们定义:

<center>


$X = (1, x_1, \cdots, x_n),~~ W = (w_0, w_1, \cdots, w_n).$


</center>

其中, $X$ 为激活函数的输入, $W$ 为权值. 则激活函数形为:

<center>


$f(X) = \begin{cases}
      0 ~~~~~~ X \cdot W^T \leqslant \theta \\ 
      1 ~~~~~~ X \cdot W^T > \theta
\end{cases}$


</center>

其中, $\theta$ 成为激活函数 $f$ 的 **阈值**. 

不难看出, 激活函数 $f$ 是一个线性分段函数, 该函数的行为可看作对一个二维平面使用一条直线进行分割. 激活函数的线性性质决定了它无法对二维平面进行较复杂的分割, 这一问题在我们使用感知机处理某些分类问题时会立即凸显, 比如使用感知机实现异或逻辑门. 首先, 我们来看几个简单的例子:

[例] 使用感知机实现与, 或, 与非门:

与, 或, 与非门的真值表如下:

|$x_1$|$x_2$|AND|OR|NAND|
|:-:|:-:|:-:|:-:|:-:|
|$1$|$1$|$1$|$1$|$0$|
|$1$|$0$|$0$|$1$|$1$|
|$0$|$1$|$0$|$1$|$1$|
|$0$|$0$|$0$|$0$|$1$|


```
import numpy as np


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1
```
将与门, 或门, 与非门的两个输入 $x_1, x_2$ 视为平面直角坐标系的两轴, 将平面上的点视为所定义的逻辑门函数的两个输入, 可以看出: 对于上述的三种逻辑门电路而言, 其输入-输出分别将平面划分为了两个部分, 且这样的划分是线性的. 

<br>

## 2. 感知机的局限性和扩展的多层感知机: 以异或门电路的实现为例
下面, 我们以异或门电路的分析和实现说明感知机 (单层感知机) 的局限性. 

异或逻辑门的真值表如下:

|$x_1$|$x_2$|XOR|
|:-:|:-:|:-:|
|$1$|$1$|$0$|
|$1$|$0$|$1$|
|$0$|$1$|$1$|
|$0$|$0$|$0$|

实际上, 我们并不能使用单层感知机实现异或逻辑门. 不妨这样思考: 假定我们使用单层感知机实现了一个异或逻辑门函数 `XOR(x_1, x_2)`, 则它在几何意义上必定是一个使用线性函数对二维平面的二分. 而在平面上描点观察可知, 我们并不能使用线性函数对其基于函数输出的不同而进行分割, 因此由矛盾推出, 单层感知机无法实现异或逻辑门:

![xor-gate](https://cdn.jsdelivr.net/gh/KirisameMarisaa/KirisameMarisaa.github.io@master/img/blogpost_images/xor-gate.jpg)

<br>

## 3. 神经网络的定义和层级结构


<br>

## 4. 激活函数的定义和常见的激活函数


<br>

## 5. 三层神经网络的实现


<br>

## 6. 神经网络的输出

