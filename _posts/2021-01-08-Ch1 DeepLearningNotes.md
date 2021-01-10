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

其中, $\theta$ 称为激活函数 $f$ 的 **阈值**. 函数的分段条件可被转为 $X \cdot W^{T} + b ~~~(b = -\theta)$. 称 $b$ 为 **偏置**, 控制该感知机被激活的难易程度, $W$ 为 **权重**. 控制各个变量的重要程度. 

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

实际上, 我们并不能使用单层感知机实现异或逻辑门. 不妨这样思考: 假定我们使用单层感知机实现了一个异或逻辑门函数 `XOR(x_1, x_2)`, 则它在几何意义上必定是一个使用线性函数对二维平面的二分. 而在平面上描点观察可知, 我们并不能使用线性函数对其基于函数输出的不同而进行分割 (即将所有的红色点和蓝色点分隔开). 因此由矛盾推出, 单层感知机无法实现异或逻辑门:

![xor-gate](https://cdn.jsdelivr.net/gh/KirisameMarisaa/KirisameMarisaa.github.io@master/img/blogpost_images/xor-gate.jpg)

不过, 我们可以使用多层感知机的叠加层实现对该平面的分割. 简单推导逻辑表达式可知:

<center>

$\mathbf{XOR}(x_1, x_2) = \mathbf{AND} (\mathbf{NAND}(x_1, x_2), \mathbf{OR}(x_1, x_2))$

</center>

使用之前定义的与门, 或门和与非门就可以这样实现异或门:
```
def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
```

形如异或门这样, 叠加了多层的感知机称为 **多层感知机** (`Multi-layered Perceptron`). 在我们的例子中, 异或门由两级含有权重的层和一级输出层组成. 通过叠加层, 感知机可以进行更为灵活的数据分类和表示. 

<br>

## 3. 神经网络的定义和层级结构
感知机可以通过多层叠加表示复杂函数, 适应多种分类问题. 但感知机的权重需要人为设定, 而且不同的权重会影响感知机的分类表现. 神经网络的作用是自动地从数据中 “学习” 并更新到合适的权重参数. 

如下图所示, 神经网络一般有三种分层: 输入层, 中间层 (隐藏层) 和输出层. 神经网络的结构和多层感知机所组成的网络没有区别, 但其本质差异是: 神经网络采用连续函数而非线性分段函数作为激活函数. 

![neural-network-diagram](https://cdn.jsdelivr.net/gh/KirisameMarisaa/KirisameMarisaa.github.io/img/blogpost_images/neural-network-diagram.jpg)

<br>


## 4. 激活函数的定义和常见的激活函数
在介绍感知机时我们已经了解, 激活函数接收所有的输入信号, 并将输入信号的总和转换为输出信号, 而其核心作用在于决定如何激活输入信号的总和. 
下面, 我们介绍数个常用的神经网络激活函数:

1. `Sigmoid` 函数<br>
   `Sigmoid` 函数 $s(x)$:

   <center>

    $s(X) = \frac{1}{1 + \exp^{(-x)}}.$

   </center>

在 `Python` 中, `Sigmoid` 函数实现如下:
```
def sigmoid(X):
    return 1/ (1 + numpy.exp(-x))
```



2. `ReLU` 函数<br>
    `ReLU` 函数 $r(x)$: 

    <center>
    
    $r(X) = \begin{cases} x ~~~ (x > 0) \\ 0 ~~~ (x \leqslant 0)\end{cases}$

    </center>

在 `Python` 中, `ReLU` 函数实现如下:
```
def relu(X):
    return numpy.maximum(0, x)
```

<br>


## 5. 三层神经网络的实现



<br>

## 6. 神经网络的输出

