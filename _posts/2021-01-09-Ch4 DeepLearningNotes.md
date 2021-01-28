---
layout:     post
title:      从零开始的深度学习 Ch4
subtitle:   Introducing numerous optimization methods for better neural network performance
date:       2021-01-09
author:     R1NG
header-img: img/post-bg-deeplearning.jpg
description: 神经网络优化算法技巧
catalog: true
tags:
    - 扩展自习
    - 2020
---

# 神经网络优化算法技巧
在前一章的介绍中, 我们已经明白, 神经网络学习的本质是一个优化问题: 通过多轮迭代对参数进行最优化, 从而使得损失函数的值尽可能的小. 和普通的最优化问题不同, 神经网络的最优化问题涉及的参数空间复杂, 无法使用解析的方式求得最优解, 而在深度神经网络中, 参数的规模则更加庞大. 

上一章中, 我们简述了随机梯度下降法作为神经网络训练过程中的参数优化方法. 在本节中, 我们将继续介绍其余几种更复杂, 性能更好的参数优化方法. 

<br>

## 1. 随机梯度下降法 `SGD`

随机梯度下降法在上一章中已经简要介绍. 在介绍新方法之前, 我们先简要回顾一下: 

`SGD` 的核心原理如下式所示:

$$W \leftarrow W - \eta \frac{\partial{L}}{\partial{W}}$$

此处, $W$ 为待更新的权重参数, $\eta$ 为每一次更新所执行的优化步长 (学习率), $\frac{\partial{L}}{\partial{W}}$ 为损失函数 $L$ 关于 $W$ 的偏导数. 

其 `Python` 实现如下:

~~~python
class SGD:
    def __init__(self, lr=0.01):
    
    # lr: learning rate, set to 0.01 here as an example

        self.lr = lr
    
    def update(self, params, grads):
        for key in params.keys():
            params[key] -= self.lr * grads[key]
~~~

`SGD` 的优化思想很简单: 始终认为梯度指向极值, 按照它的方向不断行进, 最后就会到达极值处. 而当函数的梯度并不和假设一样指向极值时, 其效率就会明显下降. 考虑一个曲面或超曲面, 使用 `SGD` 方法求其最低点时, 极有可能陷入曲面的鞍点处而难以逃脱:

![20210128224300](https://cdn.jsdelivr.net/gh/KirisameMarisaa/KirisameMarisaa.github.io/img/blogpost_images/20210128224300.png)

下面, 我们介绍几个更优化的方法. 

<br>

## 2. 动量方法 `Momentum`

动量方法是物理模型和数学意义的结合, 它将参数进行最优化的过程假定为参数取值点在高低不一的平面或超平面上运动的过程, 引入了 “动量“ 的概念, 这与其名称对应. 其核心原理如下式所示:

$$v \leftarrow \alpha v - \eta \frac{\partial{L}}{\partial{W}}$$

$$W \leftarrow W + v$$

此处出现的新变量 $v$ 对应物体运动的速度, $\alpha$ 对应和运动方向相反的摩擦阻力, 而 $\alpha v$ 即为动量的改变量. 在动量方法的优化过程中, 参数取值点除了会逐渐向梯度所指向的方向移动外, 其来回往复的运动行为会被抑制, 而单向的运动行为会被强化, 由此可以更快地从鞍点处逃逸. 

其 `Python` 实现如下:
~~~python
class Momentum:
    def __init__(self, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.v = None

    def update(self, params, grads):

        # initialize at the begining
        # store the velocity of each params, encode as a dict
    
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
        for key in params.keys():
            self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
            params[key] += self.v[key]
~~~

<br>

## 3. 自适应梯度法 `AdaGrad`

自适应梯度法会在学习过程中为每一个参数适当调节其学习率. 其核心原理如下:

$$h \leftarrow h + \frac{\partial{L}}{\partial{W}} \otimes \frac{\partial{L}}{\partial{W}}$$

$$W \leftarrow W - \eta \frac{1}{\sqrt{h}}\frac{\partial{L}}{\partial{W}}$$

此处的 $\otimes$ 是 **矩阵乘法**, 变量 $h$ 保存了对参数 $W$ 而言的, 此前学习中所有梯度的平方和. 通过在每一次更新时对学习率乘以不断减小的权值 $\frac{1}{\sqrt{h}}$, 可使学习的尺度随着学习周期的增加而减小, 从而实现学习率的衰减. 

其 `Python` 实现如下:
~~~python
class AdaGrad:
    def __init__(self, lr=0.01):
        self.lr = lr
        self.h = None

    def update(self, params, grads):
        if self.h is None:
            self.h = {}
            for key, val in params.items():
                self.h[key] = np.zeros_like(val)
        for key in params.keys():
            self.h[key] += grads[key] * grads[key]
            params[key] -= self.lr * grads[key] / (np.sqrt(self.h[key]) + 1e-7)

            # plus a small num incase overflow
~~~

<br>

## 4. `Adam` 算法

`Adam` 方法基本可以视为前两种方法的结合. 

<br>

## 5. 批标准化 `Batch Normalization`


<br>

## 6. 抑制过拟合: 正则化


<Br>

## 7. 超参数的最优化