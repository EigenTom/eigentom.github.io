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


<iframe frameborder="0" style="width:100%;height:273px;" src="https://viewer.diagrams.net/?highlight=0000FF&edit=_blank&layers=1&nav=1&title=perceptron.png#R7Vtbk6I4FP41PHYXSQD1Udvu3X3YrdnyYWafthCiZhYIg6HF%2BfWbEK4hdKvtbbTtKpuckBPO933kckADPYXZb4kbr%2F6kPg4MaPqZgaYGhNbQ4d%2FCsJUGhKA0LBPiSxOoDTPyExdGs7CmxMfr1omM0oCRuG30aBRhj7VsbpLQTfu0BQ3avcbuEncMM88NutavxGcraR3CQW3%2FHZPlquwZOCNZE7rlyUUk65Xr003DhJ4N9JRQyuRRmD3hQGBX4iLbvfTUVheW4Ijt0uAvbxj8%2FZrO8b%2Fm1Pk%2BQHS8%2BfEAi6t9dYO0iLi4WrYtIUhoGvlYeDENNNmsCMOz2PVE7YZzzm0rFga8BPjhggTBEw1okrdFC1v8cfuaJfQ%2F3Khx8o9oQSPWsMsPtxcXhhOGs96QQQUkFyCmIWbJlp9SNEAF9IX2QEnFpmbStgrbqsEiHBRGt1DPsnJdA8wPCoz3wBto4HYC3u0kboHu%2FEiFLiahmyxJZKAxrzXjjH%2FnyJgCtAeOKWbeStZGNAndoDohIBF%2BKGPS1OcOFm5Igq2s5326YZxXImTljJFsiSOciFZKXX2B%2FGhZ%2FM%2FDEH61kcgrzu9v0Z2VR9Px8mIao6kxGhuTcemR4yydtjvi5ri2KZLlTPABAr8vV3cdy1FjQTIh8a5OTdN5Hr90dbrIP13Fg7kLMDyOfqHTFnAl6IaAhxr9Dk8lX9QrX1JysSkt63Re2swGaQ1zw0p6qaxHH7D36DMHvr8wdawCc4BG%2BDgsAYWlqtxgSTfKWKdiyTqMJXBXLEHNZHBWluzDWIL3xdLwwiw5h7EU3RVLyLowS4MOS%2BBW1wTWlU02ww702T5zy12QdPG5prvL05HUO7XcB0mXnmrKlMc7LPVOLXfB0sWnGvC5hd9pCw%2BN4eAyW%2FiLLIks88q26gBqdKpiHPljkSPlJS9w12vitaHlsSfbb4ZM%2FOWFf0Th0S6L06xZOd0WJdkR9jupVQVNaDJ%2Bc2D2RhhIj3ozg6dBtbQlOHAZeW1fhg7qoocvlOTaz9rklGstlaw1TRMPF42aqVfFD1ATkUhxJGHoOMqJr6L%2BgBZ0eZsr08KVcAzR6NE%2BkOVOtq7j6tQ86zI%2FnzzreEbmofeyuvdVHZ2aY13e6JNjLcf20e5ljatT86zLPO3Lc0bYt8Zxg2VeqkkWhW2T8ZPO%2BZKDN0J3dlwbXFRrHbE5Bw4oSFkcIHDmAaWbPLsXodm%2FhNCAog%2F14fSuQoOjnoTHuYTWTRVe3cx19WJQt5lqnmNnMdjKlmR0ZjHs8OLJjY466JcQmjq9HSo0dXqDZxZamXo%2FhtDMx9HIaonNct5WGy98wQnhQeDkHQVKV29IMC%2Bp3o6oy50HwE9hHkWYukTyoSPgowlgW5jWcG9lHlNLF9WIdSyN2GpKx3HOqxGo0YhMnlcPfr7m%2BGGfnzZLw0Zuvf%2FREMMZU5LoAVlGQmWcaaGHiUiGE88NxkVFSHxfNJ8keE1%2BuvPclRBNLGLP0bAnhj0VvlJG5VOC3HURwYfeg1MWK9UqqCEnYGr0BM1%2B6Xwouw7734SrYJ8xLJ4IvaSRxwiNbpIZpO4pLs9M%2F9tvFex%2FRHHKkblJQuwWH0DzGEo38J6Ojv7X3JQR7Db5AOo%2BS0OIcxxCeLH%2BNYGcgeqfZKDn%2FwE%3D"></iframe>

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

<div class="mxgraph" style="max-width:100%;border:1px solid transparent;" data-mxgraph="{&quot;highlight&quot;:&quot;#0000FF&quot;,&quot;nav&quot;:true,&quot;resize&quot;:true,&quot;toolbar&quot;:&quot;zoom lightbox&quot;,&quot;edit&quot;:&quot;_blank&quot;,&quot;xml&quot;:&quot;&lt;mxfile host=\&quot;app.diagrams.net\&quot; modified=\&quot;2021-01-10T06:48:39.900Z\&quot; agent=\&quot;5.0 (Macintosh; Intel Mac OS X 10_15_6) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15\&quot; etag=\&quot;hOoH4pZXKsEQ_Rn00lTR\&quot; version=\&quot;14.1.9\&quot; type=\&quot;device\&quot;&gt;&lt;diagram id=\&quot;02z9DBB4H2bCFNeVZXgb\&quot; name=\&quot;Page-1\&quot;&gt;7Vptk5owEP41frwbIPL28U7PtjN9uc596PVTJ0LUtEhsiKfer28iASHgGK3AaevN3JANWZJnH3Y3S3pgMF+/o3Ax+0RCFPUsI1z3wLBnWX3P4f+FYJMKALBSwZTiMBWZO8ETfkVSaEjpEocoKd3ICIkYXpSFAYljFLCSDFJKVuXbJiQqP3UBp6gieApgVJV+wyGbpVLPcnfy9whPZ9mTTcdPe+Ywu1muJJnBkKwKIvDQAwNKCEuv5usBigR2GS7puNGe3nxiFMVMZ8DnwIu+vizH6IcxdH66gNytft/YdqrmBUZLuWI5W7bJIKBkGYdIaDF64H41www9LWAgelfc5lw2Y/OIt0x+OcFRNCARoduxYGKLPy5PGCW/UKHH2f7ECBKzgjz9cbmcGKIMrfcu2cyB5AREZI4Y3fBb5ABPQi+515fN1c6Q+S2zghFzk0FJnmmueYcvv5AQHwE3MA/DzdVwcqPDUMNkkTJ+gtfCPCr2IUTeJKjFPvDQeHIejE2rDHLeLqDcrwG53xjG1vVj7HWNMWgR47EZhhOjDmPTcIGPzoOxZSgY211j3L96jPMo3xnGGuHvwjH2u4bYaRFiw3BHo9GZgPTeWmBzrwXJzsOXr4FkHN6J/QNvBRFMEhyUAeRrp5tnmRRvG99F49bOmsN1sXO4yVprzJ4zHfy6MIq3doNEIxuTTg6Fla2KYgG+ALKkAdLweAzSKWIH39uqTQs2s2tslskoiiDDL+UJ1xlSPuGRYL6UHWXUjMdUuJAuVI4q7npURW5ZkaPoSXGo6NnSKl/16UzLKHwept3yt7LMNqcBtm01PCKK+eIR7ZKCbqcUtPf4/6Mp6O/JLNvioM6m8xAHO/ZaoFMmOIoBwalMUAOhGuGaZoLO1rgBJpwaK8/JoP5lxD3V6XiX6nR0KgRXSjVwGfFN9Wr+iVQDauWk7RxLp1DSJNVueGLm+kduA5pOsrT9XbckVNLzvHpxNAlNhYRuyyTUqSRdqb/LvssdpJrdKdWU5Ms6dUsJ9lXj26KaTkXtSqlm6VKt3ynVlOTLUpMvbar1D7jHpqmmU3K8Uqppe7VO96amGkBPLpQpAbRSfW2aat5/qr1tr6ZSrbKlPJVqrQfQDsv/+rR5647HOVPmXin1q/vQhtmQ4fEvOh7tdKrTzN1Ua1nq3k6baqDbdMquVuI/xIulmOFHuEG0wjuG1qzMtPLZgZjESPl2K0UwwtNY0JWTQtQX7sUXXBzA6E52zHEYisfUfiguH/mTk/6bT8C+XQK+5lCCX8Ofxs7h2VaNIThOcxRiyND120M9JaJ52Kk5g1TLxV+W7LhXQx9iihL8CsdbVQLQhXjht0uy73v2UOhaMpKkh5HNMyFut/UK8ObuVHHqunZHs8HDHw==&lt;/diagram&gt;&lt;/mxfile&gt;&quot;}"></div>
<script type="text/javascript" src="https://viewer.diagrams.net/js/viewer-static.min.js"></script>

<br>


## 4. 激活函数的定义和常见的激活函数


<br>

## 5. 三层神经网络的实现


<br>

## 6. 神经网络的输出

