---
layout:     post
title:      逻辑学探幽 Part1
subtitle:   没有逻辑, 只有heip
date:       2020-11-16
author:     R1NG
header-img: img/post-bg-ios9-web.jpg
description: 本章进一步探讨对逻辑表达式的简化方法, 引入逻辑表达式等价定义和正规化的概念. 
catalog: true
tags:
    - 逻辑学
    - 2020
---

# 逻辑表达式的等价定义和正规化

为了进一步地简化逻辑表达式, 我们需要研究它的等价定义和正规化. 

<br>

## 1. 逻辑表达式的等价

如日常生活中一样, 我们可以用不同的逻辑表达式对同一个命题进行等价描述. 下面, 我们对逻辑表达式的等价进行简要的研究.

### 1.1 语义等价 (`Semantic Equivalence`)

**定义1.1 语义学等价**
>当且仅当对任何赋值 $v$, 谓词公式 $A, B$ 的解释 (`intepretation`) 一致, 称 **$A, B$ 语义学等价**, 记为 $A \equiv B$. 

注: 
1. 特别地, 在布尔语义下, 若两个谓词公式共享同一个真值表, 即可称其语义学等价. 
2. 若 $A \equiv B$, 我们也称 $A$ 和 $B$ 逻辑意义上相等. 
3. 在布尔语义下使用真值表法判断两个谓词公式语义学等价时, 真值表內必须包含组成这两个谓词公式的所有子式!
4. 通过使用真值表法, 不难得知: $\neg A \vee B \equiv A \rightarrow B.$ 

更进一步地, 通过使用同样的方法, 我们可以证明更为复杂的谓词公式在布尔语义下的等价性. 在构造真值表时, 下面的替代定理有助于我们的判断:

<br>


### 1.2 命题变量的替代


**定理1.1 基本语义等价定理**


|公式|性质|
|:-|:-:|
|$\neg \neg P \equiv P$|双否定性|
|$P \wedge P \equiv P$|幂等性|
|$P \vee P \equiv P$|幂等性|
|$P \wedge Q \equiv Q \wedge P$|交换律|
|$P \vee Q \equiv Q \vee P$|交换律|
|$P \wedge (Q \wedge R) \equiv (P \wedge Q) \wedge R$|结合律|
|$P \vee (Q \vee R) \equiv (P \vee Q) \vee R$|结合律|
|$P \wedge (Q \vee R) \equiv (P \wedge Q)\vee(P \wedge R)$|分配律|
$P \vee (Q \wedge R) \equiv (P \vee Q)\wedge(P \vee R)$|分配律|
|$P \wedge (P \vee Q) \equiv P$|吸收性|
|$P \vee (P \wedge Q) \equiv P$|吸收性|
|$\neg(P \wedge Q) \equiv \neg P \vee \neg Q$|`De Morgan`|
|$\neg(P \vee Q) \equiv \neg P \wedge \neg Q$|`De Morgan`|
|$P \rightarrow Q \equiv \neg P \vee Q$|逆否性|
|$P \rightarrow Q \equiv \neg(P \wedge \neg Q)$|逆否性|
|$P \rightarrow Q \equiv \neg Q \rightarrow \neg P$|逆否性|
|$P \leftrightarrow Q \equiv (P \rightarrow Q) \wedge (Q \rightarrow P)$|逆否性|
|$P \leftrightarrow Q \equiv (\neg P \vee Q) \wedge (P \vee \neg Q)$|逆否性|
|$P \leftrightarrow Q \equiv (P \wedge Q) \vee (\neg P \wedge \neg Q)$|逆否性|


<br>


注意: $A \equiv B$ 指语义学等价, 这是谓词公式的一个性质, 也是一个超逻辑概念, $“\equiv”$ 不是逻辑联结词; 而 $\leftrightarrow$ 属于谓词逻辑语言中的符号, 二者在定义上有着本质的区别, 切勿混淆. 下面的定理揭示了它们之间的关系:

**定理1.2**
>$A \equiv B$ 当且仅当 $A \leftrightarrow B$ 为重言式. 

<br>

语义学替换 (`Substitution`) 是一种规范化的, 作用于整个谓词公式的谓词变量替换方法:

若我们要使用语义学替换, 将某个谓词公式中的谓词变量 $A$ 替换为与其在语义学上等价的谓词变量 $B$, 则我们要将 **在谓词公式中所有出现的 $A$ 全部替换为 $B$.**

更一般地,我们还可以将谓词变量替换为具有相同真值的谓词公式: 

<br>

**定义1.2 语义学替换定理**
>设 $A, B$ 为任两个谓词公式. 若
><center>
>
>$A(P_1, P_2, \cdots, P_n) \equiv B(P_1, P_2, \cdots, P_n)$
></center>
>
>且 $C_1, C_2, \cdots, C_n$ 为 **谓词公式**, 则有
><center>
>
>$A(C_1, C_2, \cdots, C_n) \equiv B(C_1, C_2, \cdots, C_n).$
></center>


<br>

**定理1.3 一般化的基本语义等价定理**



|公式|性质|
|:-|:-:|
|$\neg \neg A \equiv A$|双否定性|
|$A \wedge A \equiv A$|幂等性|
|$A \vee A \equiv A$|幂等性|
|$A \wedge B \equiv B \wedge A$|交换律|
|$A \vee B \equiv B \vee A$|交换律|
|$A \wedge (B \wedge R) \equiv (A \wedge B) \wedge R$|结合律|
|$A \vee (B \vee R) \equiv (A \vee B) \vee R$|结合律|
|$A \wedge (B \vee R) \equiv (A \wedge B)\vee(A \wedge R)$|分配律|
$A \vee (B \wedge R) \equiv (A \vee B)\wedge(A \vee R)$|分配律|
|$A \wedge (A \vee B) \equiv A$|吸收性|
|$A \vee (A \wedge B) \equiv A$|吸收性|
|$\neg(A \wedge B) \equiv \neg A \vee \neg B$|`De Morgan`|
|$\neg(A \vee B) \equiv \neg A \wedge \neg B$|`De Morgan`|
|$A \rightarrow B \equiv \neg A \vee B$|逆否性|
|$A \rightarrow B \equiv \neg(A \wedge \neg B)$|逆否性|
|$A \rightarrow B \equiv \neg B \rightarrow \neg A$|逆否性|
|$A \leftrightarrow B \equiv (A \rightarrow B) \wedge (B \rightarrow A)$|逆否性|
|$A \leftrightarrow B \equiv (\neg A \vee B) \wedge (A \vee \neg B)$|逆否性|
|$A \leftrightarrow B \equiv (A \wedge B) \vee (\neg A \wedge \neg B)$|逆否性|


<br>


### 1.3 命题变量的等价代换

平行于语义学替换定理, 我们还定义了命题变量的等价代换. 和替换定理不同的是, 等价代换定理的作用域非常有限, 仅仅涉及某个特定的命题变量, 只代换那一个特定的命题变量, 而对即使完全相同的其他命题变量也不予考虑. 

在下面的叙述中, 我们规定: 
<center>

$C(\cdots A \cdots)$

</center>

指一个谓词公式, 而 $A$ 为它的一个子式. 

<br>

**定理1.4 等价代换定理**
>设 $A, B$ 为两个谓词公式, 且 $A \equiv B$, 且 $A$ 为谓词公式 $C(\cdots A \cdots)$ 的一个子式. 则
><center>
>
>$C(\cdots A \cdots) \equiv C(\cdots B \cdots)$
>
></center>
>
>当对 $A$ 实行等价代换. 

<br>

### 1.4 逻辑常量 $\perp$, $\top$

我们还可以通过定义 **逻辑常量** 的方式进一步简化逻辑表达式. 

在第一章中, 我们介绍了重言式和矛盾式. 它们在任何语义系统下的解释都是永真, 或永假. 对于这些式子, 我们使用 $\perp$ 和 $\top$ 符号分别表示它们. 

我们可以认为, 设 $P$ 为一个谓词公式, 则 $\top$ 和 $\perp$ 可分别视为 $P \vee \neg P, P\wedge \neg P$ 的缩写. 从真值表中我们可以立即得出这一结论:

|$P$|$\neg P$|$P \vee \neg P$|$P \wedge \neg P$|
|:-:|:-:|:-:|:-:|
|$1$|$0$|$1$|$0$|
|$0$|$1$|$0$|$1$|

**定理1.5 逻辑常量定理**
>若 $A$ 为重言式, 则 $A\equiv \top$.
>
>若 $A$ 为矛盾式, 则 $A\equiv \perp$.


**定理1.6 含逻辑常量的一般化基本语义等价定理**


![20201121093615](https://cdn.jsdelivr.net/gh/KirisameMarisaa/KirisameMarisaa.github.io/img/blogpost_images/20201121093615.png)





### 1.5 其他语义系统中的语义学等价

我们以幂集语义为例. 显然, 在幂集语义下, 一个谓词公式的解释与对谓词变量的赋值和选定的参考集 $X$ 同时有关. 在幂集语义下, 要说明 $A\equiv B$, 就要证明在给定的谓词变量赋值方式和参考集下, $A$ 和 $B$ 的翻译都是 $X$ 的同一个子集. 


<br>

<br>


## 2. 正规形式和逻辑表达式的正规化

逻辑表达式有多种等价的表达形式, 而不同的表达形式种由于含有不同数量的联结词, 括号关系和子式, 其复杂程度也有所不同. 在构造逻辑电路和研究抽象的逻辑关系时, 我们需要对逻辑表达式进行简化, 从而降低工作的规模和难度. 对逻辑表达式在逻辑层面上进行简化的过程就是寻找和计算范式 (`Normal Form`) 的过程.  

一种最简单的逻辑表达式简化方法是基于结合律, 除去表达式內多余的括号:
<center>

$P_1\wedge (P_2 \wedge (\cdots \wedge P_n)\cdots) \Rightarrow P_1\wedge P_2 \wedge \cdots \wedge P_n.$

</center>

下面, 我们介绍两种更为复杂的范式生成法: 合取范式生成和析取范式生成. 

<br>

### 2.1 合取范式 (`Conjunctive Normal Form`)
**定义2.1 合取范式**
>称一个谓词公式为 **合取范式**, 当且仅当它为 $\top, \perp$, 或为多个子式的析取的合取. 

一般地, 我们使用下列的合取范式求解算法简化并求取谓词公式的合取范式:

![20201121100450](https://cdn.jsdelivr.net/gh/KirisameMarisaa/KirisameMarisaa.github.io/img/blogpost_images/20201121100450.png)

<br>

**定理2.1 合取范式求解算法有效性定理**
>- 合取范式求解算法的每一步都保持语义学等价关系. 
>- 合取范式求解算法总会在有限步內终止. 
>- 合取范式求解算法的返回值恒为一个合取范式. 

<br>

### 2.2 析取范式 (`Disjunctive Normal Form`)

相应地, 我们还可以定义析取范式的概念: 


**定义2.2 析取范式**
>称一个谓词公式为 **析取范式**, 当且仅当它为 $\top, \perp$, 或为多个子式的合取的析取. 

一般地, 我们使用下列的析取范式求解算法简化并求取谓词公式的析取范式:

![20201121100819](https://cdn.jsdelivr.net/gh/KirisameMarisaa/KirisameMarisaa.github.io/img/blogpost_images/20201121100819.png)

<br>

**定理2.2 析取范式求解算法有效性定理**
>- 析取范式求解算法的每一步都保持语义学等价关系. 
>- 析取范式求解算法总会在有限步內终止. 
>- 析取范式求解算法的返回值恒为一个析取范式. 



### 2.3 基于真值表构造范式

基于合取范式和析取范式的概念和相关定理, 我们立即可以得出从真值表构造
范式的方法:

![20201121101009](https://cdn.jsdelivr.net/gh/KirisameMarisaa/KirisameMarisaa.github.io/img/blogpost_images/20201121101009.png)


<br>

### 2.4 布尔函数

回顾布尔语义的概念和基本定义. 我们知道, 布尔语义的真值集为 $\mathbb{B} = \{0, 1\}$. <br>下面, 考虑一个从 $\mathbb{B} $ 的 $n$ 次笛卡尔积映到 $\mathbb{B}$ 的一个函数 (映射). 何时这样的映射既可被视在布尔语义下对联结词的解释, 亦可视为对谓词公式的解释?

**定义2.3 布尔函数 (`Boolean Function`)**
>称从 $\mathbb{B}^{n}$ 到 $\mathbb{B}$ 的映射为 **布尔函数**.

在布尔语义中, 任何一个谓词逻辑的联结词均可被视为一个布尔函数. 

<br>

**定理2.3 布尔函数计数定理**
>存在 $2^{2^n}$ 个不同的 $n$ 元布尔函数. 


显然, 对于 $n$ 个不同的函数输入, 根据布尔语义定义, 每个输入都可以从真值集中取值, 故共有 $2^n$ 种输入; 同时, 对于每一种输入, 它都可以被映射到真值集的某一个元素, 故共有 $2^{2^{n}}$ 种映射方法, 而每一种映射方法对应一个唯一的布尔函数. 

<br>

**定理2.4 布尔函数表示定理**
>任一个布尔函数: $f: \mathbb{B^{n}}\rightarrow \mathbb{B}$ 可被表示为含谓词变量 $x_1, x_2, \cdots, x_n$ 的谓词公式 $P(x_1, x_2, \cdots, x_n)$.

结合真值表方法可知, 该定理结论显然. 

