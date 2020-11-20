---
layout:     post
title:      逻辑学探幽 Part1
subtitle:   没有逻辑, 只有heip
date:       2020-11-16
author:     R1NG
header-img: img/post-bg-ios9-web.jpg
catalog: true
tags:
    - 逻辑学
    - 2020
---

# 逻辑表达式的等价定义和逻辑表达式的正规化

## 1. 逻辑表达式的等价

如日常生活中一样, 我们可以用不同的逻辑表达式对同一个命题进行等价描述. 下面, 我们对逻辑表达式的等价进行简要的研究.

1.1 语义等价 (`Semantic Equivalence`)

**定义1.1 语义学等价**
>当且仅当对任何赋值 $v$, 谓词公式 $A, B$ 的解释 (`intepretation`) 一致, 称 **$A, B$ 语义学等价**, 记为 $A \equiv B$. 

注: 
1. 特别地, 在布尔语义下, 若两个谓词公式共享同一个真值表, 即可称其语义学等价. 
2. 若 $A \equiv B$, 我们也称 $A$ 和 $B$ 逻辑意义上相等. 
3. 在布尔语义下使用真值表法判断两个谓词公式语义学等价时, 真值表內必须包含组成这两个谓词公式的所有子式!
4. 通过使用真值表法, 不难得知: $\neg A \vee B \equiv A \rightarrow B.$ 

更进一步地, 通过使用同样的方法, 我们可以证明更为复杂的谓词公式在布尔语义下的等价性. 在构造真值表时, 下面的替代定理有助于我们的判断:

<br>


1.2 命题变量的替代


**定理1.1 替代定理**
<center>

|公式|性质|
|-|-|
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
</center>


<br>

**定理1.2 一般化的基本语义等价定理**


1.3 命题变量的等价代换



**定理1.3 等价代换定理**



1.4 逻辑常量 $\perp$, $\top$

**定理1.4 逻辑常量定理**

**定理1.5 含逻辑常量的一般化基本语义等价定理**


1.5 其他语义系统中的语义学等价



## 2. 正规形式和逻辑表达式的正规化

2.1 合取范式 (`Conjunctive Normal Form`)


2.2 析取范式 (`Disjunctive Normal Form`)


2.3 使用真值表构造范式


2.4 布尔函数