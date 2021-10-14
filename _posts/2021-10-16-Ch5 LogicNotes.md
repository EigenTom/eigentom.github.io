---
layout:     post
title:      逻辑学探幽 Part5
subtitle:   没有逻辑 只有heip
date:       2021-10-16
author:     R1NG
header-img: img/post-bg-logicnotes.jpg
description: 本章介绍一种检验可满足性的算法: `DPLL`.
catalog: true
tags:
    - 逻辑学
    - 2021
---

# `DPLL`

本章讨论的主要对象是最常见的可满足性检验方法: `DPLL`,其名称来源自它的发明者. 该方法所检验的对象是 **合取范式** 或 **由一系列子句构成的集合**. 

我们首先引入 **子句** 与 **合取范式** 的概念, 并在第二节中回顾对于谓词公式的合取范式求解算法. 我们会在第三章中讨论合取范式求解算法的时间复杂度问题, 并给出一种基于定义上的将谓词公式转换为由一系列子句构成的集合的方法. 

我们会在第四节中介绍一个关于子句的可满足性的特殊问题: $k-\text{SAT}$. 在第五节中, 我们对 `DPLL` 所使用的核心规则: `单元子句传播` (`Unit Propagation`). 在剩下的几节中, 我们会描述 `DPLL` 算法, 并最后简单讨论算法的优化问题. 

<br>

## 5.1 子句 (`Clauses`)

为了规避分裂法 (`Splitting Method`) 中由于对任意谓词公式的可满足性检查不能始终保证高效率的问题, 大多数的自动推理 (`Automated Reasoning`) 算法都使用 **子句** 作为可满足性检查的最小单位, 无论自动推理的作用域为谓词逻辑还是一节谓词逻辑都是如此. 

这种选择的合理性有以下两点:

首先 (在后面我们对子句与合取范式定义的介绍中我们可以立即看出), 所有的针对任意谓词公式的可满足性检查问题都可以被转换为一个等价的, 针对一系列子句的可满足性的检查问题. 

其次, 相比任意结构的谓词公式, 子句具有显著的结构简易的特点.

粗略地说, 在一个基于 **子句** 的可满足性检测问题中, 作为输入的谓词公式或者一系列谓词公式将首先被转换为一系列子句, 然后某个用于检测子句可满足性的算法会作用在这个由子句组成的集合上.

下面, 我们为 **子句** 给出精确的定义. 在引入子句的定义之前, 我们首先需要引入另一种特殊的谓词公式, **文字** (`literal`).

**定义 5.1.1** (文字)
> **文字** 或为某个原子公式 $A$ 本身或它的否定形式 (`negation`) $\neg A$.
> 
> 若一个 **文字** 形为某个原子公式, 称其为 **正的** (`Positive`), 反之称其为 **负的** (`Negative).

我们随后定义 **文字的补**:

<br>

**定义 5.1.2** (文字的补)
> 给定文字 $L$, 它的 **补** $\widetilde{L}$ 定义为:
>
>$$\widetilde{L} \overset{\text{def}}{=} \begin{cases} \neg L ~~~ \text{if L is positive} \\ ~~~L ~~~
 \text{if L has the form} ~ \neg A\end{cases} $$

不难得出, 文字 $p$ 和 $\neg p$ 互为对方的 $补$. 并且, 显然地:

$$\widetilde{\widetilde{L}} = L.$$

本质上, 文字就是原子公式的别名. 在完成对 **文字** 和 **文字的补** 的定义后, 我们下面引入 **子句** 的严格定义:

<br>

**定义 5.1.3** (子句)
> 子句 `clause` 本质上就是一系列文字 $L_1, L_2, \cdots, L_n, ~ n\geqslant0$ 的析取 $L_1 \vee L_2 \vee \cdots \vee L_n.$

下面再给出一些特殊形式子句的额外定义. 

1. 若 $n=0$, 称这样的子句为 **空** (`empty`), 用 $\square$ 表示. 

2. 若 $n=1$, 此时子句恰由一个文字组成, 称这样的子句为 **单位** (`unit`). 

3. 若组成子句的文字中 **最多有一个为真**, 称这样的子句为 **`Horn` 子句**. 

4. 类似地, 若组成子句的文字全为真, 称该子句也为 **正的**, 反之若全为假, 则称该子句为 **负的**.

结合上述定义, 我们同时可以得到以下结论:

1. 显然, 任何单位子句都是 `Horn` 子句.
2. 一个 **正的** 子句为 `Horn` 子句, 当且仅当它或为单位子句, 或为空.
3. 任何 **负的** 子句都是 `Horn` 子句.

<br>

我们下面考虑基于 **文字** 和 **子句** 框架下的可满足性. 

首先, 对于 **正的** 文字 $p$, 任一解释 $I$ 满足它, 当且仅当 $I(p)=1$; 对于 **负的** 文字 $\neg p$, 满足它的解释 $I$ 需要满足 $I(p)=0$, 也就是 $I(\neg p) = 1$. 

其次, 我们考虑子句. 由于子句本质上是一系列文字的析取, 因此若某个解释 $I$ 满足某个子句 $C$, 当且仅当该解释满足组成该子句的至少一个文字. 基于这个事实, 我们认为空子句实际上和 $\perp$ 等价, 因为任何解释 $I$ 都永无法满足空子句. 

我们同时可以看到, 与空子句的永不可满足性相对, 任何非空的子句都是可满足的. 此外, 如果某个子句中同时包含相对的一对文字 $p$ 和 $\neg p$, 这个子句实际上就是一个重言式 (`tautology`), 因为无论 $p$ 取和值, 它恒为真. 

相应地, 若某个子句中不包含任何一对相对的文字, 我们总可以构造出一个让它为假的解释 $I$:

$$I(p) = \begin{cases} 0 ~~~ \text{if literal p is positive} \\ 1 ~~~ \text{if literal p is negative}\end{cases}.$$

从上面的讨论可以看出, 对单独某个子句的可满足性的检测是非常简单的, 但对于一系列子句而言则并非如此. 我们下面引入 **合取范式** 和 **析取范式** 的概念, 并借此说明任何形式的谓词公式的可满足性问题都可以被转化为某些子句的可满足性问题.

**定义 5.1.4** (合取范式)
>称一个谓词公式为 **合取范式**, 当且仅当它为 $\top, \perp$, 或为多个文字析取 (也就是子句) 的合取:
>
> $$A = \bigwedge_{i} \bigvee_{j} L_{i,j}$$

**定义 5.1.5** (析取范式)
>称一个谓词公式为 **析取范式**, 当且仅当它为 $\top, \perp$, 或为多个文字合取的析取:
>
> $$A = \bigvee_{i} \bigwedge_{j} L_{i,j}$$

我们称谓词公式 $B$ 为谓词公式 $A$ 的 **合取 (析取) 范式**, 当且仅当两个谓词公式 **等价** 且公式 $B$ 为 **合取 (析取) 范式**.

显然根据上述定义, 针对合取范式的可满足性问题实质上和针对一系列子句的可满足性问题是等价的. 

<br>

## 5.2 向合取范式的转换

下面给出一个最基础的合取范式转换算法: 

**算法 5.2.1** (基础合取范式转换算法)
> 该算法基于下图展示的 **改写律**. 在不考虑同级符号下谓词文字的交换问题和结合问题 (也就是说, 我们认为它满足交换律和结合律) 的情况下, 通过多次应用改写律, 我们最终可以得到所输入算法的合取范式.

![20211022221637](https://cdn.jsdelivr.net/gh/KirisameR/KirisameR.github.io/img/blogpost_images/20211022221637.png)

![20201121100819](https://cdn.jsdelivr.net/gh/KirisameMarisaa/KirisameMarisaa.github.io/img/blogpost_images/20201121100819.png)

<br>

## 5.3 子句形式和定义变换

首先来用一个例子侧面说明引入 **子句形式** 的必要性. 

考虑谓词公式

$$p_1 \leftrightarrow (p_2 \leftrightarrow (p_3 \leftrightarrow (p_4 \leftrightarrow (p_5 \leftrightarrow p_6))))$$

不难看到, 在我们要采用基础合取范式转换算法尝试求得它的合取范式的过程中, 在改写律的规定下, 每一次对外层 $\leftrightarrow$ 的改写都会在原来包含子式 $p_5 \leftrightarrow p_6$ 的谓词公式基础上引入两个新的 $p_5 \leftrightarrow p_6$. 因此, 不难看出对于给定的谓词公式而言, 随着逐步应用改写律, 我们得到的新谓词公式中子式数量的增长速度将会是指数级别, 进一步处理如此复杂度的子式将会令其更加复杂, 这是在实际应用中绝不可接受的.

为了规避这一问题, 我们不再要求算法在对给定谓词公式进行转换后给出一个严格与之等价的新公式, 而将要求弱化到只需要给出一个在可满足性上与原公式相同的结果即可. 为了实现这一点, 我们下面对 **定义 (命名)** 和 **子句形式** 给出定义. 

**定义 5.3.1** (命名)
> 给定谓词公式 $P$, 称引入谓词变量 $n$, 规定 $n \leftrightarrow P$, 则原先包含 $P$ 的公式 $P \leftrightarrow Q$ 可被转换为:
>
>$$n \leftrightarrow Q \\ n \leftrightarrow P$$
>
>我们称这个过程为 **用 $n$ 对 $P$ 的命名**, 并且称公式 $n \leftrightarrow P$ 为对变量 $n$ 的 **定义**.

命名的对象是某个已经存在的谓词公式, 定义的对象是某个尚不明确的谓词变量或公式. 命名的本质是用一个被定义为与原来的较为复杂的子式等价的谓词变量, 通过在原公式中对复杂子式的替换, 从而实现简化子式的效果. 在大部分情况下, 只要我们注意选择被 **命名** 的子式, 原公式的复杂度都可以得到相当程度的简化. 

需要注意的是, 对给定谓词公式的某个或某些子式命名后得到的新公式和原来的并不等价:

考虑这样的例子. 不妨假设 $S$ 为一个只包含了一个谓词公式 $p$ 的集合, 可知 $S$ 的模型 (`model`) 即为所有使 $p$ 为真的解释 $I_S$. 若此时添加一个对 $p$ 的定义, 则有: 

$$S' = \{p, n \leftrightarrow p\}$$

显然可知, $S'$ 的 模型为所有使得 $p$ 和 $q$ 为真的解释 $I_{S'}$ (因为我们需要让 $p$ 为真的同时 $p \leftrightarrow q$ 也为真, 那就只能确保 $p$, $q$ 同真), 自然 $I_{S} \neq I_{S'}$, 也就得到了 $S$ 和 $S'$ 不等价的结论. 

为了进一步说明 **命名 (定义)** 的概念, 并便于我们将谓词公式转换为一系列子句, 我们下面引入 **子句形式** 的概念. 

**定义 5.3.2** (子句形式)
> 记 $A$ 为某个谓词公式. $A$ 的 **子句形式** 即为和 $A$ 共享可满足性的一系列子句.
> 
>  也就是说, $A$ 为可满足的, 当且仅当 $A$ 的子句形式也是可满足的.

子句形式和合取范式最本质的区别在于它们和原公式之间的等价性不同. 子句形式只在可满足性上和原公式相同, 而合取范式是和原公式完全等价的. 这说明对任一谓词公式或一系列谓词公式, 我们总能找到一个足够短的子句形式. 通过定义变换所得到的新的谓词公式的长度增长速度将会是随着原谓词公式规模的增长的多项式级别而非基础合取范式转换算法的指数级别.

下面我们通过一个引理详细解释 **命名 (定义)** 的作用.

**引理 5.3.1**
> 设 $S$ 为一系列谓词公式, $B$ 为 $S$ 中的一个谓词公式, 记 $n$ 为一个不在 $B$ 和 $S$ 中出现过的谓词变量. 
> 
> 则 $S$ 为可满足的, 当且仅当 $S \cup \{n \leftrightarrow B\}$ 为可满足的. **注意我们一般记 $S \cup \{n \leftrightarrow B\}$ 为 $S'$**.

**证明**
1. 必要性: 显然可知, $S \cup \{n \leftrightarrow B\}$ 的任一模型都是 $S$ 的一个模型, 故必要性成立. $\blacksquare$

2. 充分性: 不妨假设 $S$ 在解释 $I$ 下为可满足的. 我们定义新的解释 $I'$ 如下:
   $$I'(q) \overset{\text{def}}{=} \begin{cases} I(B) ~~~ \text{if} ~q=n \\ I(q) ~~~~ \text{else}\end{cases}$$

   显然, $I'$ 同时满足 $S$ 和在对 $B$ 命名的过程中新引入的公式 $n \leftrightarrow B$. 同时由于 $I'$ 对 $n$ 的解释完全不影响 $S$, 故 $I'$ 同时也是 $S$ 的一个模型, 且有 $I(B) = I'(B)$. 

   由 $I'$的定义我们又知, $I'(n) = I(B) \rightarrow I'(n) = I'(B)$. 因此, $I‘$ 也满足 $n \leftrightarrow B$.
   由此可知, $I'$ 也是 $S \cup \{n \leftrightarrow B\}$ 的一个模型.

   因此, 无论对于 $S$ 的任意一个模型 $I$, 我们总能基于它构造出一个 $S \cup \{n \leftrightarrow B\}$ 的模型, 记为 $I’$, 故充分性成立. $\blacksquare$

综上, “$S$ 为可满足的” 和 “$S \cup \{n \leftrightarrow B\}$ 为可满足的” 互为充要条件. $\blacksquare$

由此可见, **命名 (定义)** 可以用于对某个谓词公式中的复杂且重复出现多次的子项进行命名并将其等价替换, 从而得到一个更加简化的式子. 我们下面介绍一个应用了这一技巧的算法:

**算法 5.3.1** (定义变换)
> 我们可以使用如下的算法将任意谓词公式 $A$ 替换为一个由一系列子句组成的集合 $S$, 其中 $S$ 为 $A$ 的子句形式:
> 1. 若 $A$ 本身已经是一系列子句 $C_1, C_2, \cdots, C_n$ 的合取: 则 
> 
> $$S \overset{\text{def}}{=} \{C_1, C_2, \cdots, C_n\}.$$
>
>2. 若不然, 对 $A$ 中的每一个不是文字的子式 $B$, 我们定义一个执行转换的函数 $n(B)$:
>
>       $$n(B) \overset{\text{def}}{=} \begin{cases} B ~~~ \text{if} ~B~ \text{is a literal} \\ p_B ~~~ \text{else}\end{cases}$$
> 
>       并且记 $\widetilde{n}(B)$ 为 $n(B)$ 的否定. 我们基于如下的规则定义 $p_B \leftrightarrow B$:<br>
>
>       `i`. 若 $B$ 形为 $B_1 \wedge B_2 \wedge \cdots \wedge B_n$, 则将对变量 $p_B$ 的定义: 
>
>       $$p_B \leftrightarrow n(B_1) \wedge n(B_2) \cdots \wedge n(B_m)$$
>
>       加入 $S$ 中, 也就是:
>
>       $$\begin{aligned} &\neg p_B \vee n(B_1), \\ &\cdots \\ &\neg p_B \vee n(B_m), \\ &\widetilde{n}(B_1) \vee \cdots \vee \widetilde{n}(B_m) \vee p_B \end{aligned}$$
>
>       `ii`. 若 $B$ 形为 $B_1 \wedge B_2 \vee \cdots \vee B_n$, 则将对变量 $p_B$ 的定义: 
>
>       $$p_B \leftrightarrow n(B_1) \vee n(B_2) \cdots \vee n(B_m)$$
>
>       加入 $S$ 中, 也就是:
>        
>       $$\begin{aligned} &p_B \vee \widetilde{n}(B_1), \\ &\cdots \\ &p_B \vee \widetilde{n}(B_m), \\ &n(B_1) \vee \cdots \vee n(B_m) \vee \neg p_B \end{aligned}$$
>
>      `iii`. 若 $B$ 形为 $B_1 \rightarrow B_2$, 则将对变量 $p_B$ 的定义: 
>
>       $$p_B \leftrightarrow (n(B_1) \rightarrow n(B_2))$$
>
>       加入 $S$ 中, 也就是:
> 
>       $$\begin{aligned} &\neg p_B \vee \widetilde{n}(B_1) \vee \widetilde{n}(B_2), \\ &n(B_1) \vee p_B,  \\ &\widetilde{n}(B_2) \vee p_B\end{aligned}$$
>
>       `iv`. 若 $B$ 形为 $\neg B_1$, 则将对变量 $p_B$ 的定义: 
>
>       $$p_B \leftrightarrow \neg n(B_1)$$
>
>       加入 $S$ 中, 也就是:
> 
>       $$\begin{aligned} &\neg p_B \vee \widetilde{n}(B_1), \\ &n(B_1) \vee p_B \end{aligned}$$
>
>      `v`. 若 $B$ 形为 $B_1 \leftrightarrow B_2$, 则将对变量 $p_B$ 的定义: 
>
>       $$p_B \leftrightarrow (n(B_1) \leftrightarrow n(B_2))$$
>
>       加入 $S$ 中, 也就是:
> 
>       $$\begin{aligned} &\neg p_B \vee \widetilde{n}(B_1) \vee \widetilde{n}(B_2), \\ &\neg p_B \vee \widetilde{n}(B_2) \vee \widetilde{n}(B_1), \\ &n(B_1) \vee n(B_2) \vee p_B,  \\ &\widetilde{n}(B_2) \vee \widetilde{n}(B_2) \vee p_B\end{aligned}$$
>
> 最终将单位子句 $p_A$ 加到 $S$ 中.

我们同时可以得到:

**引理 5.3.2**
> 设 $S$ 为一系列谓词公式, $B$ 为 $S$ 中的一个谓词公式, 记 $p$ 为一个不在 $B$ 和 $S$ 中出现过的谓词变量. 
> 
> 设 $S'$ 为 $S$ 通过用 $p$ 替换掉至少一处正的 (或负的) $B$  得到的一系列谓词公式, 则 $S$ 为可满足的, 当且仅当 $S‘ \cup \{p \rightarrow B\}$ (或$S‘ \cup \{B \rightarrow p\}$) 为可满足的.

**证明**
> 此处指考虑 $p$ 为正的情况, $p$ 若为负的话同理. 
>
> 必要性: 设 $S$ 为可接受的, 则由 `引理 5.3.1`, 显然 $S'$ 是可接受的, 进一步 $S' \cup \{p \leftrightarrow B\}$ 也是可接受的. 由于显然 $S' \cup \{p \leftrightarrow B\}$ 的任一个模型均为 $S' \cup \{p \rightarrow B\}$ 的模型, 故对于 $S‘$ 的任一个模型, 我们可以相应地一步步构造出 $S’$ 的, $S' \cup \{p \leftrightarrow B\}$ 的, 直到 $S' \cup \{p \rightarrow B\}$ 的模型, 因此 $S' \cup \{p \rightarrow B\}$ 也是可接受的. $\blacksquare$
>
>充分性: 设 $S' \cup \{p \rightarrow B\}$ 为可满足的, 要证 $S$ 也是可满足的. 取 $S' \cup \{p \rightarrow B\}$ 的任意模型 $I$. 对 $\forall A \in S$, 由定义可知, $S'$ 中必存在某个子句 $A'$, 通过将 $A'$ 中的某 (几) 处 $p$ 用 $B$ 替代即可得到 $A$. 由于在我们的假设下这些 B 都以 **正** 的形式出现, 且我们有 $I \vDash A', ~~ I \vDash p \rightarrow B$, 自然地有 $I \vDash A$, 充分性得证. $\blacksquare$

我们下面引入一个利用了上述引理进一步简化了转换过程的, **优化的定义转换算法**:

**算法 5.3.2** (优化的定义变换)
> 我们可以使用如下的算法将任意谓词公式 $A$ 替换为一个由一系列子句组成的集合 $S$, 其中 $S$ 为 $A$ 的子句形式:
> 1. 若 $A$ 本身已经是一系列子句 $C_1, C_2, \cdots, C_n$ 的合取: 则 
> 
> $$S \overset{\text{def}}{=} \{C_1, C_2, \cdots, C_n\}.$$
>
>2. 若不然, 对 $A$ 中的每一个不是文字的子式 $B$, 我们定义一个执行转换的函数 $n(B)$:
>
>       $$n(B) \overset{\text{def}}{=} \begin{cases} B ~~~ \text{if} ~B~ \text{is a literal} \\ p_B ~~~ \text{else}\end{cases}$$
> 
>       并且记 $\widetilde{n}(B)$ 为 $n(B)$ 的否定. 我们基于如下的规则定义 $p_B \leftrightarrow B$.

`i`. 若 $B$ 形为 $B_1 \wedge B_2 \wedge \cdots \wedge B_n$, 则将对变量 $p_B$ 的定义加入 $S$ 中:

|极性 $+1$|极性 $-1$|
|-|-|
|$p_B \rightarrow n(B_1) \wedge n(B_2) \cdots \wedge n(B_m)$|$n(B_1) \wedge n(B_2) \cdots \wedge n(B_m) \rightarrow p_B$|
|$\begin{aligned} &\neg p_B \vee n(B_1), \\ &\cdots \\ &\neg p_B \vee n(B_m)\end{aligned}$| $\widetilde{n}(B_1) \vee \cdots \vee \widetilde{n}(B_m) \vee p_B$|

`ii`. 若 $B$ 形为 $B_1 \wedge B_2 \vee \cdots \vee B_n$, 则将对变量 $p_B$ 的定义加入 $S$ 中:

|极性 $+1$|极性 $-1$|
|-|-|
|$p_B \rightarrow n(B_1) \vee n(B_2) \cdots \vee n(B_m)$|$n(B_1) \vee n(B_2) \cdots \vee n(B_m) \rightarrow p_B$|
|$n(B_1) \vee \cdots \vee n(B_m) \vee \neg p_B$|$\begin{aligned} &p_B \vee \widetilde{n}(B_1), \\ &\cdots \\ &p_B \vee \widetilde{n}(B_m), \\ & \end{aligned}$|

`iii`. 若 $B$ 形为 $B_1 \rightarrow B_2$, 则将对变量 $p_B$ 的定义加入 $S$ 中:

|极性 $+1$|极性 $-1$|
|-|-|
|$p_B \rightarrow (n(B_1) \rightarrow n(B_2))$|$(n(B_1) \rightarrow n(B_2)) \rightarrow p_B$|
|$\neg p_B \vee \widetilde{n}(B_1) \vee n(B_2)$|$\begin{aligned} &n(B_1) \vee p_B, \\ &\widetilde{n}(B_2) \vee p_B\end{aligned}$|

iv`. 若 $B$ 形为 $\neg B_1$, 则将对变量 $p_B$ 的定义加入 $S$ 中: 

|极性 $+1$|极性 $-1$|
|-|-|
|$p_B \rightarrow \neg n(B_1)$|$\neg n(B_1)\rightarrow  p_B$|
|$\neg p_B \vee \widetilde{n}(B_1)$|$n(B_1) \vee p_B$|

`v`. 若 $B$ 形为 $B_1 \leftrightarrow B_2$, 则将对变量 $p_B$ 的定义加入 $S$ 中: 

|极性 $+1$|极性 $-1$|
|-|-|
|$p_B \rightarrow (n(B_1) \leftrightarrow n(B_2))$|$(n(B_1) \leftrightarrow n(B_2)) \rightarrow p_B$|
|$\begin{aligned} &\neg p_B \vee \widetilde{n}(B_1) \vee n(B_2), \\ &\neg p_B \vee \widetilde{n}(B_2) \vee n(B_1) \end{aligned}$|$\begin{aligned} &n(B_1) \vee n(B_2) \vee p_B,  \\ &\widetilde{n}(B_2) \vee \widetilde{n}(B_2) \vee p_B\end{aligned}$|

最终将单位子句 $p_A$ 加到 $S$ 中.

<br>

## 5.4 `SAT` 和 `k-SAT`

通过上一节, 我们知道定义的子句形式变换 (`Definitional Clausal Form Transformation`) 可以在多项式时间内将对某个命题公式的可满足性问题转换为对一系列子句的可满足性问题. 下面我们对 “对一系列子句的可满足性问题” 本身进行简单讨论.

**定义 5.4.1** (`SAT`)
> 记 `SAT` 为这样的决策问题: 考虑一个有限的, 由子句组成的集合, 若该集合是可满足的, 则该决策问题的答案为 “是”, 反之为 “否”.

我们随后讨论的对象基本集中于能够有效求解 `SAT` 问题的算法. 下面对一些情况特殊的 `SAT` 给出定义:

**定义 5.4.2** (`k-SAT`)
> 记正整数 $k$, 称 $k-\text{clause}$ 为恰包含$k$ 个文字的子句. 称 `k-SAT` 问题为关于一系列最多含有 $k$ 个文字的子句的可满足性问题. 

注: 

1. $2-\text{SAT}$ 问题可在多项式时间内求解.
2. $k-\text{SAT}$ 在 $k \geqslant 3$ 的情况下为一个 `\text{NP}`-完全问题.

<br>

## 5.3 单位子句传播




























