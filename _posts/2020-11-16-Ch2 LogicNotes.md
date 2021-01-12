---
layout:     post
title:      逻辑学探幽 Part2
subtitle:   没有逻辑 只有heip
date:       2020-11-16
author:     R1NG
header-img: img/post-bg-logicnotes.jpg
description: 本章简要介绍自然演绎推理和一阶谓词逻辑. 
catalog: true
tags:
    - 逻辑学
    - 2020
---

# 自然演绎推理和一阶谓词逻辑

在本章节中, 我们将简要介绍自然演绎推理和一阶谓词逻辑. 本文假定读者已基本掌握命题逻辑, 并熟悉包括 `Truth` 和 `Falsium` 等在内的语义符号, 并了解逻辑学的基本概念. 


## 1. 自然演绎推理
在命题逻辑中, 我们除了可以使用真值表方法判断命题推论的有效性以外, 我们还可以使用自然演绎推理法. 

演绎的本质就是将复杂的推论分解为若干个规模较小, 较简单的子推论. 若这些子推论的有效性显然, 则原复杂推论的有效性也相应地被轻易确立. 若我们使用一些特定的简单推论作为推理规则, 从一组已知为真的事实出发, 直接运用这些推理规则, 逐步推出结论, 我们就使用了所谓的 **自然演绎** 法解决了问题. 

<br>

**定义 3.1 (自然演绎推理)**
>从一组已知为真的预设事实出发, 仅使用逻辑系统所定义的数个推理规则 (定理) 进行逻辑推演的过程成为 **自然演绎推理**.

<br>

为了研究和判断难以直接使用逻辑直觉推断出结果的, 更为复杂的逻辑表达式, 我们引入了自然演绎推理系统. 

在自然演绎系统中, 我们的推理过程注意以下两点: 
- 若推理的前提为真, 则结论依旧为真. 
- 推理只依据规则, 而不依赖公理. 

通过规定不同的 `初始符号`, `形成规则` 以及 `推演规则`, 我们可以构造不同的自然演绎推理系统. 在本文中, 我们的自然演绎推理系统以 `COMP11120` 所介绍的为准. 

### 1.1 初始符号<br>
- 命题变项<br>
     $p, q, r, p_1, \cdots$

- 连接词<br>
     否定: $\neg$    
     析取: $\vee$<br>
     合取: $\wedge$<br>
     蕴含: $\rightarrow$<br>
     等值: $\leftrightarrow$

- 辅助符号<br>
     `.`<br>
     `,`<br>
     `()`

<br>

### 1.2 形成规则<br>
- 任何一个命题变项都是谓词公式. 
- 若 $A$, 为谓词公式, 则 $\neg A$ 亦为谓词公式, 反之亦然.
- 若 $A, B$ 为谓词公式, 则 $A \vee B, A\wedge B, A \rightarrow B, A\leftrightarrow B$ 均为谓词公式. 
- 只有符合上述规定的公式为谓词公式. 

(以上是自然演绎系统中的谓词公式构造条件, 它和一阶谓词逻辑中的谓词公式的构造条件是有区别的!)

<br>

### 1.3 推演规则 <br>
推演规则是我们在使用自然演绎推理系统证明或验证逻辑表达式时 **唯一应该使用** 的推理规则. 
   

- `Axiom`: 公理规则<br>
命题 $A$ 总可以从 $A$ 以及其他一组命题 $\Gamma$ 中被引出:


$$\frac{}{\Gamma, A \vdash A}$$

<br>

- `Weakening`: 假设引入规则<br>
可随时按照推演需要引入一个假设前提:


$$\frac{\Gamma \vdash B}{\Gamma, A \vdash B}$$


<br>

 - `Conjunction Elimination`: 合取消除规则<br>
  若 $A\wedge B$ 可从一组命题中被引出, 则 $A, B$ 均可从中被引出:
  

$$\frac{\Gamma \vdash A \wedge B}{\Gamma \vdash A} ~~~ \frac{\Gamma \vdash A \wedge B}{\Gamma \vdash B}$$


<br>

- `Conjunction Introduction`: 合取引入规则<br>
若 $A, B$ 可从一组命题中被引出, 则 $A\wedge B$ 均可从中被引出:


$$\frac{\Gamma \vdash A ~~~ \Gamma \vdash B}{\Gamma \vdash A \wedge B}$$


<br>

- `Disjunction Elimination`: 析取消除规则<br>
若 $A\wedge B$ 可从一组命题 $\Gamma$ 中被引出, 并且 $C$ 不仅可以从 $\Gamma, A$ 中, 还可从 $\Gamma, B$ 中引出, 则 $C$ 可从 $\Gamma$ 中被引出:


$$\frac{\Gamma \vdash A \wedge B ~~~ \Gamma, A \vdash C ~~~ \Gamma, B \vdash C}{\Gamma \vdash C}$$


<br>

- `Disjunction Introduction`: 合取引入规则<br>
若 $A$ 可从一组命题 $\Gamma$ 中被引出, 则 $A\vee B, B \vee A$ 均可从中被引出:


$$\frac{\Gamma \vdash A}{\Gamma \vdash A \vee B} ~~~ \frac{\Gamma \vdash A}{\Gamma \vdash B \vee A}$$


<br>

- `Implication Elimination`: 蕴含消除规则<br>
若 $A$ 可从一组命题 $\Gamma$ 中被引出, 并且 $A \rightarrow B$ 也可以从 $\Gamma$ 中被引出, 则 $B$ 可从 $\Gamma$ 中被引出:


$$\frac{\Gamma \vdash A ~~~ \Gamma\vdash A \rightarrow B}{\Gamma \vdash B}$$


<br>

- `Implication Introduction`: 蕴含引入规则<br>
若 $B$ 可从一组命题 $\Gamma$ 和 $A$ 中被引出, 则 $A\rightarrow B$ 可从中被引出:


$$\frac{\Gamma, A \vdash B}{\Gamma \vdash A \rightarrow B}$$


<br>

- `Negation Elimination`: 否定消除规则<br>
若 $A$ 以及一组命题 $\Gamma$ 可引出矛盾 $\perp$, 则 $\neg A$ 可从 $\Gamma$ 中被引出:


$$\frac{\Gamma, A \vdash \perp}{\Gamma \vdash \neg A}$$


<br>

- `Negation Introduction`: 否定引入规则<br>
若 $A, \neg A$ 可从一组命题 $\Gamma$ 中被引出, 则可以从 $\Gamma$ 中引出任何命题, 包括 $\perp$. 


$$\frac{\Gamma \vdash A ~~~ \Gamma \vdash \neg A}{\Gamma \vdash anything}$$


<br>

- `Double Negation Elimination`: 双否定消除规则<br>
若 $\neg \neg A$ 可从一组命题 $\Gamma$ 中引出, 则 $A$ 可从 $\Gamma$ 中被引出:


$$\frac{\Gamma \vdash \neg \neg A}{\Gamma \vdash A}$$


<br>

- `Double Negation Introduction`: 双否定引入规则<br>
若 $A$ 可从一组命题 $\Gamma$ 中被引出, 则可以从 $\Gamma$ 中引出 $\neg \neg A$:


$$\frac{\Gamma \vdash A}{\Gamma \vdash \neg \neg A}$$


<br>

注: 
1. 在我们使用自然演绎推理系统进行演绎证明时, 每一步都是一系列谓词逻辑中的判断句. 
2. 所有使用自然演绎推理系统进行的证明均以公理规则 `Axiom Rule` 开始. 
3. 对于一些命题而言, 可能存在不止一种自然演绎推理证明. 

**定理 3.1 (自然演绎推理系统完备性定理)**
>本章所介绍的自然演绎推理系统具完备性, 可用于证明任何为真的谓词逻辑判断句. 

<br>

**定理 3.2 (永真式/重言式证明定理)**
> 若 $A$ 为重言式, 则可使用本章所介绍的自然演绎推理系统证明 $\vdash A$. 


## 2. 一阶谓词逻辑
在此前的内容中, 我们介绍了谓词逻辑. 通过使用谓词逻辑, 我们可以对一些命题进行建模, 但我们无法对涉及多个对象或多个条件的命题进行建模, 也无法对包含全称/特称量词的命题进行建模. 通过引入一阶谓词逻辑, 我们解决了这个问题. 

一阶谓词逻辑的主要特点是, 明确划分和指明命题的两个主要组成部分: 主语和谓语 (`Subject & Predicate`), 并且引入了 “所有” 和 “存在” 量词的概念. 

### 2.1 一阶谓词逻辑的符号化
**定义 3.2 论域**
>一段论述所讨论的全体对象组成 **论域**, 论域中的对象称为 **个体**, 它们是该论述中的语句主语和宾语的所指. 

<br>

**定义 3.3 主词**
>主词相当于自然语法中句子的主语, 指代论域中某个个体或者任何个体. 

<br>

**定义 3.4 谓词**
>相当于自然语法中句子的谓语, 指代个体的性质或多个个体之间的某种关系. 用于指代 $n$ 元关系的谓词称为 **$n$ 元谓词**. 

<Br>

**定义 3.5 量词**
>用于修饰主词的两种特殊定语: 表示 "所有" 和 “存在”的词语. 

<br>

任何一个一阶逻辑语言均由下列的三个部分组成: 

1. 符号表<br>
符号表包含非逻辑符号和逻辑符号:<br>

     逻辑符号:<br>
- 个体常元 `Constant`:<br>
用于固定地表示某个 **特定个体** 的标示符. 
- 函数符号 `function`:<br>
用于表示函数的符号. 
- 谓词符号:<br>
用于表示关系和性质, 必须至少有一个.

     非逻辑符号:<br>
- 个体变元 `Variables`:<br>
表示某个个体域内 **任何个体** 的符号, 无限多
- 量词:<br>
共有两个不同的量词: 全称量词 $\forall$ 和存在量词 $\exists$. 
- 连接词<br>
- 界符:<br>
界符共有两种: 用于分隔并列的主词的 “.”, 以及划定范围的 “()”.

以及项 `Term` 和公式 `Formula`.

<br>

### 2.2 一阶谓词逻辑中的项 `Term` 和公式 `Formula`
**定义 3.6 项 `Term`**<br>
>这个名词源于数学公式中的项. 在公式中, 项是具有完整意义的字符串. 

注:
1. 个体常元与个体变元都是项. 
2. 若 $f$ 为函数符号, $x_1, x_2, \cdots, x_n$ 均为项, 则 $f(x_1, x_2, \cdots, x_n)$ 也是项, 称为 **函项**. 

<br>

**定义 3.7 原子公式 `Atomic Formula`**<br>
>在某个一阶逻辑语言中, 若 $R$ 为谓词符号且$x_1, x_2, \cdots, x_n$ 均为项, 则称 $R(x_1, x_2, \cdots, x_n)$ 为该语言的 **原子公式**. 

注:<br>
原子公式可以表示一个简单命题. 其中, 谓词表示该命题的谓语, 谓词后面的各个项表示该命题的主语. 

**定义 3.8 谓词公式**
>1. 任何一个谓词变量都是谓词公式. 
>1. 原子公式, 以及 $\perp, \top$ 也是公式. 
>2. 若 $A$, 为谓词公式, 则 $\neg A$ 亦为谓词公式, 反之亦然.
>3. 若 $A, B$ 为谓词公式, 则 $A \vee B, A\wedge B, A \rightarrow B, A\leftrightarrow B$ 均为谓词公式. 
>4. 若 $x$为一个变量, $A$ 为一个公式, 则 $\forall x. A, \exists x. A$ 均为公式. <br>

注:<br>
一阶逻辑语言的谓词公式一般称为一阶逻辑公式或合式公式, 简称 **公式**. 

<br>

**定义 3.9**
>在公式 $\forall x.A, \exists x. A$ 中, 位于量词后面的个体变元 $x$ 称为 **指导变元**, 公式 $A$ 称为这两个量词的 **辖域**, 约束出现的变元称为 **约束变元**, 自由出现的变元称为 **自由变元**.

<br>

### 2.3 一阶谓词逻辑语言的解释
对标准的一个一阶逻辑语言的解释包含以下的四个部分:
1. 个体域: 指定某个非空集合作为该语言个体变元的值域. 
2. 个体常元所指: 将该语言中所有常元一一对应至个体域中的个体. 
3. 函数符号所指: 将该语言中所有函数符号一一对应至个体域上的函数. 
4. 谓词符号所指: 将该语言中所有谓词符号一一对应至个体域上的谓词. 

公式的解释: 即根据某一个一阶逻辑语言的解释, 对公式中的符号进行语义替换, 所得到的命题就是该公式的解释 (`interpretation`), 又称为该公式的语义 (`semanteme`).

<br>

### 2.4 一阶谓词逻辑公式的等价性
**定义 3.10**
>设 $A, B$ 为公式. 当且仅当对于所选择的个体域中全部可能的符号与变量的解释, 对 $A$ 的解释和对 $B$ 的解释相同时, 这两个公式 **语义学等价 (`Semantically Equivalent`)**, 记为 $A\equiv B$. 

<br>

**定理 3.3 (量词对偶定理)**
>以下等价式成立:<br>
>      <center>$\neg \forall x. A \equiv \exists x. \neg A$<br></center>
>      <center>$\neg \exists x. A \equiv \forall x . \neg A$</center>

<Br>

**定理 3.4 (量词基本等价定理)**
>以下等价式成立:
>      <center>$\forall x. (A \wedge B) \equiv (\forall x .A)\wedge(\forall x .B)$<br></center>
>      <center>$\exists x. (A \vee B) \equiv (\exists x.A)\vee (\exists x.B)$</center>
> <br>
> 以下等价式不成立:
>      <center>$\forall x. (A \vee B) \not\equiv (\forall x .A)\vee(\forall x .B)$<br></center>
>      <center>$\exists x. (A \wedge B) \not\equiv (\exists x.A)\wedge (\exists x.B)$</center>

<br>

**定理 3.5 (等价代换定理)**
>设 $A, B, C$ 为一阶公式, $A$ 为 $C$ 的一个子式. <br>
>若
>     <center>$A \equiv B$</center>
> 则
>     <center>$C(\cdots A \cdots) \equiv C(\cdots B \cdots)$</center>




