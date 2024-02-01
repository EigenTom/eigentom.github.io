---
layout:     post
title:      注意力机制长文
subtitle:   避免退学
date:       2024-01-06
author:     R1NG
header-img: img/post-bg-algorithm.jpg
description: 头好痛，要长脑子了
catalog: true
tags:
    - 奇技淫巧
    - 人工智障
---

# 注意力机制长文

注意力机制源于对自然语言处理领域解决 `Seq2seq` 问题的经典神经网络结构: `RNN` (循环神经网络) 的改进, 因其接近直觉的 "基于注意力和重要性对数据中元素和其他元素之间的相关性" 的逻辑设计, 在包括了自然语言处理, 计算机视觉任务, 多模态视觉语言模型等不同领域均展现出了强大的潜力和泛用性. 

本文将尝试对注意力机制的起源和主流使用场景进行简要的总结, 并对其基本原理和发展沿革进行谨慎的解释和描述.

我们首先可以从直觉上体会认知任务中 “注意力” 的有效性和重要性. 我们的 **视觉注意力** 使我们事实上以 "高关注度" 的模式观察图片的特定区域, 而对其他区域的感知则是 "低关注度" 的: 我们在看到图像时会本能地分配大脑有限的注意力资源, 将视线聚焦在更重要的目标上. 如下图所示, 人们的注意力会更多地投入到截图中的人脸, 文本标题和段落首句等关键的位置上. 

![20240128101101](https://cdn.jsdelivr.net/gh/KirisameR/KirisameR.github.io/img/blogpost_images/20240128101101.png)

同样地, 如果将图像抠掉一块, 我们也可以根据图片中剩余的信息 “脑补” 出被扣除的内容可能是什么. 以上图为例, 我们可以基于耳朵, 头发和下颚等关键信息推测出图中婴儿被遮盖的部分显然是他的侧脸, 而尿布, 四肢等元素则无助于我们对婴儿脸部内容的预测. 

除了图像信息外, 我们也能在文本处理任务中观察到 **文本注意力** 的关键作用. 在给定的句子中, 单词之间的关系存在明显的差异, 且这样的差异往往由其语义信息所决定. 在下图的例子中, “green” 和 “apple” 这一对单词之间的 **相关性** 就要显著高于 “eating” 和 ”green“. 在我们看到这句话时, 我们会本能地预测和把握高相关性的单词对, 并通过这种方式正确地理解句子的含义:

![20240128102505](https://cdn.jsdelivr.net/gh/KirisameR/KirisameR.github.io/img/blogpost_images/20240128102505.png)

显然, 在上面的例子中, 如果计算机视觉/自然语言处理深度学习神经网络具备了理解和区分输入信息中不同区域 **重要性** 或 **相关性** 的能力, 则它们更有可能准确理想地完成 **图片补全**, **目标检测**, **内容理解** 等任务. 

深度学习中的注意力机制在本质上被广义地理解为 **描述输入数据中不同区域重要性的权重向量**. 它可以 **定量** 地描述输入数据中 **目标元素** 和 **其他元素** 之间 **相关性的强弱**. 

其基本思想为: 在处理 **序列化** 的数据时, 每个元素都与给定序列中的 **其他所所有元素** 而非相邻元素 **建立相关性的关联**, 由此通过计算元素之间的相对重要性来 **自适应** 地捕捉元素之间的 **长距离依赖关系**. 

下面, 我们从注意力机制的起源开始, 逐步描述注意力机制的发展演变和主要使用场景. 

## `Bahdanau` 注意力机制 (`Additive Attention`)

在讨论注意力机制的原始形式前, 首先我们需要再向前回溯一步, 回顾自然语言处理领域的一个经典问题: `Seq2Seq` (序列到序列) 问题. 

`Seq2Seq` 模型于 $2014$ 年在论文: [Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) 和 [Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) 中提出, 将自然语言处理问题建模为 **从任意长度的输入序列 (源序列) 转换为新的输出序列 (目标序列)** 的过程, 在机器翻译, 对话问答, 内容理解等任务上都取得了在当时优秀的表现. 

一般地, `Seq2Seq` 模型均由 **编码器 (`Encoder`)** 和 **解码器 (`Decoder`)** 两部分组成:

1. 前者负责处理输入序列, 将其中包含的信息压缩为 **固定长度** 的 **上下文向量 (`Context Vector`)**, 由此实现对输入信息的总结和抽象化表示. 

2. 后者负责对上下文向量进行 **变换** 并生成输出序列, 也就是 **解码**:

![20240128105308](https://cdn.jsdelivr.net/gh/KirisameR/KirisameR.github.io/img/blogpost_images/20240128105308.png)

数学上, 可以将编码-解码的流程视为: 

给定输入 

$$X = < x_1, \cdots, x_m>$$

目标输出 $Y$ 记为

$$Y = <y_1, \cdots, y_n>$$

则编码器将输入通过非线性变换转化为上下文向量 $C$:

$$C = \mathscr{F}(x_1, x_2, \cdots, x_m)$$

而解码器则根据输入序列 $X$ 的中间表示 $C$ 以及 **之前已经生成的历史信息** $<y_1, \cdots, y_{i-1}>$ 生成第 $i$ 时刻要得出的输出序列元素 $y_i$:

$$y_i = \mathscr{G}(C, y_1, \cdots, y_{i-1})$$

换言之, 生成目标输出序列中每个元素的过程形式为:

$$y_1 = \mathscr{G}(C) \\ y_2 = \mathscr{G}(C, y_1) \\ y_3 = \mathscr{G}(C, y_1, y_2) \\ \cdots$$ 

### 起源

被设计用于解决 `Seq2Seq` 任务, 以 **编码器 - 固定长度上下文向量 - 解码器** 为架构的神经网络模型存在 `Encoding Bottleneck`, `Gradient Exploding/Vanishing` 等致命缺点: 由于在编码过程中序列信息在每一步中都会被总结编码并作为下一状态编码器的输入. 

对较长的时序信息而言, 序列中较早的信息更有可能在整个编码过程中被逐步丢失, 这也就是 **长时依赖** 问题. 为了解决该问题, 注意力机制由 `Bahdanau` 等人于 [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) 中提出. 

在上面的 `Encoder-Decoder` 模型中我们不难看出, 该架构在生成目标序列时, 无论生成什么元素, 都使用了 **相同** 的 **输入隐表示 $C$**. 这意味着, 无论解码器生成输出序列中的哪个元素, 输入序列中的元素对它的 **影响力** 都是等同的.

### 解释

通过将编码器生成的中间表示 $C$ 替换为根据当前解码器生成元素 $y_i$ 而 **不断变化** 的可变 $C_i$, 就可以为 `Encoder-C-Decoder` 模型引入注意力机制. 也就是说, 生成目标序列中元素的过程可被表示为:


$$y_1 = \mathscr{G}(C_1) \\ y_2 = \mathscr{G}(C_2, y_1) \\ y_3 = \mathscr{G}(C_3, y_1, y_2) \\ \cdots$$ 

其中, 每个可变 $C_i$ 都表示输入序列中每个元素的 **注意力分配概率分布**.

在本篇论文中, $C_i$ 被定义为

$$C_i = \sum_{j=1}^{T_x}\alpha_{ij} \cdot h_j$$

其本质是 **输入序列中每个元素在RNN编码器对应模块的隐表示** 和 **每个输入元素关于输出序列的第 $i$ 个元素之间相关性概率值** 的 **加权和**.

其中, $T_x$ 是输入序列中元素的数量 (也就是输入序列的长度), 对每个编号 $j \leqslant T_x \in \mathbb{N}$, $h_j$ 即为编码器中对应RNN块的 **隐表示 (`Hidden Representation`)**. 

而 $\alpha_{ij}$ 则为 **输入序列** 关于 **输出序列** 中第 $i$ 个元素的 **注意力分配概率分布值**, 计算方式为 

$$\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}$$

也就是通过 `softmax` 函数计算得来. 其中 $e_{i, k}$ 为 输入序列中第 $k$ 个元素和输出序列中第 $i$ 个元素之间的 **相关性打分 (`score`)**, 评估它们之间的匹配性 (`match`). 

在本论文中, 相关性打分形式化为

$$e_{ij} = a(s_{i-1}, h_j)$$

以:

1. **输入序列的第 $j$ 个元素关于编码器RNN模块的隐表示**: $h_j$
2.  **解码器从输出序列的第 $i-1$ 位置的RNN模块所计算出来的, 给第 $i$ 位置处的RNN模块作为输入的一部分的隐表示**: $s_{i-1}$

作为参数, 通过 **对齐模型 (`alignment model`)**: $a()$ 计算得出. 

在本论文中, 对齐模型被设计为一个简单的 **前馈神经网络**:

$$e_{i, j} = a(s_{i-1}, h_j) = v^{\intercal} \cdot \tanh(W \cdot s_{i-1} + U\cdot h_j)$$

其中 $W \in \mathbb{R}^{n \times n}$, $U \in \mathbb{R}^{n \times 2n}$, $v \in \mathbb{R}^{n}$, 均为权重矩阵.

对齐模型的输入参数为: **输入和输出序列的隐表示层状态**. 它和编码器, 解码器等模型的其余部分一同参与模型训练, 学习输入和输出序列之间的对齐关系 (注意力权重).

![20240128144900](https://cdn.jsdelivr.net/gh/KirisameR/KirisameR.github.io/img/blogpost_images/20240128144900.png)

### 实现

参照 [Seq2Seq Learning & Neural Machine Translation](https://nthu-datalab.github.io/ml/labs/12-1_Seq2Seq-Learning_Neural-Machine-Translation/12-1_Seq2Seq-Learning_Neural-Machine-Translation.html) 基于 `PyTorch` 重新进行了实现, 下面进行简要解释;

首先进行数据处理. 给定的数据为用 `\t` 分列, 用 `\n` 换行的 `txt` 文件. 首先将 `txt` 文件按行读入, 然后对中文一列和英文一列分别进行预处理.

在预处理过程中, 将 `Unicode` 字符全部转换为 `ASCII` 字符, 并分别清除中文/英文列中的所有非中文/英文字符, 并对英文按单词, 对中文按汉字用空格分割. 

~~~python
def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')

def preprocess_eng(w):
    w = unicode_to_ascii(w.lower().strip())

    # creating a space between a word and the punctuation following it
    # eg: "he is a boy." => "he is a boy ."
    # Reference:- https://stackoverflow.com/questions/3645931/
    # python-padding-punctuation-with-white-spaces-keeping-punctuation
    w = re.sub(r"([?.!,])", r" \1 ", w)
    # replace several spaces with one space
    w = re.sub(r'[" "]+', " ", w)

    # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
    w = re.sub(r"[^a-zA-Z?.!,]+", " ", w)
    w = w.rstrip().strip()

    # adding a start and an end token to the sentence
    # so that the model know when to start and stop predicting.
    w = '<start> ' + w + ' <end>'
    return w

def preprocess_chinese(w):
    w = unicode_to_ascii(w.lower().strip())
    w = re.sub(r'[" "]+', "", w)
    w = w.rstrip().strip()
    w = " ".join(list(w))  # add the space between words
    w = '<start> ' + w + ' <end>'
    return w

# 1. Remove the accents
# 2. Clean the sentences
# 3. Return word pairs in the format: [ENGLISH, CHINESE]
def create_dataset(path, num_examples=None):
    lines = open(path, encoding='UTF-8').read().strip().split('\n')
    word_pairs = [[w for w in l.split('\t')] for l in lines[:num_examples]]
    word_pairs = [[preprocess_eng(w[0]), preprocess_chinese(w[1])]
                  for w in word_pairs]

    # return two tuple: one tuple includes all English sentenses, and 
    # another tuple includes all Chinese sentenses
    return word_pairs
~~~

经过处理后的中英文本形式如下:

~~~bash
<start> if a person has not had a chance to acquire his target language by the time he s an adult , he s unlikely to be able to reach native speaker level in that language . <end>
<start> 如 果 一 個 人 在 成 人 前 沒 有 機 會 習 得 目 標 語 言 ， 他 對 該 語 言 的 認 識 達 到 母 語 者 程 度 的 機 會 是 相 當 小 的 。 <end>
Size: 20289
~~~

随后对中文和英文训练数据分别进行 `tokenize` 和 `padding` 处理. 此处调用 `tensorflow` 内置的 `tokenizer`, 实际功能是对英文符号/中文汉字/句子起始和终止符从 $1$ 开始编号. 

最后使用 `tf.keras.preprocessing.sequence.pad_sequences` 分别对中文列和英文列在每一行的后面填充 $0$, 直到所有中文/英文行的长度都达到最长的中文/英文句子长度. 

然后加载数据集, 并返回索引化的中文和英文训练语料, 以及对应的中英文 `tokenizer`, 留作模型推理过程中备用. 

~~~python
def max_length(tensor):
    return max(len(t) for t in tensor)

def tokenize(lang):
    # “lang”: a tuple of strings, in the string each token is separated using spaces
    # 1. for each string in the tuple, separate all tokens
    # 2. add "<start>", "<end>" as the 1st and 2nd token in the token dictionary
    # 3. convert each sentence in the tuples into tensor arrays utilizing tokenizers
    # 3. pad each sentence using "0" backwards for each sentence tensor array

    lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
        filters='')
    # generate a dictionary, e.g. word -> index(of the dictionary)
    lang_tokenizer.fit_on_texts(lang)

    # output the vector sequences, e.g. [1, 7, 237, 3, 2]
    tensor = lang_tokenizer.texts_to_sequences(lang)

    # padding sentences to the same length
    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor,
                                                           padding='post')
    return tensor, lang_tokenizer

def load_dataset(path, num_examples=None):
    # creating cleaned input, output pairs
    # regard Chinese as source sentence, regard English as target sentence
    targ_lang, inp_lang = zip(*create_dataset(path, num_examples))
    
    print(type(inp_lang))
    print(inp_lang[:5])

    input_tensor, inp_lang_tokenizer = tokenize(inp_lang)
    target_tensor, targ_lang_tokenizer = tokenize(targ_lang)

    return input_tensor, target_tensor, inp_lang_tokenizer, targ_lang_tokenizer
~~~

随后构建数据集并声明相关变量:

~~~python
BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 128
steps_per_epoch = len(input_tensor_train)//BATCH_SIZE
embedding_dim = 256
units = 1024
# 0 is a reserved index that won't be assigned to any word, so the size of vocabulary should add 1
vocab_inp_size = len(inp_lang.word_index) + 1
vocab_tar_size = len(targ_lang.word_index) + 1

dataset = tf.data.Dataset.from_tensor_slices(
    (input_tensor_train, target_tensor_train)).shuffle(BUFFER_SIZE)
dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)
~~~

此处的 `unit` 指 `GRU unit` 输出隐表示向量的维度. 

然后分别定义 `Encoder`, `BahdanauAttention` 和 `Decoder`:

~~~python
from torch import nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim
        )

        # pytorch use tanh as activation function for GRU units
        # and initialize them using uniform distribution
        self.gru = nn.GRU(
            embedding_dim,
            enc_units, 
            batch_first = True
        )
    
    def forward(self, x, hidden):
        # x: training data with shape (batch_size, max_length) -> (128, 46)
        # hidden: hidden states with shape (batch_size, units) -> (128, 1024)
        # after embedding, the input batch's shape is converted into (batch_size, max_length, embedding_dim) -> (128, 46, 256)
        x = self.embedding(x)   # convert the input sentence into hidden representations
        
        # Now perform computation through GRU units
        # output contains the state(in GRU, the hidden state and the output are same) from all timestamps,
        # output shape == (batch_size, max_length, units) -> (128, 46, 1024)
        # state is the hidden state of the last timestamp, shape == (batch_size, units) -> (128, 1024)
        output, state = self.gru(x, hidden)

        return output, state

    def initialize_hidden_state(self):
        return torch.zeros((1, self.batch_sz, self.enc_units))

class BahdanauAttention(nn.Module):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = nn.Linear(1024, units)    # fully connected layer: 1024*units matrix; input_dim = 1024, output_dim = `units`
        self.W2 = nn.Linear(1024, units)    # fully connected layer: 1024*units matrix; input_dim = 1024, output_dim = `units`
        self.V = nn.Linear(units, 1)         # fully connected layer: input_dim = 1024, output_dim = 1

    def forward(self, query, values):
        # query shape: (1, batch_size, hidden_size) -> (1, 128, 1024)
        # values shape: (batch size, sequence length, units) -> (128, 46, 1024)
        # (values) hidden_with_time_axis shape: (batch_size, hidden_size)
        query_with_time_axis = query.permute(1, 0, 2)   # from [1, 128, 1024] to [128, 1, 1024]
        
        # score shape: (batch_size, max_length, 1)
        # acording to paper: v * tanh(W1*values + W2 * query)
        
        # score: [10x1] * [128*10] => [128x46x1]
        score = self.V(
            # [128x46x10] + [128x1x10] (broadcast along 2nd dimension)
            torch.tanh(self.W1(values) + self.W2(query_with_time_axis))
        )

        # score shape: (batch_size, max_length, 1)
        # produced by the "alignment model"
        attention_weights = F.softmax(score, dim=1) # [128x46x1]

        # context_vector shape == (batch_size, max_length, hidden_size)
        # perform weighted sum with computed attention weights with GRU units' hidden outputs 
        context_vector = attention_weights * values
        context_vector = torch.sum(context_vector, dim=1)   # [128x1024]


        return context_vector, attention_weights

class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.embedding = nn.Embedding(
            vocab_size, 
            embedding_dim
        )   # 6082x256

        self.gru = nn.GRU(
            embedding_dim + dec_units, 
            dec_units, 
            batch_first = True
        )

        self.fc = nn.Linear(dec_units, vocab_size)
        self.attention = BahdanauAttention(dec_units)

    def forward(self, x, hidden, enc_output):
        context_vector, attention_weights = self.attention(hidden, enc_output)
        x = self.embedding(x)

        x = torch.cat(
            tensors = (context_vector.unsqueeze(1), x),
            dim = -1
        )

        output, state = self.gru(x, hidden)
        output = output.view(
            -1, 
            output.size(2)
        )

        x = self.fc(output)
        return x, state, attention_weights
~~~

模型的训练, 推理和效果参见原 `Tensorflow` 实现. 

## 注意力机制的一般化形式

下面我们将注意力机制从 `Encoder-Decoder` 框架中剥离, 并对齐进行进一步的抽象. 

本质上, 注意力机制的计算可被视为:

$$\text{Attention}(\text{Query}, \text{Source}) = \sum_{i=1}^{L_x}\text{Similarity}(\text{Query}, \text{Key}_i) \cdot \text{Value}_i$$

其中, $L_x$ 表示输入序列, 即 $\text{Source}$ 的长度. 由此可以将注意力机制视为:

1. $\text{Source}$ 中的元素由一系列的 $\text{<Key, Value>}$ 键值对构成.
2. $\text{Query}$ 是目标输出序列 $\text{Target}$ 中的某个元素. 
3. 给定 $\text{Query}$, 通过计算它和 $\text{Source}$ 中不同 $\text{Key}$ 之间的相似性/相关性, 就可以得到每个 $\text{Key}$ 的权重系数: $\text{Similarity(Query, Value)}$. 
4. 对每个 $\text{Key}$ 对应的 $\text{Value}$ 关于权重系数求和, 就得到了最终的 `Attention` 值. (如果 $\text{Value}$ 是向量, 则最终得到的就是注意力向量)

在 `Bahdanau` 注意力机制中, 计算注意力向量的过程中 $\text{Source}$ 中的 $\text{Key}$ 和 $\text{Value}$ 本质是合二为一的, 也就是输入句子中每个单词对应的隐表示 (语义编码). 

### 对注意力机制的抽象化理解

从抽象化的注意力机制计算流程来看, 我们可以用如下的两种方式理解注意力机制:

#### 解释1: 信息聚焦

注意力机制被解释为: 从大量信息中选择性地筛选出少量重要信息, 并聚焦到这些信息上而忽略其余不重要信息的过程. 在这种解释中, “聚焦” 体现在对输出序列中不同的 `Query` 和输入序列中不同 `Key` 之间 **权重的计算** 上, 其大小代表了 `Key` 关于 `Query` 的重要性, 而和 `Key` 对应的 `Value` 就是信息本身. 

#### 解释2: 软寻址 (`Soft Addressing`)

注意力机制被解释为: 某种软寻址. 输入序列 `Source` 被视为 **存储器中所存储的内容**, 其元素均为 **键值对**: `Key-Value`. 

而给定输出序列中某个位置 `Query`, 要计算它关于 `Source` 的注意力值 (注意力向量) 的过程就是对这个存储器执行 `key = Query` 的 **查询** 的过程, 查询目的就是从该存储器中 根据 `Query` 和存储的元素 `Key` 进行 **相似性/相关性比较** 进行 **寻址** 取出对应的数值, 也就是注意力值 (注意力向量). 

我们称这种寻址为 **软寻址**, 因为它在寻址过程中会从 **每个 `Key` 地址** 取出内容, 计算它的重要性 (由它关于 `Query` 的相似性/相关性决定), 然后对每个 `Key` 对应的 `Value` 加权求和取出 **最终的 `Attention` 值**, 而非像常规寻址一样, 只从存储内容中找出一条. 

### `Attention` 值 (向量)的计算

在绝大部分注意力机制中, `Attention` 值 (向量) 的计算过程都可被抽象归纳为两个步骤:

1. 根据输出序列中指定的位置 `Query` 和 输入序列中的不同 `Key`, 计算权重系数. 
2. 根据第一步中所计算出的权重系数, 对输入序列中每个 `Key` 对应的 `Value` 进行 **加权求和**.

进一步地, 第一步还可被细分为两个阶段: **计算 `Query` 和 `Key` 的原始相关性打分** 和 **对计算得出的原始分值归一化 (`Normalize`)**. 

由此, 注意力机制中 `Attention` 值 (向量) 的计算流程可以被如下可视化:

![20240130105949](https://cdn.jsdelivr.net/gh/KirisameR/KirisameR.github.io/img/blogpost_images/20240130105949.png)

具体地, 在第 `1.1` 阶段, 也就是对 `Query` 和 `Key` 的原始相关性打分的计算中, 可以选择 **求两者的向量点积** (如 `Luong` 注意力), **求两者的向量余弦相似度** 或引入 **前馈神经网络** (如 `Bahdanau` 注意力) 等方法.

在第 `1.2` 阶段, 可以通过简单的归一化 (`Normalization`) 将 `1.1` 阶段中计算得出的相关性打分整理为 **所有元素权重和为 $1$ 的概率分布**, 也可使用 `SoftMax` 将原始数值进行转换, 进一步突出重要元素的权重. 

$$a_i = \text{SoftMax}(S) = \frac{s_i}{\sum_{i=1}^{L_x}\exp(s_i)}$$

在第 `2` 阶段, 执行的操作就是简单的加权求和:

$$\text{Attention(Query, Source)} = \sum_{i=1}^{L_x} a_i \cdot \text{Value}_i$$

由此, 针对 `Query` 的注意力值 (向量) 得以求出.

### 注意力机制的各种变形

#### 简单的 `Scaled Dot-Product Attention`

首先介绍一种结构较为简单且被广泛使用的 `Attention` 构造方式. 该构造又称 `Scaled Dot-Product Attention`. 我们用文本分类任务作为例子对其进行介绍:

![20240130154710](https://cdn.jsdelivr.net/gh/KirisameR/KirisameR.github.io/img/blogpost_images/20240130154710.png)

假设输入的文本长度为 $n$, 记为 $x = e_1, e_2, \cdots, e_n$. 则经过 `RNN` 编码后得到输入文本序列中每一步的 `RNN` 模块 `hidden state` (均为 $1 \times d$ 维的向量): 

$$h_1, \cdots, h_n \in \mathbb{R}^{d}.$$

接下来将它们同时视为 `Key` 和 `Value`, 而对应输出序列每一步的查询向量用 $V \in \mathbb{R}^{d}$ 表示. 

接下来使用 **向量点积** 作为打分函数. 即: 对任意给定的第 $i$ 个 `Key`, 其关于查询的分数 $\text{score}$ 为:

$$\text{score}_i = \frac{h_i^{\top}\cdot V}{\lambda}$$

其中, $\lambda$ 为一个用于放缩点积结果, 让经过 `SoftMax` 后的所得权重更平滑的超参数, 一般设为 $\sqrt{d}$. 

这种通过使用超参数 $\lambda$ 进行缩放的点积计算注意力权值的方式又称 **缩放点积**.

注意力权重 (`Attention Weight`) 由得到的打分通过 `SoftMax` 函数计算得来:

$$\alpha_i = \frac{\exp(a_i)}{\sum_{j=1}^{n} \exp(a_j)}$$

最后对 `Value` 关于对应的注意力权重做加权求和, 就得到了该 `Query` 的注意力值.

#### `Self-Attention` 中的注意力机制实现

`Self-Attention` 可视为 **特征提取层**, 给定输出 $I$, 得出经过特征融合后得到的输出 $O$. 此处我们只对其内部的注意力机制实现和注意力值 (向量) 计算方法进行介绍. 

![20240130163222](https://cdn.jsdelivr.net/gh/KirisameR/KirisameR.github.io/img/blogpost_images/20240130163222.png)

假设输入的 **特征序列** 长度为 $n$, 记为 $I = a_1, a_2, \cdots, a_n$. `Self-Attention` 中规定, 注意力机制概念中的 `Query`, `Key` 和 `Value` 分别由特征向量 $a_i$ 和三个 **参数可学习** 的全连接层 (可视为矩阵) $W^q, W^k, W^v$ 计算得到:

$$\begin{align}&\text{Query}: q^i = W^q \cdot a_i \\ &\text{Key}: k^i = W^k \cdot a_i \\ &\text{Value}: v^i = W^v \cdot a_i\end{align}$$

由此关于输入特征 $I$ 得到关于它自身的 `Query`, `Key` 和 `Value` 矩阵: 

$$\begin{align}&\text{Query}: Q = W^q \cdot I \\ &\text{Key}: K = W^k \cdot I \\ &\text{Value}: V = W^v \cdot I\end{align}$$

随后计算注意力值矩阵 $A$:

$$A = K^{\top} \cdot Q$$

然后对 $A$ 通过 **归一化** 后得到 $\hat{A}$, 乘上 $V$, 计算得出最后的输出特征 $O$:

$$O = V \cdot \hat{A}$$

简单的 `PyTorch` 实现如下:

~~~python
import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, embed_dim):
        super(SelfAttention, self).__init__()
        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        attn_weights = torch.matmul(q, k.transpose(1, 2))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attended_values = torch.matmul(attn_weights, v)
        return attended_values

class SelfAttentionClassifier(nn.Module):
    def __init__(self, embed_dim, hidden_dim, num_classes):
        super(SelfAttentionClassifier, self).__init__()
        self.attention = SelfAttention(embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        attended_values = self.attention(x)
        x = attended_values.mean(dim=1)  
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return x
~~~

## 注意力机制的应用

下面, 我们对注意力机制的应用进行简要讨论.

### 自注意力机制 `Self-Attention`

自注意力机制 (`Self-Attention`) 首先在2017年的经典论文 [Attention is All you need]() 中被提出, 其核心思想为: 在处理序列数据时, 使用注意力机制计算输入序列中每个元素与 **同一个序列** 中其他元素之间的关系. `Self-Attention` 可视为 **特征提取层**, 给定输出 $I$, 得出经过特征融合后得到的输出 $O$.

由此, 自注意力机制得以解决 **神经网络对多个相关的输入之间无法建立起相关性** 的问题. 作为注意力机制的变体, 自注意力机制中的 $Q, K, V$ 均 **来源相同**, 因此自注意力机制中的 `Q, K, V` 计算方式也和常规的注意力机制有很大不同. 

自注意力机制中 `Attention Mechanism` 的实现方式, 计算方法和原理参见本文章节: “注意力机制的一般化形式-注意力机制的各种变形-`Self-Attention` 中的注意力机制实现”.

自注意力机制的功能是筛选出重要信息而将相对不那么重要的信息过滤. 在 `CV` 领域应用时, 由于自注意力机制无法充分利用图像的尺度和平移不变性, 以及图像的特征局部性等先验知识, 因此需要通过大量数据学习它们, 由此导致自注意力机制对有效信息的提取能力要 **比卷积神经网络更差**, 只有在 **训练数据规模较大** 的前提下才能有效地建立准确的全局关系. 

此外, 自注意力机制 **并不保留输入序列中向量之间的位置关系**. 在 `NLP` 领域应用中, 由于单词/文字在句中的位置会导致词语具有不同的性质, 因此需要引入 **位置编码 (`Positional Encoding`)** 进行信息的补齐: 对每个输入向量加上位置向量, 一并带入自注意力层进行计算. 

### 多头自注意力机制 `Multi-Head Self Attention`

`Self-Attention` 中实现的注意力机制中, **参数可学习** 的全连接层 (可视为矩阵) 为: $W^q, W^k, W^v$. 

每个输入特征向量 $a_i$ 分别和这三个矩阵相乘得到对应的 $q^i, k^i, v^i$ 参与后续注意力权值和输出特征 $O$ 的计算. 

由于只有 **一组** 可学习的, 参与计算注意力权值的矩阵 $W^q, W^k, W^v$, 这样的注意力机制又称 **单头自注意力机制**. 为了更好的并行计算捕获更多维度的信息, 包括 **多组** 可学习的, 参与计算注意力权值的矩阵 的注意力机制算法被提出, 这就是所谓的 **多头自注意力机制**.

在实践中, 我们希望在给定相同的查询, 键值对的集合时, 模型能够基于 **相同的注意力机制** 学习到不同的相关关系. 随后, 通过将这些 **独立** 学习得到的信息组合起来, 从而得以捕捉到序列内各种范围的依赖关系. 

由此, 我们使用 **独立学习** 得到的多组不同的注意力权值计算矩阵 `W^q, W^k, W^v` 变换给定的查询和键值对, 将这些独立学习得到的信息提取输出相 **拼接**, 通过另一个可学习的线性投影 (`Linear Projection`) 进行变换, 产生最终输出. 

![20240130211250](https://cdn.jsdelivr.net/gh/KirisameR/KirisameR.github.io/img/blogpost_images/20240130211250.png)

![6ba45518a73649e9818594897369ff57](https://cdn.jsdelivr.net/gh/KirisameR/KirisameR.github.io/img/blogpost_images/6ba45518a73649e9818594897369ff57.gif)

上图的例子中描述的就是一个包含了两组可学习矩阵的 **双头自注意力机制**.

#### 多头自注意力机制的实现

首先定义注意力机制的头数, 以及输入特征维度. `hidden_size` 为 $q^i, k^i, v^i$ 的总维度.

~~~python
self.num_attention_heads = num_attention_heads
self.attention_head_size = int(hidden_size / num_attention_heads)
self.all_head_size = hidden_size
~~~

其次定义 $W^q, W^k, W^v$:

~~~python
self.key_layer = nn.Linear(input_size, hidden_size)
self.query_layer = nn.Linear(input_size, hidden_size)
self.value_layer = nn.Linear(input_size, hidden_size)
~~~

随后使用输入特征 $x$ 和 $W^q, W^k, W^v$ 相乘得到 `Query`, `Key` 和 `Value` 矩阵, 维度为 `(batch_size, seq_length, hidden_size)`:

~~~python
key = self.key_layer(x)
query = self.query_layer(x)
value = self.value_layer(x)
~~~

计算后需要将 `seq_length` 和 `num_attention_heads` 维度对换, 让最终得到的 $Q, K, V$ 张量维度为 `(batch_size, num_attention_heads, seq_length, attention_head_size)`:

~~~python
def trans_to_multiple_heads(self, x):
    new_size = x.size()[ : -1] + (self.num_attention_heads, self.attention_head_size)
    x = x.view(new_size)
    return x.permute(0, 2, 1, 3)
key_heads = self.trans_to_multiple_heads(key)
query_heads = self.trans_to_multiple_heads(query)
value_heads = self.trans_to_multiple_heads(value)
~~~

然后进行计算: $\frac{Q \cdot K}{\sqrt{\text{attention\_head\_size}}}$:

~~~python
attention_scores = torch.matmul(query_heads, key_heads.permute(0, 1, 3, 2))
attention_scores = attention_scores / math.sqrt(self.attention_head_size)

# perform normalization
attention_probs = F.softmax(attention_scores, dim = -1)
~~~

然后将得到的注意力矩阵和 `Value` 矩阵 $V$ 相乘得到输出特征, 维度为 `(batch_size, num_attention_heads, seq_length, attention_head_size)`:

~~~python
context = torch.matmul(attention_probs, value_heads)
~~~

最后将各个头得到的输出特征进行拼接:

~~~python
# “contiguous()” is crucial to store tensor into a consecutive memory space
# to prevent errors happen when performing .view() operation
# REF: https://blog.csdn.net/kdongyi/article/details/108180250
context = context.permute(0, 2, 1, 3).contiguous()
new_size = context.size()[ : -2] + (self.all_head_size , )
context = context.view(*new_size)
~~~

### 与RNN的结合
参考本文章节:  "`Bahdanau` 注意力机制-实现".

### Transformer

`Transformer` 是 `Google` 在 [Attention is all you need](https://arxiv.org/abs/1706.03762) 中提出的一种 **完全基于注意力机制** 的新模型结构. 传统的 `Seq2Seq` 模型基于结构复杂的 `CNN` 或 `RNN`, 一般由编码器和解码器两部分组成, 但这样的模型难以并行计算和训练, 且输入序列中较早的信息容易丢失. `Transformer` 架构完全摒弃了传统 `Seq2Seq` 模型架构中的 `RNN` 结构, 具有并行度高, 训练时间短等显著优势. 

和使用 `CNN` 代替 `RNN` 的 `Seq2Seq` 模型相比, 由于卷积神经网络实际上可以视为 **特殊的 `Attention`** (卷积神经网络中的感受野大小固定, `Attention` 机制的感受野是动态计算得出的), 因此引入自注意力机制就避免了卷积神经网络需要多次卷积才能结合距离较远的数据, 在长时序信息的特征提取上表现不佳的问题.

下面介绍 `Transformer` 模型的架构:

#### 解释

大多数表现优秀的 `Seq2Seq` 模型都包括 `Encoder` 和 `Decoder` 两部分: 前者将输入

$$I = x_1, x_2, \cdots, x_n$$

映射为一个连续 表示的序列

$$z = z_1, z_2, \cdots, z_n.$$

给定从编码器得出的输出 $z$, 解码器以 **自回归 (`Auto-regressive`)** 的方式 **一个一个地** 生成输出序列

$$O = y_1, y_2, \cdots, y_n.$$

其中 **自回归**, 是指在生成下一个字符 $y_i$ 时, 总是将 **先前生成的符号 $y_{i-1}$** 作为解码器的附加输入的生成方式.

`Transformer` 的基础结构仍然包括编码器 (`Encoder`) 和解码器 (`Decoder`) 两部分. 编码器和解码器均由三类网络结构组成: **多头自注意力层**, **前馈神经网络 (全连接层)** 和 **`LayerNorm` 层**. 

![20240130212023](https://cdn.jsdelivr.net/gh/KirisameR/KirisameR.github.io/img/blogpost_images/20240130212023.png)

上图中左侧和右侧部分分别对应了 `Transformer` 模块中的编码器和解码器. 我们已经知道 **多头自注意力层** 和 **全连接层** 分别起到了信息提取和作线性变换, 对多维度信息进行混合的作用. 下面解释 `LayerNorm` 层的功能:

和 `Batch Normalization` 相似, `Layer Normalization` 作为 **归一化层**, 处理的对象不是如 `Batch Normalization` **一样对输入的一批样本** 的 **同一维度特征** 作归一化, 而是对 **单个样本** 的 **所有维度特征** 作归一化. `Transformer` 架构中不适用 `Batch Normalization` 的主要原因是: 成批输入的样本 `batch` 中序列长度 **可能不一致**, 而若要对这批样本进行归一化就需要补齐长短不一的输入样本. 

使用 $0$ 等空数据补齐样本, 将使较短序列中实际包含信息的向量规模相对减小而较长序列中实际包含信息的向量规模相对增大, 导致误差抖动增加, 无法起到稳定输入分布的效果. 

上图中展示的是构成 `Transformer` 中编码器和解码器的 **基本层** (`layer`):
1.  `Transformer` 的编码器由 $N=6$ 个 **编码器基本层** 串行堆叠构成, 每个编码器基本层中都包括两个主要部分: 作特征提取的 **多头自注意力机制层** 和起信息混合作用的 **全连接层**. 

    为了确保经过特征提取和信息混合后神经网络获取的信息至少不会比什么也不做要差, 这两个子层后都如同 `ResNet` 一样采用了 **残差连接**, 也就是说每个子层 `Sublayer()` 的输出都是 `LayerNorm(x + Sublayer(x))`.

    为方便残差连接的拼接, 模型中所有子层和 `embedding` 层的输出维度都是 $512$ 维.

    ![20240131155733](https://cdn.jsdelivr.net/gh/KirisameR/KirisameR.github.io/img/blogpost_images/20240131155733.png)

2. 解码器同样由 $N=6$ 个 **解码器基本层** 串行堆叠构成. 每个解码器基本层中除了多头自注意力机制层和全连接层以外, 还额外加入了 **对编码器的输出进行多头自注意力机制操作** 的中间层. 基于解码器需要执行自回归预测任务的特点, 它对输入的多头自注意力层还被添加了 **掩膜** (`Mask`), 在训练时, 掩膜会阻止注意力层中第 $i$ 个位置的预测接触到任何序列后面的已知输出, 防止 “作弊”.
   
   ![20240131155746](https://cdn.jsdelivr.net/gh/KirisameR/KirisameR.github.io/img/blogpost_images/20240131155746.png)

**自注意力机制** 和 **多头自注意力机制** 在上文中已有详细解释, 故此处不再赘述. 需要关注的是, 多头 (自) 注意力机制在 `Transformer` 中有三种不同的应用:

1. 在编码器基本层中使用了 **多头自注意力机制** 层.
2. 解码器基本层中, 三个子层里的第一层为 **加入了掩膜的多头自注意力机制层** (`Masked Multi-Head Attention`).
3. 解码器基本层中, 三个子层里的第二层为 **关于掩膜多头自注意力机制层的输出** 和 **编码器层输出** 的 **多头注意力机制**. 其中, $Q$ 由 **掩膜多头自注意力机制层的输出** 提供, 而 $K$ 和 $V$ 来源于 **编码器层输出**. 

`Transformer` 架构中引入了 `Embedding` 词向量层对输入和输出字符向量化, 并使用线性变换和 `Softmax` 函数将解码器的输出转换为字典中每一个字符是预测的下一个字符的概率. 

`Transformer` 架构完全基于注意力机制而不包含 `RNN` 或 `CNN`, 对序列输入中数据的相对位置排布不敏感: 由于注意力机制的输出为 $V$ 的加权和, 而权重仅为 $Q$ 关于 $K$ 的相似度, 相似度又与位置信息无关, 只和语义信息有关, 因此一个序列被打乱后作为输入经过注意力机制层, 输出的结果是和不打乱一致的. 

为了规避注意力机制 **不保留序列中数据的顺序信息** 的问题, 作者设计了 **位置编码 (`Positonal Encoding`)** 来编码需俩中关于字符的相对或绝对位置信息. 

位置编码被添加到编码器和解码器 (完整的编码器/解码器) 的输入向量 (`Embedding`) 中. 位置编码的维度被设计为和输入向量的维度相同, 使得它们可以被简单 **相加** 后传入编码器或解码器. 作者选择使用 **不同频率的正弦/余弦函数** 作为位置编码: 位置编码的每个维度都唯一地对应一个正弦曲线:

$$\begin{align} &\text{PositionalEmbedding}_{(\text{pos}, 2i)} = \sin(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}) \\ &\text{PositionalEmbedding}_{(\text{pos}, 2i+1)} = \cos(\frac{\text{pos}}{10000^{\frac{2i}{d_{\text{model}}}}}) \end{align}$$

#### 实现

下面展示 `PyTorch` 中预实现的 `Transformer` 层的调用:

~~~python
import torch
import torch.nn as nn

transformer_layer = nn.TransformerEncoderLayer(
    d_model = 32,        # input dimension
    nhead = 8,           # number of attention heads in multi-head self-attention layers
    batch_first = True   
)

x = torch.rand(2, 16, 32)
output = transformer_layer(x)
print(output.shape)     # torch.Size([2, 16, 32])
~~~

## Transformer的应用

接下来, 我们简述一些 `Transformer` 架构的应用. 

### Vision Transformer

通过上文的描述我们已经知道, `Transformer` 是一种端到端的, 解决 `NLP` 任务的模型架构, 采用多头自注意力机制使得模型能够并行化地快速训练并获得对全局信息的提取能力. 通常, `Transformer` 需要在大型文本语料库中进行 **预训练**, 然后在垂直领域进行微调从而在特定任务上达到更好的性能. 

`Vision Transformer` 是 `Transformer` 模型架构在计算机视觉领域中的变体, 在尽可能避免对原有模型架构的改动的前提下, 成功的将其迁移至了计算机视觉领域, 证明了 `Transformer` 架构在图形领域同样具有取代传统的 `CNN` 架构用于图像分类和检测任务的能力. 

![20240131063637](https://cdn.jsdelivr.net/gh/KirisameR/KirisameR.github.io/img/blogpost_images/20240131063637.png)

作者将图片拆分为不重叠且大小固定的 `patch` 序列, 经过向量化后作为 **线性序列** 输入传入 `Transformer` 中, 类比 `NLP` 领域中的词组序列输入.

论文中设计的 `Vision Trasformer` 图片分类器执行的操作基本如下:

1. 将图片分割为大小相同 (如 $16 \times 16$) 且不重叠的 `Patches`, 然后将每个 `Patch` 展开为 **一维向量**.
2. 一般地, 由于展开后的 `Patch` 维度较大, 还需通过线性投影层对 `Patch` 做进一步的 `Token` 化, 进行维度压缩和特征变换得到 `Patch Embeddings`.
3. 在所得的 `Patch Embeddings` 前加入作者额外引入的 **可学习的 `Class Token`**, 方便后续分类.
4. 在加入了 `Class Token` 的 `Patch Embeddings` 后面再加上 **位置编码** (`Position Embedding`), 然后将其作为多个 **串行** 的 `Transformer Encoders` 中进行 **全局注意力计算和特征提取**: `Transformer Encoder` 中的多头自注意力模块负责提取 `Patch` 内或 `Patch` 序列之间的特征, 在多头自注意力模块后的全连接层模块负责对所提取的特征进行线性变换. 
5. 提取出串行的 `Transformer Encoders` 中最后一个的输出序列中对应 `Class Token` 的部分, 以其作为编码器串最终的信息提取结果, 传入作为分类器的全连接层 (`MLP`) 中, 得出最终的分类结果. 

### 预训练语言模型和大型语言模型

预训练语言模型是 `Transformer` 架构的重要应用之一. 预训练语言模型通过使用大量的语料数据对模型进行 **无监督或弱监督学习** (`Unsupervised` or `Weak-supervised`), 使参数规模庞大的语言模型具备句法, 语法规律等足够多的语言知识. 然后在下游任务中再对模型进行相应的 `finetune`. 

注: 
> 监督学习基于成对的 **输入-输出** 数据, 学习目标是让模型能够正确拟合输入和输出之间的映射关系. 监督学习需要经过标注的训练数据, 有明确的模型训练目标, 并通过 **目标函数** 判断模型训练的效果. 
>
> 无监督学习基于 **未标注** 的数据, 没有明确的学习目标, 依赖模型自身对训练数据的内在结构和关系进行挖掘. 无监督学习不依赖经过标注的训练数据, 没有明确的模型训练目标, 难以评估模型训练的效果. 
>
>半监督学习介于监督学习和无监督学习, 使用 **大量的未标注数据** 和 **少量的标注数据** 对模型进行训练, 目的是在有限的标注数据引导下最大程度地利用未标注数据提高模型的泛化能力. 


在语言模型的无监督预训练目标中, 最为成功的是 **自回归** 和 **自编码** 语言建模. 在 `Transformer` 架构中, 其编码器和解码器就分别属于自编码和自回归模型.

因此, 预训练语言模型主要包括 **自编码模型 (又称`AE`模型, `Autoencoder Language Model`)** 和 **自回归模型 (也称 `AR`模型, `Autoregressive Language Model`)**:

![20240201213351](https://cdn.jsdelivr.net/gh/KirisameR/KirisameR.github.io/img/blogpost_images/20240201213351.png)

上图中左侧蓝色的模型均为 `Transformer Encoder`, 也就是自编码模型; 右侧红色的则为 `Transformer Decoder`, 即自回归模型. 中间的灰色模型为结合了 `Transformer Encoder` 和 `Transformer Decoder` 的 `Seq2Seq` 模型. 

#### 自编码语言模型

我们首先解释 **自编码器** 的概念. 自编码器 (`Auto Encoder`) 是用于学习 **数据的有效特征表示** 的神经网络, 由负责 **将输入数据映射到低维特征空间** 的编码器和 **将低维特征映射回原始空间** 的解码器组成. 

自编码模型的主要代表是 `Google` 推出的 `BERT`. 它们的能力可以被简要的解释为 “做完形填空”: 从被掩膜部份掩盖的输入中, 预测被掩盖的部份, 重建原始数据. 

`BERT` 及其变体, 包括 `RoBERTa, ALBERT` 等 `AE` 模型包含 **双向** `Transformer` 结构, 能够生成包含语言理解的文本表示 (`Text Representation`), 擅长自然语言理解任务 (`NLU`, 即 `Natural Language Understanding`), 常被用于生成 **句子的上下文表示** 和 **词/句子嵌入向量**, 但 **不能被直接用于文本生成**. 

自编码语言模型 (`AE`) 的优缺点总结如下:

优点: 自编码语言模型对序列中任意元素的表示可以 **涵盖这个元素前后两方向的完整上下文**, 擅长语言理解任务.

缺点: 

1. 自编码语言模型存在 **输入噪声** 的问题: 由于 `BERT` 类模型在预训练过程中必须使用在下游 `finetune` 任务中永不会出现的掩膜符号 `[MASK]` 对输入进行遮盖, 因此会产生 **预训练-微调差异**, 而自回归语言模型由于不依赖被掩膜遮盖的输入训练, 因此不存在这种问题. 
2. `BERT` 假设, 在给定 `unmasked tokens` 时, 所有待预测 (`masked`) 的 `tokens` 都是相互独立的, 而这一点是与事实不符的.

下面我们简述典型的自编码模型: `BERT`.

##### 介绍



##### 模型架构

##### 实现

#### 自回归语言模型

自回归语言模型 (`AR` 模型) 的代表为 `GPT`, 本质是 `Transformer` 架构中的解码器, 其学习模式为: 给定一系列按照时序排序的输入序列, 每一次给定一个输入, 预测下一个 `timestep` 的输出, 并将其输出作为预测下下个 `timestep` 的值时的输入. 自回归语言模型因其特性常用于 **生成式任务**, 如自然语言生成 (`Natural Language Generation`) 领域中的文本摘要, 文本翻译, 对话问答等一系列任务. 

自回归模型利用文本的上下文, 通过估计训练学习到的文本语料库中文本的 **概率分布**, 对当前输入序列的下一个词进行预测:

给定文本序列 

$$T = x_1, x_2, \cdots, x_{L_T}.$$

则该文本序列的似然序列可以被分解为 **前向连乘** (由前向后预测):

$$p(T) = \prod_{i=1}^{L_T} p(x_i \vert x_{<i})$$

或者 **后向连乘** (由后向前预测):

$$p(T) = \prod_{i={L_T}}^{1} p(x_i \vert x_{>i})$$

由于 `AR` 模型仅能 **单向编码**, 因此它在对 **双向上下文信息** 进行建模时效果不佳. 

自回归语言模型 (`AR`) 的优缺点总结如下:

优点: 擅长 `NLG` 任务, 训练目标较为简单: 预测语料库中的下一个 `token`, 训练数据方便取得.

缺点: 无法同时利用和建模双向的上下文信息, 无法完全捕捉文本中 `tokens` 的内在联系. 

下面我们简述典型的自回归模型: `GPT`.

##### 介绍

##### 模型架构

##### 实现

#### `Encoder-Decoder` 模型

`Encoder-Decoder` 模型同时使用编码和解码器, 将所有的下游任务视为 **`Seq2Seq`** 任务: 文本-文本, 文本-图像, 图像-文本, 文本-序列等. 

下面我们简述典型的 `Encoder-Decoder` 模型: `T5`.

##### 介绍

##### 模型架构


### 多模态视觉语言模型

## 参考



https://nthu-datalab.github.io/ml/labs/12-1_Seq2Seq-Learning_Neural-Machine-Translation/12-1_Seq2Seq-Learning_Neural-Machine-Translation.html

https://github.com/gursi26/seq2seq-attention

https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html

https://medium.com/@a.akhterov/step-by-step-understanding-of-the-bahdanau-attention-mechanism-with-code-5c62e280ca13

https://zhuanlan.zhihu.com/p/136559171

https://mylens.ai/lens/the-evolution-of-attention-mechanism-in-artificial-intelligence-xsq3vf/

https://zhuanlan.zhihu.com/p/631398525?utm_id=0&wd=&eqid=be38b6e2000b4b43000000036545ba6f

https://www.elecfans.com/d/883694.html

https://arxiv.org/abs/1409.3215

https://arxiv.org/abs/1406.1078

https://arxiv.org/abs/1706.03762

https://github.com/EvilPsyCHo/Attention-PyTorch

https://blog.csdn.net/weixin_53598445/article/details/125009686

https://aclanthology.org/2020.acl-main.312/

https://blog.csdn.net/beilizhang/article/details/115282604

https://blog.csdn.net/weixin_43334693/article/details/130189238?spm=1001.2014.3001.5502

https://blog.csdn.net/hxxjxw/article/details/120134012

https://zhuanlan.zhihu.com/p/481559049

https://zhuanlan.zhihu.com/p/459828118

https://zhuanlan.zhihu.com/p/348593638

https://blog.csdn.net/weixin_43334693/article/details/130208816

https://twitter.com/lvwerra/status/1507035164832055318

https://zhuanlan.zhihu.com/p/625714067?utm_id=0

https://huggingface.co/docs/transformers/model_summary

https://aman.ai/primers/ai/autoregressive-vs-autoencoder-models/