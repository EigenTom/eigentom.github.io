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

## 注意力机制的应用

下面, 我们对注意力机制的应用进行简要讨论.

### 自注意力机制 `Self-Attention`



### 多头注意力 `Multi-Head Attention`

### 与RNN的结合
参考本文章节:  "`Bahdanau` 注意力机制-实现".

### Transformer

#### 解释

#### 实现

## Transformer的应用

### Vision Transformer

### 大型语言模型

#### Encoder-Only: BERT

#### Decoder-Only: GPT和GLM

#### Encoder + Decoder: T5

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

