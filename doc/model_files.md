# 模型相关文件

## SenseVoiceSmall 模型 之 文件说明

- `am.mvn` 文件:
  全称: Acoustic Model Mean and Variance Normalization
  - 作用:
    这是一个用于归一化音频特征的文件，通常包含特征的均值 (mean) 和方差 (variance)。
    在模型推理中，输入特征需要归一化，以便与训练时的数据分布一致，
    从而提高模型的稳定性和准确性。
  - 用途:
    在推理阶段，将音频特征（如 FBANK 或 Mel-Spectrogram）通过归一化处理.
- `embedding.npy` 文件:
  全称: Embedding Lookup Table
  - 作用:
    包含了模型中预训练的嵌入向量（embedding vectors），
    这些向量通常用于表示特定的语言、语义类别或其它离散的输入信息。
  - 用途:
    在多语言模型中，可能用于表示不同语言的上下文或特定的语言类别。
    当输入的语言信息（如 language 标签）映射到一个 ID 时，
    这个 ID 会被用作索引，从 embedding.npy 中提取对应的向量，并作为模型的输入。
- `chn_jpn_yue_eng_ko_spectok.bpe.model` 文件:
  - 全称: Byte Pair Encoding Model
  - 作用:
    包含模型的分词器，用于将输入文本（如识别的语音转成的文本）分解为子词或词片段。
  - 用途:
    对于多语言模型，它支持多种语言（如中文、日语、粤语、英语和韩语）的分词。
    在推理阶段，生成的文本结果会被分词器解码为完整的句子，
    或者分词器将文本编码为模型输入格式。
  - 技术背景:
    Byte Pair Encoding (BPE) 是一种常用的子词分词算法，可以有效减少词表大小，同时处理未登录词。

### senseVoiceSmall 模型的推理过程中需要

- **输入归一化**：
  `am.mvn` 用于对音频特征（如 `Mel-Spectrogram`）进行归一化。
- **语言识别辅助**：
  `embedding.npy` 提供语言信息的嵌入向量，用于支持多语言处理。
- **分词和文本解码**：
  `chn_jpn_yue_eng_ko_spectok.bpe.model` 处理输出文本的编码和解码，确保多语言文本结果的正确性。

## FSMN-VAD

FSMN 是一种结合了前馈神经网络和记忆机制的深度学习模型，主要用于语音识别任务。
它通过前馈结构和时间延迟模块，能够有效地建模语音信号中的时序依赖，
适合处理长序列数据。
fsmn-config.yaml 文件的作用:
    文件通常是 FSMN (Frequency Selective Memory Network) 模型的配置文件，
    用于指定模型的超参数、训练和推理过程中所需的设置。
    FSMN 是一种深度学习模型，特别用于语音识别和其他时间序列任务。
    它的优势在于能够有效地处理时间序列数据，同时利用长短期记忆（LSTM）网络的优势进行优化。
    
    在深度学习的语音识别系统中，模型通常需要许多超参数来指导训练和推理过程。
    fsmn-config.yaml 文件作为配置文件，通常用于描述以下内容：
    模型架构参数：
        层数：设置模型中神经网络层的数量，如 FSMN 层数、LSTM 层数等。
        隐藏层维度：定义每个层的输出维度。
        激活函数：指定网络中使用的激活函数（如 ReLU, Sigmoid 等）。
        频率选择记忆层（FSMN）参数：与频率选择性有关的超参数，可能涉及到模型如何在频域中进行卷积处理。
    优化器和学习率参数：
        学习率：设置训练过程中的学习率。
        优化器类型：如 Adam 或 SGD 等。
        权重衰减：防止过拟合的正则化参数。
    数据处理参数：
        数据增强：有关输入音频数据的增强策略（如噪声添加、变速变调等）。
        音频特征：如 Mel 滤波器数量、特征提取的帧长度、帧步长等。
    训练与推理参数：
        批次大小：设置训练时的批次大小（batch size）。
        最大训练轮数：设置模型训练的最大轮数。
        训练/验证集路径：指定用于训练和验证的音频数据集路径。
    保存和加载模型的路径：
        模型保存路径：定义训练完成后保存模型的位置。
        预训练模型路径：如果使用预训练模型进行微调，指定预训练模型的路径。    

'''
'''