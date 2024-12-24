# 传统的 ASR 前端结构

在自动语音识别（ASR）系统中，前端（frontend）指的是处理原始音频信号并提取特征的初始阶段。提取出的特征随后将传递给后续的模型（如声学模型）进行语音识别。

## ASR 前端的传统结构

1. **预加重（Pre-emphasis）**
   - **目的**：增强音频信号中的高频部分，平衡频谱。
   - **方法**：通过高通滤波器处理原始波形，提升高频部分的能量，这对语音识别至关重要。
   - **公式**：\( y(t) = x(t) - \alpha x(t-1) \)，其中 \( \alpha \) 通常设置为 0.95 到 1.0。

2. **分帧（Framing）**
   - **目的**：将连续的语音信号切分为较小的片段（帧），每个帧代表语音的短时特性。
   - **方法**：将信号切分为重叠的小窗口（通常为 20-40 毫秒）。通常使用 50% 重叠，即每帧的步长为 10-12.5 毫秒。
   - **帧长**：通常选择 20-25 毫秒，重叠部分为 50%。

3. **加窗（Windowing）**
   - **目的**：减少帧两端的信号不连续性，避免频谱泄漏。
   - **方法**：对每个帧应用窗口函数（通常使用汉明窗或汉宁窗），使信号在帧的边界处逐渐趋近于零。
   - **常用窗口函数**：汉明窗（Hamming Window）是最常见的选择，因为它能有效减少频谱泄漏。

4. **傅里叶变换 / 短时傅里叶变换（STFT）**
   - **目的**：将时域信号转换到频域，提取每一帧的频谱信息。
   - **方法**：应用短时傅里叶变换（STFT），计算每帧的幅度谱和相位谱。
   - **输出**：STFT 输出一系列频谱（复数值），表示信号在时间上的频率分布。

5. **梅尔频率倒谱系数（MFCC）/ 梅尔频谱图**
   - **目的**：以接近人类听觉感知的方式表示语音的频率特征。
   - **方法**：
     - **梅尔频率尺度（Mel scale）**：梅尔频率尺度模拟了人类对音调的感知，低频区域的分辨率较高，高频区域的分辨率较低，具有对数特性。
     - **梅尔滤波器组**：通过梅尔滤波器对频谱进行滤波，得到每个梅尔频带的能量。
     - **对数变换**：对梅尔频带的能量值应用对数变换，模仿人耳的感知。
     - **倒谱系数（Cepstral Coefficients）**：对对数梅尔频谱应用离散余弦变换（DCT），得到梅尔频率倒谱系数（MFCC），用于表示语音的关键特征。
     - **替代方案**：有些现代 ASR 系统也直接使用 **梅尔频谱图（Mel-spectrogram）** 作为输入。

6. **特征归一化（Feature Normalization）**
   - **目的**：标准化特征，减少不同语音信号之间的变异性，如说话者音量差异或麦克风质量差异的影响。
   - **方法**：对提取的特征（如 MFCC 或梅尔频谱）进行归一化处理，常见的做法是使每个帧的均值为零、方差为 1，或者应用 **均值方差归一化（Mean-Variance Normalization）**。
   - **替代方法**：可以使用 **全局均值归一化** 或 **局部均值归一化** 来处理特征。

7. **增量特征（Delta Features，可选）**
   - **目的**：捕捉语音的动态特性，如语速变化和发音变化。
   - **方法**：计算 **delta** 特征，即对每个帧的特征（如 MFCC）进行一阶或二阶差分。常见做法是使用一阶差分来表示帧与帧之间的变化。

## 传统 ASR 前端流程示意

>原始音频 → 预加重 → 分帧 → 加窗 → STFT → 梅尔频谱 → MFCC / 梅尔谱图 → 特征归一化 → 可能的 Delta 特征

## 现代 ASR 系统

近年来，深度学习技术被广泛应用于 ASR 系统中，前端结构也有所变化。现代系统可能会直接使用原始的梅尔频谱图或其他形式的频域特征，省略传统的 MFCC 特征提取步骤，尤其是在端到端的深度学习模型中。具体来说，常见的做法是：

- 使用 **卷积神经网络（CNN）** 来直接从梅尔频谱图中提取特征。
- 使用 **LSTM（长短时记忆网络）** 或 **Transformer** 等序列模型来进一步建模语音信号中的时序信息。

## 总结

ASR 前端的目标是从原始音频信号中提取有效的特征，通常通过一系列的信号处理步骤，如预加重、分帧、加窗、傅里叶变换、梅尔频率倒谱系数（MFCC）等，最终将音频信号转换为适合模型处理的特征。随着深度学习的发展，现代 ASR 系统的前端可能会更加简化，直接使用梅尔频谱图或通过卷积网络等深度学习方法进行特征提取。