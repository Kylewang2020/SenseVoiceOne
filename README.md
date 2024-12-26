# SenseVoiceOne

## Introduction

- 是针对SenseVoice-small 之 onnx 模型的 python 类定义和实现。
- 简单、易用, 模拟OpenAi-whisper的调用方式。
- 是使用和理解[SenseVoice-python](https://github.com/lovemefan/SenseVoice-python)项目的一个记录和测试。
- 重新封装了调用接口, 方便作为一个新的引用, 使用在其他项目中。

### SenseVoice

- [SenseVoice](https://github.com/FunAudioLLM/SenseVoice) 是具有音频理解能力的音频基础模型, 包括:
  - 语音识别(ASR)
  - 语种识别(LID)
  - 语音情感识别(SER)
  - 声学事件分类(AEC)
  - 声学事件检测(AED)
- 当前SenseVoice-small支持中、粤、英、日、韩语的多语言语音识别, 情感识别和事件检测能力, 具有极低的推理延迟。

## 项目的目标

- 简单、易用
  - 只要引用一个类定义, 实例化一个对象, 就可以语音识别、语音检测
- 低硬件、算力需求；适用于边沿设备
  - 树莓派5B上测试, 低内存要求、低负载

## 项目说明&笔记

### 模型相关文件

- [模型相关文件说明](./doc/model_files.md)

### ASR 前端结构

- [ASR 前端结构 的说明](./doc/frontend.md)

## Install and models get

...

## Usages

``` python
# 引用 SenseVoice_One
from libsensevoiceOne.SenseVoiceOne import SenseVoice_One

# 初始化参数[模型目录、文件...]
senseVoice_model_dir = "./resources/SenseVoice"
device = -1
num_threads = 4

audio_path = "./data/chinese01.wav"
language = "auto"

ssOnnx = SenseVoiceOne()
ssOnnx.load_model(senseVoice_model_file=senseVoice_model_file)
res = ssOnnx.transcribe(audio_path, language=language, use_itn=True, 
                        use_vad=True, ForceMono=True)

print(res)
```

得到的结果:

``` cmd
{'isVad': True, 'channels': 1, 'language': 'auto', 
 'segments': 
  [{'channel': 0, 'parts': 
      [
        {'time': [0.0, 1.83], 'tags': ['zh', 'NEUTRAL', 'Speech', 'withitn'], 'text': '基本上就是个假消息。'}, 
        {'time': [2.11, 9.58], 'tags': ['zh', 'NEUTRAL', 'Speech', 'withitn'], 'text': '你明白我意思吗？所以呢我呢不认识这个书记, 我也不认识什么老领导, 我就认识门口这个发报纸那大爷。'}
      ]
    }
  ]
}
```

## 感谢以下项目

1. 本项目借用并模仿了[SenseVoice-python](https://github.com/lovemefan/SenseVoice-python)的库文件和代码
2. 参考了[FunAudioLLM/SenseVoice](https://github.com/FunAudioLLM/SenseVoice)
3. 本项目参考并借用 kaldi-native-fbank中的fbank特征提取算法。 FunASR 中的lrf + cmvn 算法
