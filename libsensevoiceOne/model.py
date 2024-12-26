# =========================================
# -*- coding: utf-8 -*-
# Project     : SenseVoiceOne
# Module      : SenseVoiceOne.py
# Author      : KyleWang[kylewang1977@gmail.com]
# Time        : 2024-12-23 13:05
# Version     : 1.0.0
# Last Updated: 
# Description : SenseVoice Onnx模型的使用类定义。重新封装了接口，简化了使用方式。
#               如OpenAi-whisper一样，通过调用load_model、transcribe两步完成调用。
# =========================================
import os
import re
import time
import logging
import soundfile
import numpy as np
import librosa
from typing import Union, Tuple
from libsensevoiceOne.onnx.sense_voice_ort_session import SenseVoiceInferenceSession
from libsensevoiceOne.utils.frontend import WavFrontend
from libsensevoiceOne.utils.fsmn_vad import FSMNVad

languages = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}

class SenseVoiceOne(object):
    """
    This is the main class used for ASR.
    """
    model = None; front = None; isVad = False; vad = None; isInit = False

    def __init__(self, 
        senseVoice_model_file:str = None, 
        senseVoice_model_dir:os.PathLike = "./resources/SenseVoice", 
        embedding_model_file:str = "embedding.npy", 
        bpe_model_file:str = "chn_jpn_yue_eng_ko_spectok.bpe.model",
        device:int = -1, n_threads:int = 4,
        front_dir:str = "./resources/front", cmvn_file:str = "am.mvn",
        is_vad:bool=True, 
        vad_dir:os.PathLike = "./resources/vad",
        )-> None:
        """
        Objects init, load models for sensevoice-onnx、front、vad.
        """
        if  senseVoice_model_file is not None:
            self.load_model(senseVoice_model_file, senseVoice_model_dir, 
                            embedding_model_file, bpe_model_file, 
                            device, n_threads,
                            front_dir, cmvn_file,
                            is_vad, vad_dir)


    def load_model(self, 
        senseVoice_model_file:str = "sense-voice-encoder-int8.onnx", 
        senseVoice_model_dir:os.PathLike = "./resources/SenseVoice", 
        embedding_model_file:str = "embedding.npy", 
        bpe_model_file:str = "chn_jpn_yue_eng_ko_spectok.bpe.model",
        device:int = -1, n_threads:int = 4,
        front_dir:str = "./resources/front", cmvn_file:str = "am.mvn",
        is_vad:bool=True, 
        vad_dir:os.PathLike = "./resources/vad",
        )-> None:
        '''
        Objects init, load models for sensevoice-onnx、front、vad.

        By default, it will load all the needs file from ./resources folder. 
        Or you can change them according to your needs.

        Parameters
        ----------
        senseVoice_model_dir : os.PathLike
            The folder contain the SenseVoice model files.
        senseVoice_model_file : str
            File name of SenseVoice Onnx model.
        device : int
            -1: CPU.  or GPU.
        n_threads : int
            threads number.
        is_vad : bool
            whether use the vad model to detect human voice and process the human voice only.
        vad_dir : os.PathLike
            folder of fsmn_vad model files. If is_vad==False, vad_dia could be None.
        '''
        if self.isInit:
            raise RuntimeError("Reload the model.")
        
        self.__load_ss_model(
            senseVoice_model_dir, senseVoice_model_file, 
            embedding_model_file, bpe_model_file, 
            device, n_threads
            )
        
        cmvn_file = os.path.join(front_dir, cmvn_file)
        if not os.path.exists(cmvn_file):
            raise FileNotFoundError(f"cmvn_file {cmvn_file} 不存在！")
        self.front = WavFrontend(cmvn_file)
        logging.debug("WavFrontend ready")
        
        # FSMN Vad 即基于前馈序列记忆网络（Feedforward Sequential Memory Network，FSMN）
        #          的语音活动检测（Voice Activity Detection，VAD）
        self.isVad = is_vad
        if self.isVad:
            self.vad = FSMNVad(vad_dir)
            logging.info("启用VAD. FSMNVad ready")

        self.isInit = True


    def __load_ss_model(self, 
        model_dir, model_file, 
        embedding_model_file, bpe_model_file, 
        device, n_threads
        )->None:
        '''加载SenseVoice模型'''

        model_file = os.path.join(model_dir, model_file)
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"模型文件 {model_file} 不存在！")
        
        embedding_model_file = os.path.join(model_dir, embedding_model_file)
        if not os.path.exists(embedding_model_file):
            raise FileNotFoundError(f"embedding_model_file {embedding_model_file} 不存在！")
        
        bpe_model_file = os.path.join(model_dir, bpe_model_file)
        if not os.path.exists(bpe_model_file):
            raise FileNotFoundError(f"bpe_model_file {bpe_model_file} 不存在！")
       
        self.model = SenseVoiceInferenceSession(
            embedding_model_file,
            model_file,
            bpe_model_file,
            device_id=device,
            intra_op_num_threads=n_threads,)
        logging.debug(f"SenseVoiceInferenceSession ready")


    def transcribe(
        self, 
        audio: Union[os.PathLike, np.ndarray], 
        language:str="auto", 
        use_itn:bool=True,
        use_vad:bool=True,
        ForceMono:bool=True
        )->dict:
        '''Transcribe an audio file using Whisper. 音频文件的加载、处理、转文字.
        
        By default, it will load all the needs file from ./resources folder. 
        Or you can change them according to your needs.

        Parameters
        ----------
        audio:os.PathLike
            The audio file path or ndarry from audion wareform. For the input for VAD&ASR model.
            Data rerequirment:
              sample_rate:16k; dtype: float32; channels=1; value:[-1, 1]
        language : str
            "auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13
        use_itn : bool
            ITN(Inverse Text Normalization) 指的是逆文本规范化。影响到数字、缩写、日期、标点符号...
        use_vad : bool
            True: use Vad if the self.isVad==True(from model init), otherwise not use.
            False: not use the Vad. No matter what had been set in model init.
        ForceMono : bool
            是否强制单声道。如果是, 则对双声道数据进行简单平均, 单声道数据保持不变。

        Returns
        -------
        A dictionary containing the resulting text ("text") and segment-level details ("segments"), and
        the spoken language ("language"), which is detected when `decode_options["language"]` is None.
        result: {"isVad":False, "channels":1, "language":"auto", 
                 "segments":[ { "channel":0, 
                                "parts":[{"times":[0.00, 1.96], "tags":{}, "text":{...}},
                                         {"times":[3.01, 8.96], "tags":{}, "text":{...}}]
                              },
                              { "channel":1, 
                                "parts":[ { "times":[1.01, 3.96], "tags":{}, "text":{...} },
                                          { "times":[6.06, 9.99], "tags":{}, "text":{...} }]
                              }
                            ]
                }
        '''
        waveform = self.load_audio(audio, ForceMono)
        result = {"isVad":False, "channels":1, "language":"auto", "segments":[]}
        result["language"] = language
        result["channels"] = waveform.shape[0]
        # segment = {"time":None, "tags":None, "text":None}
        if self.isVad and use_vad:
            result["isVad"] = True
            for i in range(waveform.shape[0]):
                channel_data = waveform[i]
                segments = self.vad.segments_offline(channel_data)
                segmentsRes = {"channel":i, "parts":[]}
                for part in segments:
                    audio_feats = self.front.get_features(channel_data[part[0]*16 : part[1]*16])
                    asr_result = self.model.inference(audio_feats[None, ...],
                                    language=languages[language], use_itn=use_itn,)
                    res = self.res_re(asr_result)
                    res['time'] = [part[0]/1000, part[1]/1000]
                    logging.debug(f"[{res['time'][0]}s-{res['time'][1]}s] tags: {res['tags']}")
                    logging.info(f"[{res['time'][0]}s-{res['time'][1]}s] text: {res['text']}")
                    segmentsRes["parts"].append(res)
                self.vad.vad.all_reset_detection()
                result["segments"].append(segmentsRes)
        else:
            segmentsRes = {"channel":0, "parts":[]}
            channel_data = waveform[0]
            audio_feats = self.front.get_features(channel_data)
            asr_result = self.model.inference(
                audio_feats[None, ...],
                language=languages[language],
                use_itn=use_itn,)
            res = self.res_re(asr_result)
            res['time'] = [0, round(len(channel_data)/16000, 2)]
            segmentsRes["parts"].append(res)
            logging.debug(f"tags: {res['tags']}")
            logging.info(f"text: {res['text']}")
            result["segments"].append(segmentsRes)
        return result

    def load_audio(self, 
        audio: Union[os.PathLike, np.ndarray], 
        isMone:bool
        )-> np.ndarray:
        """
        音频文件加载或数据处理,
        结果是ndarry, shape:(channels x frames), 均一化[-1,1], 数据类型是"float32"
        """
        if isinstance(audio, os.PathLike):
            waveform, sr = soundfile.read(audio, dtype="float32", always_2d=True)
            waveform = waveform.T
            # assert(sr==16000), f"只支持16000Hz采样频率, 实际是:{sr}Hz"
            if waveform.shape[0] == 2 and isMone:
                waveform = waveform.mean(axis=0).reshape(1, -1)
            if sr != 16000:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            audioArray = np.ascontiguousarray(waveform)
        elif isinstance(audio, np.ndarray):
            if audio.ndim==1:
                audio = audio.reshape(1, -1)
            if audio.ndim!=2:
                raise RuntimeError(f"音频文件加载错误的数据. 只能是1、2维数组. 实际是:{audio.ndim}")
            if audio.shape[0]>audio.shape[1]:
                audio = audio.T
            if audio.shape[0] == 2 and isMone:
                audio = audio.mean(axis=0).reshape(1, -1)
            audioArray = audio
        return audioArray

    def res_re(self, result:str)->dict:
        resDict = {"time":None, "tags":None, "text":None}
        # 使用正则表达式匹配标签和文本内容
        pattern = r"<\|([^|]+)\|>"  # 匹配 <|...|> 中的内容
        tags = re.findall(pattern, result)
        # 提取文本部分
        text = re.sub(pattern, '', result).strip()
        resDict["tags"] = tags
        resDict["text"] = text
        return resDict

