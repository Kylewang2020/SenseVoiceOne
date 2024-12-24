# =========================================
# -*- coding: utf-8 -*-
# Project     : SenseVoiceOne
# Module      : SenseVoiceOne.py
# Author      : KyleWang[kylewang1977@gmail.com]
# Time        : 2024-12-23 13:05
# Version     : 1.0.0
# Last Updated: 
# Description : Brief description
#               
# =========================================
import os
import re
import time
import soundfile

from lib.onnx.sense_voice_ort_session import SenseVoiceInferenceSession
from lib.utils.frontend import WavFrontend
from lib.utils.fsmn_vad import FSMNVad 
from lib.log import *

languages = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}

class SenseVoice_Onnx_One(object):
    """
    This is the main class used for ASR.
    """
    model = None; front = None; isVad = False; vad = None; isInit = False

    def __init__(self, model_file:str = None, device:int = -1, 
        n_threads:int = 4, embedding_model_file:str = "embedding.npy", 
        bpe_model_file:str = "chn_jpn_yue_eng_ko_spectok.bpe.model",
        front_dir:str = "./resources/front", cmvn_file:str = "am.mvn",
        use_vad:bool=True, vad_dir:str = "./resources/vad",)-> None:
        '''对象的初始化, 准备好model、front、vad三个对象.'''
        
        if model_file is not None:
            self.init(model_file, device, n_threads, embedding_model_file, 
                      bpe_model_file, front_dir, cmvn_file, use_vad, vad_dir)
    
    def init(self, model_file:str, device:int = -1, 
        n_threads:int = 4, embedding_model_file:str = "embedding.npy", 
        bpe_model_file:str = "chn_jpn_yue_eng_ko_spectok.bpe.model",
        front_dir:str = "./resources/front", cmvn_file:str = "am.mvn",
        use_vad:bool=True, vad_dir:str = "./resources/vad",)-> None:
        '''对象的初始化, 实际执行.'''
        
        if self.isInit:
            logging.error("重复初始化!")
            return
        
        self.load_ss_model(model_file, embedding_model_file, bpe_model_file, device, n_threads)
        
        cmvn_file = os.path.join(front_dir, cmvn_file)
        if not os.path.exists(cmvn_file):
            raise FileNotFoundError(f"cmvn_file {cmvn_file} 不存在！")
        self.front = WavFrontend(cmvn_file)
        logging.debug("WavFrontend ready")
        
        # FSMN Vad 即基于前馈序列记忆网络（Feedforward Sequential Memory Network，FSMN）
        #          的语音活动检测（Voice Activity Detection，VAD）
        self.isVad = use_vad
        if self.isVad:
            self.vad = FSMNVad(vad_dir)
            logging.debug("启用VAD. FSMNVad ready")

        self.isInit = True
    
    def inference(self, audio:os.PathLike, language:str="auto", use_itn:bool=True)->dict:
        '''音频文件的加载、处理、转文字'''
        # 音频数据处理，结果是ndarry, shape:(x, 1), 均一化[-1,1], 数据类型是"float32"
        waveform, sr = soundfile.read(audio_path, dtype="float32", always_2d=True)
        assert(sr==16000), f"只支持16000Hz采样频率, 实际是:{sr}Hz"
        if waveform.shape[1] == 2:
            waveform = waveform.mean(axis=1).reshape(-1, 1)
        
        channel_data = waveform[:, 0]
        result = {}
        if self.isVad:
            segments = self.vad.segments_offline(channel_data)
            result["isVad"] = True
            result["parts"] = len(segments)
            i = 0
            for part in segments:
                audio_feats = self.front.get_features(channel_data[part[0]*16 : part[1]*16])
                asr_result = self.model.inference(audio_feats[None, ...],
                                 language=languages[language], use_itn=use_itn,)
                res = self.res_re(asr_result)
                res['time'] = [part[0]/1000, part[1]/1000]
                logging.info(f"[{res['time'][0]}s-{res['time'][1]}s] tags: {res['tags']}")
                logging.info(f"[{res['time'][0]}s-{res['time'][1]}s] text: {res['text']}")
                result[str(i)] = res
                i += 1
            self.vad.vad.all_reset_detection()
        else:
            result["isVad"] = False
            result["parts"] = 1
            audio_feats = self.front.get_features(channel_data)
            asr_result = self.model.inference(
                audio_feats[None, ...],
                language=languages[language],
                use_itn=use_itn,)
            res = self.res_re(asr_result)
            res['time'] = [0, 0]
            logging.info(f"tags: {res['tags']}")
            logging.info(f"text: {res['text']}")
            result["0"] = res
        return result

    def load_ss_model(self, model_file, embedding_model_file, 
                      bpe_model_file, device, n_threads)->None:
        '''加载SenseVoice模型'''
        if not os.path.exists(model_file):
            raise FileNotFoundError(f"模型文件 {model_file} 不存在！")
        
        model_dir = os.path.abspath(os.path.dirname(model_file))

        embedding_model_file = os.path.join(model_dir, embedding_model_file)
        if not os.path.exists(embedding_model_file):
            raise FileNotFoundError(f"embedding_model_file {embedding_model_file} 不存在！")
        
        bpe_model_file = os.path.join(model_dir, bpe_model_file)
        if not os.path.exists(bpe_model_file):
            raise FileNotFoundError(f"bpe_model_file {bpe_model_file} 不存在！")
       
        logging.debug("SenseVoiceInferenceSession init")
        self.model = SenseVoiceInferenceSession(
            embedding_model_file,
            model_file,
            bpe_model_file,
            device_id=device,
            intra_op_num_threads=n_threads,)
        logging.debug(f"SenseVoiceInferenceSession ready")

    def res_re(self, result:str)->dict:
        resDict = {}
        # 使用正则表达式匹配标签和文本内容
        pattern = r"<\|([^|]+)\|>"  # 匹配 <|...|> 中的内容
        tags = re.findall(pattern, result)
        # 提取文本部分
        text = re.sub(pattern, '', result).strip()
        resDict["tags"] = tags
        resDict["text"] = text
        return resDict


if __name__ == "__main__":
    # model_path = "./resources/SenseVoice/sense-voice-encoder.onnx"
    model_path = "./resources/SenseVoice/sense-voice-encoder-int8.onnx"
    # model_path = "./resources/SenseVoice/model_quant.onnx"
    # audio_path = "./data/asr_example_zh.wav"
    # audio_path = "./data/chinese01.wav"
    audio_path = "./data/english01.wav"
    # audio_path = "./data/example_zh.mp3"
    device     = -1
    num_threads = 4
    language    = "auto"

    logInit(logLevel=logging.INFO)
    
    ssOnnx = SenseVoice_Onnx_One()
    ssOnnx.init(model_path,use_vad=True)
    for i in range(10):
        res = ssOnnx.inference(audio_path, language=language, use_itn=True)
        print(res)
        time.sleep(1)