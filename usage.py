import time
import logging
from libsensevoiceOne.model import SenseVoiceOne

if __name__ == "__main__":
    from libsensevoiceOne.log import logInit
    logInit(logLevel=logging.DEBUG)

    senseVoice_model_dir = "./resources/SenseVoice"
    # senseVoice_model_file = "sense-voice-encoder.onnx"
    senseVoice_model_file = "sense-voice-encoder-int8.onnx"
    # senseVoice_model_file = "model_quant.onnx"
    device = -1
    num_threads = 4
    
    # audio_path = "./data/asr_example_zh.wav"
    audio_path = "./data/chinese01.wav"
    # audio_path = "./data/english01.wav"
    # audio_path = "./data/example_zh.mp3"    # 48000Hz 5s

    language = "auto"
    # language = "zh"

    ssOnnx = SenseVoiceOne()
    ssOnnx.load_model(senseVoice_model_file=senseVoice_model_file)
    res = ssOnnx.transcribe(audio_path, language=language, use_itn=True, 
                            use_vad=True, ForceMono=True)
    print(res)