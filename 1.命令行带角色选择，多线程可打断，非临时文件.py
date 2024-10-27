import io
import numpy as np
import logging
import soundfile as sf
from time import time as ttime
from pygame import mixer
from Synthesizers.base import Base_TTS_Synthesizer, Base_TTS_Task
from importlib import import_module
from src.common_config_manager import app_config
import torch
import asyncio
import threading

# 初始化日志
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# 设定日志级别
for logger_name in [
    "markdown_it", "urllib3", "httpcore", "httpx",
    "asyncio", "charset_normalizer", "torchaudio._extension"
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# 检查CUDA可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"\nUsing {'GPU' if torch.cuda.is_available() else 'CPU'} for inference.")

# 动态导入语音合成器模块
synthesizer_name = app_config.synthesizer
synthesizer_module = import_module(f"Synthesizers.{synthesizer_name}")
TTS_Synthesizer = synthesizer_module.TTS_Synthesizer
TTS_Task = synthesizer_module.TTS_Task

# 创建合成器实例
tts_synthesizer: Base_TTS_Synthesizer = TTS_Synthesizer(debug_mode=True)

# 初始化音频播放库
mixer.init()
characters_and_emotions_dict = {}

# 当前音频播放对象和锁
current_sound = None
sound_lock = threading.Lock()

def get_characters_and_emotions():
    """获取角色和情感信息"""
    global characters_and_emotions_dict
    if not characters_and_emotions_dict:
        characters_and_emotions_dict = tts_synthesizer.get_characters()
        # logger.info(f"\n可用角色及情感：{characters_and_emotions_dict}")
    return characters_and_emotions_dict

async def get_audio(data, streaming=False):
    """生成音频数据"""
    if not data.get("text"):
        raise ValueError("文本不能为空")

    try:
        task: Base_TTS_Task = tts_synthesizer.params_parser(data)

        # 将任务数据移到 GPU（如果支持）
        if hasattr(task, 'to'):
            task = task.to(device)  # 确保任务在正确的设备上
        else:
            logger.error("任务不支持移动到设备")
            
        gen = tts_synthesizer.generate(task, return_type="numpy")

        if not streaming:
            audio_data = next(gen)
            logger.info(f"\n生成的音频数据（元组）: {audio_data}, 长度: {len(audio_data)}")
            return audio_data
        else:
            # 流式音频，逐块返回
            return b''.join(chunk for chunk in gen)

    except Exception as e:
        logger.error(f"\n错误: {e}")
        raise RuntimeError(f"\n错误: {e}")

async def play_audio(audio_data, sample_rate=32000):
    """播放生成的音频数据"""
    global current_sound

    logger.info(f"\n音频数据类型: {type(audio_data)}, 长度: {len(audio_data)}")

    # 如果 audio_data 是元组，提取音频数据和采样率
    if isinstance(audio_data, tuple):
        sample_rate, audio_data = audio_data
        logger.info(f"\n提取音频数据: {audio_data}, 采样率: {sample_rate}")

    audio_data = np.array(audio_data)

    # 确保音频数据为一维数组
    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(-1, 1)
    elif audio_data.ndim > 2:
        raise ValueError("不支持的音频数据维度")

    # logger.info(f"\n处理后的音频数据形状: {audio_data.shape}")

    # 将音频数据保存到内存中并播放
    audio_buffer = io.BytesIO()
    sf.write(audio_buffer, audio_data, sample_rate, format='WAV')
    audio_buffer.seek(0)  # 将缓冲区指针移动到开头

    # 停止上一次播放的音频
    with sound_lock:
        if current_sound is not None:
            current_sound.stop()  # 停止当前音频
        current_sound = mixer.Sound(audio_buffer)  # 创建新的 Sound 对象
        current_sound.play()  # 播放新的音频

    # 等待播放结束
    while mixer.get_busy():
        await asyncio.sleep(0.1)

def play_audio_sync(audio_data, sample_rate=32000):
    """同步播放生成的音频数据"""
    asyncio.run(play_audio(audio_data, sample_rate))

async def text_to_speech(text, character="", emotion="default"):
    """文本转语音流程"""
    data = {"text": text, "character": character, "emotion": emotion}

    logger.info("\n开始生成音频...")
    audio_data = await get_audio(data)
    logger.info("\n生成完成，正在播放音频...")
    # 在新的线程中播放音频
    threading.Thread(target=play_audio_sync, args=(audio_data,)).start()

def input_thread(loop):
    """输入线程函数"""
    asyncio.set_event_loop(loop)
    loop.run_forever()

async def main():
    """主函数，接受用户输入并启动TTS"""
    get_characters_and_emotions()
    
    # 选择角色
    character_names = list(characters_and_emotions_dict.keys())
    logger.info(f"\n可用角色：{character_names}")
    
    character = input("选择角色（按回车键选择默认角色）：")
    if character not in character_names:
        character = character_names[0] if character_names else ""
    
    # 选择情感
    emotion_options = characters_and_emotions_dict.get(character, ["default"])
    logger.info(f"\n{character} 可用情感：{emotion_options}")
    
    emotion = input("选择情感（按回车键选择默认情感）：")
    if emotion not in emotion_options:
        emotion = "default"

    # 进入文本输入循环，不再提示是否继续
    while True:
        text = input("请输入要转换为语音的文本（输入'退出'以结束）：\n")
        
        # 如果输入为退出指令，则结束循环
        if text.lower() == '退出':
            logger.info("\n退出程序。")
            break
        
        # 如果输入为空，提示重新输入
        if not text:
            logger.info("\n输入不能为空，请重新输入。")
            continue
        
        # 生成音频
        await text_to_speech(text, character, emotion)

if __name__ == "__main__":
    # 创建事件循环
    loop = asyncio.new_event_loop()
    
    # 启动输入线程
    thread = threading.Thread(target=input_thread, args=(loop,))
    thread.start()
    
    # 运行主函数
    asyncio.run(main())  
    
    # 停止输入线程
    loop.call_soon_threadsafe(loop.stop)
    thread.join()
