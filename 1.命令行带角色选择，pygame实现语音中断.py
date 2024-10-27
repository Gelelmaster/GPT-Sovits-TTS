import json
import os
import requests
import numpy as np
import wave
import io
import logging
import pygame
import soundfile as sf
from time import time as ttime
from pygame import mixer
from Synthesizers.base import Base_TTS_Synthesizer, Base_TTS_Task
from importlib import import_module
from src.common_config_manager import app_config
import torch
import asyncio
from datetime import datetime

# 设置日志级别
for logger_name in [
    "markdown_it", "urllib3", "httpcore", "httpx",
    "asyncio", "charset_normalizer", "torchaudio._extension"
]:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

# 检查CUDA可用性
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'} for inference.")

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

def get_characters_and_emotions():
    """获取角色和情感信息"""
    global characters_and_emotions_dict
    if not characters_and_emotions_dict:
        characters_and_emotions_dict = tts_synthesizer.get_characters()
        print("可用角色及情感：", characters_and_emotions_dict)
    return characters_and_emotions_dict

async def get_audio(data, streaming=False):
    """生成音频数据"""
    if not data.get("text"):
        raise ValueError("文本不能为空")

    try:
        task: Base_TTS_Task = tts_synthesizer.params_parser(data)

        # 将任务数据移到 GPU（如果支持）
        if hasattr(task, 'to'):
            task = task.to(device)

        gen = tts_synthesizer.generate(task, return_type="numpy")

        if not streaming:
            audio_data = next(gen)
            print(f"生成的音频数据（元组）: {audio_data}, 长度: {len(audio_data)}")
            return audio_data
        else:
            # 流式音频，逐块返回
            return b''.join(chunk for chunk in gen)

    except Exception as e:
        raise RuntimeError(f"错误: {e}")

async def play_audio(audio_data, sample_rate=32000):
    """播放生成的音频数据并实现空格键暂停/继续控制"""
    
    # 检查并处理音频数据格式
    print(f"音频数据类型: {type(audio_data)}, 长度: {len(audio_data)}")

    # 如果是元组，提取采样率和数据
    if isinstance(audio_data, tuple):
        sample_rate, audio_data = audio_data
        print(f"提取音频数据: {audio_data}, 采样率: {sample_rate}")

    # 确保数据为 numpy 数组格式
    audio_data = np.array(audio_data)
    if audio_data.ndim == 1:
        audio_data = audio_data.reshape(-1, 1)
    elif audio_data.ndim > 2:
        raise ValueError("不支持的音频数据维度")

    print(f"处理后的音频数据形状: {audio_data.shape}")

    # 将音频数据写入临时文件
    temp_file = "temp_audio.wav"
    sf.write(temp_file, audio_data, sample_rate)

    # 初始化 Pygame
    pygame.init()
    pygame.mixer.init()
    
    # 加载并播放音频
    pygame.mixer.music.load(temp_file)
    pygame.mixer.music.play()

    # 创建一个控制窗口
    screen = pygame.display.set_mode((400, 150))
    pygame.display.set_caption("音频播放控制")
    
    # 主事件循环
    running = True
    try:
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:  # 空格键暂停/继续
                        if pygame.mixer.music.get_busy():  # 如果音频正在播放
                            pygame.mixer.music.pause()
                        else:
                            pygame.mixer.music.unpause()

            # 监听音频是否在播放
            if not pygame.mixer.music.get_busy():
                running = False

            await asyncio.sleep(0.1)  # 确保异步行为保持一致
            screen.fill((0, 0, 0))
            font = pygame.font.Font(None, 36)
            text = font.render("按空格键暂停/继续", True, (255, 255, 255))
            screen.blit(text, (10, 50))
            pygame.display.flip()

    finally:
        # 清理资源
        pygame.quit()
        if os.path.exists(temp_file):
            os.remove(temp_file)

async def text_to_speech(text, character="", emotion="default"):
    """文本转语音流程"""
    data = {"text": text, "character": character, "emotion": emotion}
    
    print("开始生成音频...")
    audio_data = await get_audio(data)
    
    print("生成完成，正在播放音频...")
    await play_audio(audio_data)

async def main():
    """主函数，接受用户输入并启动TTS"""
    get_characters_and_emotions()
    
    # 选择角色
    character_names = list(characters_and_emotions_dict.keys())
    print("可用角色：", character_names)
    
    character = input("选择角色（按回车键选择默认角色）：")
    if character not in character_names:
        character = character_names[0] if character_names else ""
    
    # 选择情感
    emotion_options = characters_and_emotions_dict.get(character, ["default"])
    print(f"{character} 可用情感：", emotion_options)
    
    emotion = input("选择情感（按回车键选择默认情感）：")
    if emotion not in emotion_options:
        emotion = "default"

    # 进入文本输入循环，不再提示是否继续
    while True:
        text = input("请输入要转换为语音的文本：")
        
        # 如果输入为空，提示重新输入
        if not text:
            print("输入不能为空，请重新输入。")
            continue
        
        # 生成音频
        await text_to_speech(text, character, emotion)

if __name__ == "__main__":
    asyncio.run(main())
