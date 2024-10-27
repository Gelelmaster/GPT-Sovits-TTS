from datetime import datetime
import os
import sys
import json
import requests
import numpy as np
import wave
import io
import gradio as gr
import logging
from string import Template
from importlib import import_module
from functools import partial
from time import time as ttime

# 导入路径和设置日志
now_dir = os.getcwd()
sys.path.insert(0, now_dir)

# 设置日志级别
logging_levels = [
    "markdown_it", "urllib3", "httpcore", "httpx",
    "asyncio", "charset_normalizer", "torchaudio._extension"
]
for logger_name in logging_levels:
    logging.getLogger(logger_name).setLevel(logging.ERROR)

from Synthesizers.base import Base_TTS_Synthesizer, Base_TTS_Task, get_wave_header_chunk
from src.common_config_manager import app_config, __version__
import soundfile as sf
import tools.i18n.i18n as i18n_module

# 初始化国际化
i18n = i18n_module.I18nAuto(
    language=app_config.locale,
    locale_path=f"Synthesizers/{app_config.synthesizer}/configs/i18n/locale"
)

# 动态导入合成器模块
synthesizer_module = import_module(f"Synthesizers.{app_config.synthesizer}")
TTS_Synthesizer = synthesizer_module.TTS_Synthesizer
TTS_Task = synthesizer_module.TTS_Task

# 创建合成器实例
tts_synthesizer: Base_TTS_Synthesizer = TTS_Synthesizer(debug_mode=True)

all_gradio_components = {}
characters_and_emotions_dict = {}

def load_character_emotions(character_name: str) -> gr.Dropdown:
    emotion_options = characters_and_emotions_dict.get(character_name, ["default"])
    return gr.Dropdown(emotion_options, value="default")

'''文本转语音（TTS）合成并返回音频数据'''
def get_audio(*data, streaming=False):
    data_dict = dict(zip(all_gradio_components.keys(), data))
    '''    
    all_gradio_components.keys()获取 all_gradio_components 字典中的所有键。这个字典通常包含了与 Gradio 界面组件相关的名称（如文本框、下拉列表等）。
    
    data 是一个可变参数（使用 *data 表示），它包含了从 Gradio 界面传入的值，这些值与 all_gradio_components 中的键相对应。
    '''
    data_dict["stream"] = streaming
    
    if not data_dict.get("text"):
        gr.Warning(i18n("文本不能为空"))
        return None, None

    try:
        task: Base_TTS_Task = tts_synthesizer.params_parser(data_dict)
        t2 = ttime()
        
        if not streaming:
            if app_config.synthesizer == "remote":
                save_path = tts_synthesizer.generate(task, return_type="filepath")
                yield save_path
            else:
                yield next(tts_synthesizer.generate(task, return_type="numpy"))
        else:
            gen = tts_synthesizer.generate(task, return_type="numpy")
            sample_rate = 32000 if task.sample_rate in [None, 0] else task.sample_rate
            yield get_wave_header_chunk(sample_rate=sample_rate)
            for chunk in gen:
                yield chunk
        
    except Exception as e:
        gr.Warning(f"Error: {e}")

get_streaming_audio = partial(get_audio, streaming=True)

def stopAudioPlay():
    return

def get_characters_and_emotions() -> dict:
    global characters_and_emotions_dict
    if not characters_and_emotions_dict:
        characters_and_emotions_dict = tts_synthesizer.get_characters()
        print(characters_and_emotions_dict)
    return characters_and_emotions_dict

def change_character_list(character="", emotion="default") -> tuple:
    characters_and_emotions = get_characters_and_emotions()
    character_names = list(characters_and_emotions.keys())
    character_name_value = character if character in character_names else character_names[0] if character_names else ""
    emotions = characters_and_emotions.get(character_name_value, ["default"])
    
    return (
        gr.Dropdown(character_names, value=character_name_value, label=i18n("选择角色")),
        gr.Dropdown(emotions, value=emotion, label=i18n("情感列表"), interactive=True),
        characters_and_emotions
    )

def cut_sentence_multilang(text: str, max_length: int = 30) -> tuple:
    if max_length == -1:
        return text, ""

    word_count = 0
    in_word = False
    
    for index, char in enumerate(text):
        if char.isspace():
            in_word = False
        elif char.isascii() and not in_word:
            word_count += 1
            in_word = True
        elif not char.isascii():
            word_count += 1
        
        if word_count > max_length:
            return text[:index], text[index:]
    
    return text, ""

default_text = i18n("1.本模型不得创作任何违反法律法规的内容，不得用于任何商业用途，不得二次配布。如有滥用行为，该模型将永久停止公开！2. 发视频请注明模型训练者、整合包作者和数据集整理者。")

information = ""
try:
    with open("Information.md", "r", encoding="utf-8") as f:
        information = f.read()
except FileNotFoundError:
    pass

max_text_length = app_config.max_text_length if hasattr(app_config, 'max_text_length') else -1
url_setting = tts_synthesizer.ui_config.get("url_settings", [])
params_config = TTS_Task().params_config
has_character_param = "character" in params_config

with gr.Blocks() as app:
    gr.Markdown(information)
    with gr.Row():
        with gr.Column(scale=2):
            with gr.Tabs():
                max_text_length_tip = "" if max_text_length == -1 else f"( {i18n('最大允许长度')} : {max_text_length} ) "
                text = gr.Textbox(value=default_text, label=i18n("输入文本") + max_text_length_tip, interactive=True, lines=9.5)
                text.blur(lambda x: gr.update(value=cut_sentence_multilang(x, max_length=max_text_length)[0]), [text], [text])
                all_gradio_components["text"] = text
                
        with gr.Column(scale=2):
            with gr.Tabs():
                with gr.Tab(label=i18n("角色选项"), visible=has_character_param):
                    with gr.Group():
                        character, emotion, characters_and_emotions_ = change_character_list()
                        characters_and_emotions = gr.State(characters_and_emotions_)
                        scan_character_list = gr.Button(i18n("扫描人物列表"), variant="secondary")
                    
                    all_gradio_components["character"] = character
                    all_gradio_components["emotion"] = emotion

                    character.change(
                        load_character_emotions,
                        inputs=[character],
                        outputs=[emotion],
                    )

                    scan_character_list.click(
                        change_character_list,
                        inputs=[character, emotion],
                        outputs=[character, emotion, characters_and_emotions],
                    )
                    
        if url_setting:
            with gr.Column(scale=2):
                with gr.Tabs():
                    
                        with gr.Tab(label=i18n("URL设置")):
                            url_setting_tab = GradioTabBuilder(url_setting, params_config)
                            url_setting_components = url_setting_tab.build()
                            all_gradio_components.update(url_setting_components)

    with gr.Tabs():
        with gr.Tab(label=i18n("请求完整音频")):
            with gr.Row():
                get_full_audio_button = gr.Button(i18n("生成音频"), variant="primary")
                full_audio = gr.Audio(
                    None, label=i18n("音频输出"), type="filepath", streaming=False
                )
                get_full_audio_button.click(lambda: gr.update(interactive=False), None, [get_full_audio_button]).then(
                    get_audio,
                    inputs=[value for key, value in all_gradio_components.items()],
                    outputs=[full_audio],
                ).then(lambda: gr.update(interactive=True), None, [get_full_audio_button])
        with gr.Tab(label=i18n("流式音频")):
            with gr.Row():
                get_streaming_audio_button = gr.Button(i18n("生成流式音频"), variant="primary")
                streaming_audio = gr.Audio(
                    None, label=i18n("音频输出"), type="filepath", streaming=True, autoplay=True
                )
                get_streaming_audio_button.click(lambda: gr.update(interactive=False), None, [get_streaming_audio_button]).then(
                    get_streaming_audio,
                    inputs=[value for key, value in all_gradio_components.items()],
                    outputs=[streaming_audio],
                ).then(lambda: gr.update(interactive=True), None, [get_streaming_audio_button])


if app_config.also_enable_api:
    import uvicorn
    from pure_api import tts, character_list, set_tts_synthesizer
    from fastapi import FastAPI
    from fastapi.middleware.cors import CORSMiddleware
    from src.api_utils import get_gradio_frp, get_localhost_ipv4_address

    set_tts_synthesizer(tts_synthesizer)
    fastapi_app: FastAPI = app.app
    fastapi_app.add_api_route("/tts", tts, methods=["POST", "GET"])
    fastapi_app.add_api_route("/character_list", character_list, methods=["GET"])
    
    fastapi_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    local_link = f"http://127.0.0.1:{app_config.server_port}"
    link = local_link
    if app_config.is_share:
        share_url = get_gradio_frp(app_config.server_name, app_config.server_port, app.share_token)
        print("This share link expires in 72 hours.")
        print(f"Share URL: {share_url}")
        link = share_url
        
    if app_config.inbrowser:
        import webbrowser
        webbrowser.open(link)

    ipv4_address = get_localhost_ipv4_address(app_config.server_name)
    ipv4_link = f"http://{ipv4_address}:{app_config.server_port}"
    print(f"INFO:     Local Network URL: {ipv4_link}")
    
    fastapi_app = gr.mount_gradio_app(fastapi_app, app, path="/")
    uvicorn.run(fastapi_app, host=app_config.server_name, port=app_config.server_port)
else:
    app.queue().launch(share=app_config.is_share, inbrowser=app_config.inbrowser, server_name=app_config.server_name, server_port=app_config.server_port)
