# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai-whisper",
#     "openvino>=2024.0.0",
#     "sounddevice",
#     "numpy",
#     "pynput",
#     "pyperclip",
#     "torch",
# ]
# ///

import os
import time
import numpy as np
import sounddevice as sd
import pyperclip
import torch
import whisper
import openvino as ov
from pynput import keyboard

# --- 設定 ---
PT_MODEL_PATH = r"D:\offline_models\small.pt"               # トークナイザー・設定用の元モデル
OV_ENCODER_PATH = r"./whisper-small-ov/whisper_encoder.xml" # 変換したエンコーダ
SAMPLE_RATE = 16000
TRIGGER_KEY = keyboard.Key.f8

# "GPU" (内蔵 Arc GPU: 推奨・爆速) または "NPU"
DEVICE = "GPU"

print("1. 元の Whisper 構造とトークナイザーを準備中...")
model = whisper.load_model(PT_MODEL_PATH, device="cpu")

print(f"2. 変換した OpenVINO エンコーダを {DEVICE} にコンパイル中...")
core = ov.Core()
ov_model = core.read_model(OV_ENCODER_PATH)
compiled_encoder = core.compile_model(ov_model, device_name=DEVICE)
encoder_output_layer = compiled_encoder.output(0)

# --- 本家 Whisper の Encoder を OpenVINO 実装でハイジャック ---
class OpenVINOEncoderWrapper(torch.nn.Module):
    def __init__(self, compiled_model, original_encoder):
        super().__init__()
        self.compiled_model = compiled_model
        self.conv1 = original_encoder.conv1  # 前処理層のプロパティを維持
        
    def forward(self, mel: torch.Tensor):
        # mel: (batch, n_mels, 3000)
        np_mel = mel.detach().cpu().numpy().astype(np.float32)
        
        # Core Ultra (Arc GPU / NPU) で実行
        result = self.compiled_model([np_mel])[encoder_output_layer]
        
        # PyTorch テンソルに戻してデコーダへ渡す
        return torch.from_numpy(result)

# エンコーダを差し替え
model.encoder = OpenVINOEncoderWrapper(compiled_encoder, model.encoder)

print(f"\n準備完了！ [{DEVICE} 稼働] [F8] キーを押しながら話してください。")

# --- 録音制御 ---
audio_buffer = []
is_recording = False

def on_press(key):
    global is_recording, audio_buffer
    if key == TRIGGER_KEY and not is_recording:
        is_recording = True
        audio_buffer = []
        print("\n● 録音中...", end="", flush=True)

def on_release(key):
    global is_recording, audio_buffer
    if key == TRIGGER_KEY and is_recording:
        is_recording = False
        print(" [推論中...]")
        process_audio()

def audio_callback(indata, frames, time_info, status):
    if is_recording:
        audio_buffer.append(indata.copy())

def process_audio():
    if not audio_buffer:
        return
    
    audio_data = np.concatenate(audio_buffer, axis=0).flatten().astype(np.float32)
    if len(audio_data) < SAMPLE_RATE * 0.3:
        return

    # 通常通り model.transcribe を実行（エンコーダ計算のみ OpenVINO 側で実行される）
    result = model.transcribe(
        audio_data,
        language="ja",
        fp16=False,
        beam_size=1,
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False
    )
    
    result_text = result.get("text", "").strip()
    if result_text:
        print(f"認識結果: {result_text}")
        pyperclip.copy(result_text)
        ctrl = keyboard.Controller()
        with ctrl.pressed(keyboard.Key.ctrl):
            ctrl.tap('v')

with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
