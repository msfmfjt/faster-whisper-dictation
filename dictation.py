# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "pywhispercpp",
#     "sounddevice",
#     "numpy",
#     "pynput",
#     "pyperclip",
# ]
# ///

import time
import numpy as np
import sounddevice as sd
import pyperclip
from pynput import keyboard
from pywhispercpp.model import Model

# --- 設定 ---
# whisper.cpp形式のモデルバイナリのパス
MODEL_PATH = r"D:\offline_models\ggml-small.bin"
SAMPLE_RATE = 16000
TRIGGER_KEY = keyboard.Key.f8  # 録音トリガーキー

print("whisper.cpp モデル読み込み中...")
# n_threads でCPUのスレッド数（物理コア数推奨）を指定
model = Model(MODEL_PATH, n_threads=4)
print("準備完了！ [F8] キーを押しながら話してください。")

# --- 録音バッファ管理 ---
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

# --- 推論 & テキスト自動ペースト ---
def process_audio():
    if not audio_buffer:
        return
    
    # 録音データを結合して1次元のfloat32配列へ
    audio_data = np.concatenate(audio_buffer, axis=0).flatten().astype(np.float32)
    
    # 0.3秒未満の入力はノイズとして除外
    if len(audio_data) < SAMPLE_RATE * 0.3:
        return

    # pywhispercpp による推論
    # n_processors=1, language='ja' を指定
    segments = model.transcribe(
        audio_data,
        language="ja",
        beam_size=1  # 速度優先のGreedy探索
    )
    
    result_text = "".join([s.text for s in segments]).strip()
    
    if result_text:
        print(f"認識結果: {result_text}")
        # クリップボード経由で即時貼り付け
        pyperclip.copy(result_text)
        ctrl = keyboard.Controller()
        with ctrl.pressed(keyboard.Key.ctrl):
            ctrl.tap('v')

# --- 監視ループ開始 ---
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
