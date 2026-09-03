# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "openai-whisper",
#     "sounddevice",
#     "numpy",
#     "pynput",
#     "pyperclip",
#     "torch",
# ]
# ///

import time
import numpy as np
import sounddevice as sd
import pyperclip
import torch
import whisper
from pynput import keyboard

# --- 設定 ---
# 手元にある .pt ファイルのパス
MODEL_PATH = r"D:\offline_models\small.pt"
SAMPLE_RATE = 16000
TRIGGER_KEY = keyboard.Key.f8  # 録音トリガーキー

# CPUでのスレッド競合を防ぎ、推論速度を最適化
torch.set_num_threads(4)

print("OpenAI Whisper モデル読み込み中...")
# device="cpu" を指定して手元の .pt を直接ロード
model = whisper.load_model(MODEL_PATH, device="cpu")
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
    
    # 0.3秒未満の入力は破棄
    if len(audio_data) < SAMPLE_RATE * 0.3:
        return

    # CPU推論向け高速化パラメータ
    result = model.transcribe(
        audio_data,
        language="ja",
        fp16=False,                      # CPU実行時は必須（警告・エラー防止）
        beam_size=1,                     # 探索幅を最小化（Greedy）
        best_of=1,
        temperature=0.0,
        condition_on_previous_text=False # 前の文脈を引きずらず高速化
    )
    
    result_text = result.get("text", "").strip()
    
    if result_text:
        print(f"認識結果: {result_text}")
        # クリップボード経由で自動貼り付け
        pyperclip.copy(result_text)
        ctrl = keyboard.Controller()
        with ctrl.pressed(keyboard.Key.ctrl):
            ctrl.tap('v')

# --- 監視ループ開始 ---
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
