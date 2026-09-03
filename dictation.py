import io
import time
import numpy as np
import sounddevice as sd
import pyperclip
from pynput import keyboard
from faster_whisper import WhisperModel

# --- 1. 初期設定 ---
MODEL_PATH = r"D:\offline_models\faster-whisper-small" # 手動ダウンロードしたモデルパス
SAMPLE_RATE = 16000
TRIGGER_KEY = keyboard.Key.f8  # 録音トリガーキー（F8）

print("モデル読み込み中...")
model = WhisperModel(MODEL_PATH, device="cpu", compute_type="int8", cpu_threads=4)
print("準備完了！ [F8] キーを押しながら話してください。")

# --- 2. 録音バッファの管理 ---
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
        print(" [処理中...]")
        process_audio()

def audio_callback(indata, frames, time_info, status):
    if is_recording:
        audio_buffer.append(indata.copy())

# --- 3. 音声認識 & テキスト自動ペースト ---
def process_audio():
    if not audio_buffer:
        return
    
    # 録音データを結合して1次元のfloat32配列へ
    audio_data = np.concatenate(audio_buffer, axis=0).flatten().astype(np.float32)
    
    # 短すぎる入力（0.3秒未満）は誤打としてスキップ
    if len(audio_data) < SAMPLE_RATE * 0.3:
        return

    # 推論実行（CPU向け最速オプション）
    segments, _ = model.transcribe(
        audio_data,
        language="ja",
        beam_size=1,
        without_timestamps=True,
        condition_on_previous_text=False
    )
    
    result_text = "".join([s.text for s in segments]).strip()
    
    if result_text:
        print(f"認識結果: {result_text}")
        # クリップボード経由で即座に貼り付け（日本語の文字化けを防ぐ）
        pyperclip.copy(result_text)
        ctrl_c = keyboard.Controller()
        with ctrl_c.pressed(keyboard.Key.ctrl):
            ctrl_c.tap('v')

# --- 4. メインループ開始 ---
with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, callback=audio_callback):
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()
