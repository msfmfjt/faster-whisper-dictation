# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "optimum-intel[openvino]",
#     "sounddevice",
#     "numpy",
#     "pynput",
#     "pyperclip",
#     "transformers",
# ]
# ///

import time
import numpy as np
import sounddevice as sd
import pyperclip
from pynput import keyboard
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from transformers import AutoProcessor, pipeline

# --- 設定 ---
MODEL_PATH = r"./whisper-small-ov"  # ステップ1で生成したフォルダパス
SAMPLE_RATE = 16000
TRIGGER_KEY = keyboard.Key.f8

# "GPU" (内蔵 Arc GPU: 最速) または "NPU" (省電力・ファン静音)
DEVICE = "GPU"

print(f"OpenVINO モデルを {DEVICE} に展開中...")
# 初回のみハードウェア向けコンパイルが走るため十数秒〜1分ほど要します
model = OVModelForSpeechSeq2Seq.from_pretrained(
    MODEL_PATH,
    device=DEVICE,
    ov_config={"PERFORMANCE_HINT": "LATENCY"}
)
processor = AutoProcessor.from_pretrained(MODEL_PATH)

pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
)
print(f"準備完了！ [{DEVICE} 稼働] [F8] キーを押しながら話してください。")

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

def process_audio():
    if not audio_buffer:
        return
    
    audio_data = np.concatenate(audio_buffer, axis=0).flatten().astype(np.float32)
    if len(audio_data) < SAMPLE_RATE * 0.3:
        return

    # 推論実行
    result = pipe(
        audio_data,
        generate_kwargs={"language": "japanese", "task": "transcribe"}
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
