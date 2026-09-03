import os
import torch
import whisper
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from optimum.intel.openvino import OVModelForSpeechSeq2Seq

# --- 設定 ---
PT_MODEL_PATH = r"D:\offline_models\small.pt"  # 手元の .pt ファイルのパス
MODEL_SIZE = "small"                           # モデルの規模 ("tiny", "base", "small", "medium", "large-v3" など)
OUTPUT_DIR = r"./whisper-small-ov"             # OpenVINOモデルの保存先フォルダ

print(f"[1/4] 公式の .pt ファイルを読み込み中: {PT_MODEL_PATH}")
checkpoint = torch.load(PT_MODEL_PATH, map_location="cpu")
state_dict = checkpoint["model_state_dict"]

print(f"[2/4] Transformers 形式の骨格モデルを初期化中 (openai/whisper-{MODEL_SIZE})...")
# ベースとなる設定とトークナイザーをロード
hf_model_id = f"openai/whisper-{MODEL_SIZE}"
hf_model = WhisperForConditionalGeneration.from_pretrained(hf_model_id)
processor = WhisperProcessor.from_pretrained(hf_model_id)

# 公式の重みキーを Transformers のキー構造に変換してロード
# （transformers の内部変換ユーティリティを利用）
from transformers.models.whisper.convert_openai_to_hf import convert_openai_whisper_to_hf
converted_state_dict = convert_openai_whisper_to_hf(state_dict, hf_model.config)
hf_model.load_state_dict(converted_state_dict)

# 一時的な PyTorch 形式として保存
temp_hf_dir = "./temp_hf_model"
hf_model.save_pretrained(temp_hf_dir)
processor.save_pretrained(temp_hf_dir)

print(f"[3/4] OpenVINO IR 形式 (FP16) にエクスポート中...")
ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
    temp_hf_dir,
    export=True,
    compile=False
)

# OpenVINO 形式で出力先フォルダに保存
ov_model.save_pretrained(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# 一時フォルダのクリーンアップ（任意）
import shutil
shutil.rmtree(temp_hf_dir, ignore_errors=True)

print(f"[4/4] 変換が完了しました！ 保存先: {OUTPUT_DIR}")
