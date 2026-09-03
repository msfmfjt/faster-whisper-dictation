import os
import torch
import whisper
import openvino as ov

PT_MODEL_PATH = r"D:\offline_models\small.pt"
OUTPUT_DIR = r"./whisper-small-ov"
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("モデル読み込み中...")
model = whisper.load_model(PT_MODEL_PATH, device="cpu")
model.eval()

# 1. エンコーダの変換（Mel -> Audio Features）
encoder = model.encoder
dummy_mel = torch.zeros((1, model.dims.n_mels, 3000), dtype=torch.float32)
temp_onnx = os.path.join(OUTPUT_DIR, "temp_encoder.onnx")

print("エンコーダをエクスポート中...")
torch.onnx.export(
    encoder,
    dummy_mel,
    temp_onnx,
    input_names=["mel"],
    output_names=["audio_features"],
    dynamic_axes={"mel": {0: "batch_size"}, "audio_features": {0: "batch_size"}},
    opset_version=17  # SDPAを安定して分解できるバージョン
)

# OpenVINO IR 形式へ変換
ov_model = ov.convert_model(temp_onnx)
ov.save_model(ov_model, os.path.join(OUTPUT_DIR, "whisper_encoder.xml"), compress_to_fp16=True)
os.remove(temp_onnx)
print("エンコーダ変換完了！")
