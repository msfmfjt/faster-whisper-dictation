import os
import torch
import whisper
import openvino as ov

# --- 設定 ---
PT_MODEL_PATH = r"D:\offline_models\small.pt"  # 手元の .pt パス
OUTPUT_DIR = r"./whisper-small-ov"             # 保存先フォルダ

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[1/3] ローカルの .pt ファイルを whisper で読み込み中: {PT_MODEL_PATH}")
# 外部通信なしでローカルの .pt を直接ロード
model = whisper.load_model(PT_MODEL_PATH, device="cpu")
model.eval()

# --- 1. エンコーダの変換 ---
print("[2/3] エンコーダを OpenVINO IR 形式へ変換中...")
class EncoderWrapper(torch.nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder
    def forward(self, mel):
        return self.encoder(mel)

# Whisper の Mel スペクトログラム入力シェイプ: (batch, n_mels=80, n_frames=3000)
# ※large-v3 のみ n_mels=128
n_mels = model.dims.n_mels
dummy_mel = torch.zeros((1, n_mels, 3000), dtype=torch.float32)

encoder_wrapper = EncoderWrapper(model.encoder)
ov_encoder = ov.convert_model(encoder_wrapper, example_input=dummy_mel)
ov.save_model(ov_encoder, os.path.join(OUTPUT_DIR, "whisper_encoder.xml"), compress_to_fp16=True)

# --- 2. デコーダの変換 ---
print("[3/3] デコーダを OpenVINO IR 形式へ変換中...")
class DecoderWrapper(torch.nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder
    def forward(self, tokens, audio_features):
        return self.decoder(tokens, audio_features)

# デコーダのダミー入力: トークン列 (1, seq_len) と エンコーダ出力 (1, 1500, n_audio_state)
dummy_tokens = torch.tensor([[50258]], dtype=torch.long)  # 開始トークン例
dummy_audio_features = torch.zeros((1, 1500, model.dims.n_audio_state), dtype=torch.float32)

decoder_wrapper = DecoderWrapper(model.decoder)
ov_decoder = ov.convert_model(
    decoder_wrapper,
    example_input=(dummy_tokens, dummy_audio_features)
)
ov.save_model(ov_decoder, os.path.join(OUTPUT_DIR, "whisper_decoder.xml"), compress_to_fp16=True)

print(f"\n変換完了！ 以下のファイルが生成されました:")
print(f"  - {os.path.join(OUTPUT_DIR, 'whisper_encoder.xml / .bin')}")
print(f"  - {os.path.join(OUTPUT_DIR, 'whisper_decoder.xml / .bin')}")
