import os
import torch
import whisper
import openvino as ov

# --- SDPA のハードウェア最適化フラグをオフにして標準実装へフォールバック ---
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

PT_MODEL_PATH = r"D:\offline_models\small.pt"
OUTPUT_DIR = r"./whisper-small-ov"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"[1/3] ローカルの .pt ファイルを読み込み中...")
model = whisper.load_model(PT_MODEL_PATH, device="cpu")
model.eval()

# --- 変換実行（SDPA を強制無効化するコンテキスト内で実行） ---
with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
    
    # 1. エンコーダの変換
    print("[2/3] エンコーダを変換中...")
    n_mels = model.dims.n_mels
    dummy_mel = torch.zeros((1, n_mels, 3000), dtype=torch.float32)

    class EncoderWrapper(torch.nn.Module):
        def __init__(self, encoder):
            super().__init__()
            self.encoder = encoder
        def forward(self, mel):
            return self.encoder(mel)

    encoder_wrapper = EncoderWrapper(model.encoder)
    ov_encoder = ov.convert_model(encoder_wrapper, example_input=dummy_mel)
    ov.save_model(ov_encoder, os.path.join(OUTPUT_DIR, "whisper_encoder.xml"), compress_to_fp16=True)

    # 2. デコーダの変換
    print("[3/3] デコーダを変換中...")
    class DecoderWrapper(torch.nn.Module):
        def __init__(self, decoder):
            super().__init__()
            self.decoder = decoder
        def forward(self, tokens, audio_features):
            return self.decoder(tokens, audio_features)

    dummy_tokens = torch.tensor([[50258]], dtype=torch.long)
    dummy_audio_features = torch.zeros((1, 1500, model.dims.n_audio_state), dtype=torch.float32)

    decoder_wrapper = DecoderWrapper(model.decoder)
    ov_decoder = ov.convert_model(
        decoder_wrapper,
        example_input=(dummy_tokens, dummy_audio_features)
    )
    ov.save_model(ov_decoder, os.path.join(OUTPUT_DIR, "whisper_decoder.xml"), compress_to_fp16=True)

print("\n変換成功しました！")
