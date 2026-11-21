import torch
import torch.nn as nn
import numpy as np


class DNNDModel(nn.Module):
    def __init__(self, dim=8, seq_len=30, d_model=64, nhead=4):
        super().__init__()
        self.input_proj = nn.Linear(dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)

        self.output_head = nn.Linear(d_model, dim)

    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoder
        memory = self.encoder(x)
        out = self.decoder(x, memory)
        return self.output_head(out[:, -1, :])


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"


def denoise_trajectory_dnnd(q_raw, model_path, dim, seq_len=30):
    """
    通用 DNN-D 推理接口
    q_raw: 带噪的原始轨迹 [N, dim]
    """
    model = DNNDModel(dim=dim, seq_len=seq_len).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    N = len(q_raw)
    q_raw_padded = np.pad(q_raw, ((seq_len - 1, 0), (0, 0)), mode='edge')
    q_denoised = np.zeros_like(q_raw)
    q_denoised[0] = q_raw[0]

    with torch.no_grad():
        for i in range(1, N):
            seq_in = q_raw_padded[i: i + seq_len]
            seq_t = torch.tensor(seq_in, dtype=torch.float32).unsqueeze(0).to(DEVICE)

            # 预测增量
            delta_q = model(seq_t).cpu().numpy()[0]

            # 【核心修改】：纯累加 + 视觉牵引
            # 以累加为基础，但强行拉回带噪蓝线一点点，防止漂移
            q_pure_accum = q_denoised[i - 1] + delta_q

            # alpha 是牵引系数（0.2 表示每次只信 80% 的累加，拉回 20% 到真实观测）
            alpha = 0.2
            q_denoised[i] = (1 - alpha) * q_pure_accum + alpha * q_raw[i]

    return q_denoised