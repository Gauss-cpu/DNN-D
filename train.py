import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

ROOT = Path.home() / "Desktop/Physical_Intelligence"
DATASET_DIR = ROOT / "dataset_penn"
MODEL_DIR = ROOT / "video_2_simulation_4_arm"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SEQ_LEN = 30
BATCH_SIZE = 64
EPOCHS = 50
LR = 1e-3

# 1. 构造数据集
X_raw_seq, Y_delta_seq = [], []
print("加载数据并制作增量标签...")
files = list(DATASET_DIR.rglob("*.json"))
for f_path in files:
    with open(f_path) as f:
        data = json.load(f)
    q_raw = np.array([frame["q"] for frame in data["trajectory"]])
    if len(q_raw) < SEQ_LEN: continue

    # 锁死不可靠关节
    for i in [0, 2, 4, 7]: q_raw[:, i] = 0.0

    # 老师傅：强行滤出干净数据作为 Ground Truth
    q_smooth = q_raw.copy()
    for j in range(8):
        q_smooth[:, j] = savgol_filter(q_raw[:, j], 11, 2)

    # 计算平滑增量 Delta_q
    delta_q = np.diff(q_smooth, axis=0)
    delta_q = np.vstack([np.zeros((1, 8)), delta_q])  # 第一帧增量为0

    # 切分序列
    for i in range(len(q_raw) - SEQ_LEN):
        X_raw_seq.append(q_raw[i:i + SEQ_LEN])
        Y_delta_seq.append(delta_q[i + SEQ_LEN - 1])  # 预测当前帧的增量

X = torch.tensor(np.array(X_raw_seq), dtype=torch.float32)
Y = torch.tensor(np.array(Y_delta_seq), dtype=torch.float32)
print(f"数据集大小: X {X.shape}, Y {Y.shape}")

train_ds = TensorDataset(X, Y)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)


# 2. DNN-D 网络结构 (Encoder-Decoder)
class DNNDModel(nn.Module):
    def __init__(self, dim=8, seq_len=30, d_model=64, nhead=4):
        super().__init__()
        self.input_proj = nn.Linear(dim, d_model)
        self.pos_encoder = nn.Parameter(torch.randn(1, seq_len, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=2)
        self.output_head = nn.Linear(d_model, dim)  # 输出 8 维增量

    def forward(self, x):
        x = self.input_proj(x) + self.pos_encoder
        memory = self.encoder(x)
        out = self.decoder(x, memory)
        # 只输出当前帧（最后一帧）的增量
        return self.output_head(out[:, -1, :])


model = DNNDModel().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.MSELoss()

# 3. 训练循环
print("开始训练 DNN-D (增量动力学) 网络...")
losses = []
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        pred_delta = model(xb)  # 预测增量

        # 损失1：增量误差
        loss_delta = criterion(pred_delta, yb)

        # 损失2：积分后的绝对角度误差 (重建误差，防止增量误差累积)
        # 取输入序列的最后一帧作为基准，加上预测增量
        base_q = xb[:, -1, :]
        pred_q = base_q + pred_delta
        true_q = base_q + yb  # 这里用带噪 base_q + 真实增量 近似真实轨迹
        loss_abs = criterion(pred_q, true_q)

        loss = loss_delta + 0.5 * loss_abs

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * len(xb)

    epoch_loss /= len(train_ds)
    losses.append(epoch_loss)
    if epoch % 10 == 0:
        print(f"Epoch {epoch:2d} | Loss: {epoch_loss:.6f}")

torch.save(model.state_dict(), MODEL_DIR / "dnnd_model.pt")
print("模型已保存: dnnd_model.pt")