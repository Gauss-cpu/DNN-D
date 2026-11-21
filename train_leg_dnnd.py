import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from scipy.signal import savgol_filter
import sys
from dnnd_module import DNNDModel

ROOT = Path.home() / "Desktop/Physical_Intelligence"
DATA_PATH = ROOT / "dataset_cmu/features/leg_dataset.npz"
SAVE_PATH = ROOT / "video_2_simulation_4_leg/dnnd_leg.pt"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

SEQ_LEN = 30
BATCH_SIZE = 1024
EPOCHS = 10  # 数据少了，10 轮足够看效果
LR = 1e-3

if __name__ == '__main__':
    print("加载腿部 NPZ 数据...")
    data = np.load(DATA_PATH, allow_pickle=True)
    X_raw_20d = data["X"]  # [N, 20] 特征矩阵

    # 提取第 8 和第 9 列（髋和膝的角度），绝对是 2 维
    q_raw = X_raw_20d[:, 8:10]

    # ==========================================
    # 【核心提速：直接切片！】
    # 取前 200,000 条数据，瞬间跑完！
    # ==========================================
    MAX_SAMPLES = 200000
    if len(q_raw) > MAX_SAMPLES:
        q_raw = q_raw[:MAX_SAMPLES]

    print(f"切片后的 2-DOF 数据形状: {q_raw.shape}")

    # 造平滑标签
    q_smooth = np.zeros_like(q_raw)
    for j in range(2):  # 循环 2 次
        q_smooth[:, j] = savgol_filter(q_raw[:, j], 11, 2)

    # 计算增量 Delta
    delta_q = np.diff(q_smooth, axis=0)
    delta_q = np.vstack([np.zeros((1, 2)), delta_q])  # 补 2 维的 0

    # 切分序列
    X_seq, Y_seq = [], []
    for i in range(len(q_raw) - SEQ_LEN):
        X_seq.append(q_raw[i: i + SEQ_LEN])
        Y_seq.append(delta_q[i + SEQ_LEN - 1])

    X_tensor = torch.tensor(np.array(X_seq), dtype=torch.float32)
    Y_tensor = torch.tensor(np.array(Y_seq), dtype=torch.float32)
    print(f"最终训练数据形状: X {X_tensor.shape}, Y {Y_tensor.shape}")

    # DataLoader
    train_loader = DataLoader(
        TensorDataset(X_tensor, Y_tensor),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    # 实例化 2-DOF 模型
    model = DNNDModel(dim=2, seq_len=SEQ_LEN).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()

    print("开始训练 2-DOF 腿部 DNN-D (快速切片版)...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)

            pred_delta = model(xb)
            loss_delta = criterion(pred_delta, yb)

            base_q = xb[:, -1, :]
            pred_q = base_q + pred_delta
            true_q = base_q + yb
            loss_abs = criterion(pred_q, true_q)

            loss = loss_delta + 0.5 * loss_abs

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)

        epoch_loss /= len(X_tensor)
        print(f"Epoch {epoch:2d} | Loss: {epoch_loss:.6f}")

    torch.save(model.state_dict(), SAVE_PATH)
    print(f"✅ 腿部模型已保存至: {SAVE_PATH}")