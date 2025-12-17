import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random

# =====================
# 常量定义
# =====================
SEQ_LEN = 100
SOS = 100
EOS = 101
PAD = 102
VOCAB_SIZE = 103  # 0-99 + SOS + EOS + PAD

# =====================
# 数据集定义
# =====================
class SignalToIndexDataset(Dataset):
    """
    signal: (100,) 0/1 序列
    target: [SOS, x1, x2, ..., xn, EOS]
    """
    def __init__(self, num_samples=5000, max_len=10):
        self.samples = []
        for _ in range(num_samples):
            n = random.randint(1, max_len)
            xs = sorted(random.sample(range(SEQ_LEN), n))
            signal = torch.zeros(SEQ_LEN)
            signal[xs] = 1.0
            tgt = [SOS] + xs + [EOS]
            self.samples.append((signal, torch.tensor(tgt)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    signals, tgts = zip(*batch)
    signals = torch.stack(signals)

    max_len = max(len(t) for t in tgts)
    tgt_pad = torch.full((len(tgts), max_len), PAD)
    for i, t in enumerate(tgts):
        tgt_pad[i, :len(t)] = t

    return signals, tgt_pad.long()


# =====================
# Causal Mask
# =====================
def generate_square_subsequent_mask(sz, device):
    return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()


# =====================
# Transformer 模型
# =====================
class TransformerGenerator(nn.Module):
    def __init__(self, d_model=128, nhead=4, num_layers=3):
        super().__init__()

        # Encoder
        self.signal_embed = nn.Linear(1, d_model)
        self.pos_embed = nn.Parameter(torch.randn(SEQ_LEN, d_model))

        # Decoder
        self.tgt_embed = nn.Embedding(VOCAB_SIZE, d_model)
        # 【新增】Decoder 的位置编码 (假设最大输出长度不会超过 SEQ_LEN)
        self.tgt_pos_embed = nn.Parameter(torch.randn(SEQ_LEN, d_model))

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            batch_first=True
        )

        self.fc_out = nn.Linear(d_model, VOCAB_SIZE)

    def forward(self, signal, tgt):
        device = signal.device
        B, T = tgt.shape

        # ===== Encoder =====
        enc = signal.unsqueeze(-1)                # (B, 100, 1)
        enc = self.signal_embed(enc)              # (B, 100, d)
        enc = enc + self.pos_embed.unsqueeze(0)   # positional encoding

        # ===== Decoder =====
        tgt_emb = self.tgt_embed(tgt)             # (B, T, d)
        # 【新增】加上位置编码 (切片以匹配当前 tgt 长度)
        tgt_emb = tgt_emb + self.tgt_pos_embed[:T].unsqueeze(0)

        # 生成 masks
        tgt_mask = generate_square_subsequent_mask(T, device)
        # 【新增】Padding Mask
        tgt_key_padding_mask = (tgt == PAD)

        out = self.transformer(
            src=enc,
            tgt=tgt_emb,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask # 【新增】传入 mask
        )
        return self.fc_out(out)


# =====================
# 训练流程
# =====================
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device:", device)
    
    dataset = SignalToIndexDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

    model = TransformerGenerator().to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(100):
        model.train()
        total_loss = 0

        for signal, tgt in loader:
            signal = signal.to(device)
            tgt = tgt.to(device)

            optimizer.zero_grad()

            # teacher forcing
            logits = model(signal, tgt[:, :-1])

            # 禁止第一个位置预测 EOS
            logits[:, 0, EOS] = -1e9

            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),
                tgt[:, 1:].reshape(-1)
            )
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch}: loss={total_loss / len(loader):.4f}")

    return model


# =====================
# 推理函数
# =====================
@torch.no_grad()
def inference(model, signal, max_len=20):
    model.eval()
    device = next(model.parameters()).device

    signal = signal.unsqueeze(0).to(device)
    ys = torch.tensor([[SOS]], device=device)

    for _ in range(max_len):
        logits = model(signal, ys)
        next_token = logits[:, -1].argmax(dim=-1)
        ys = torch.cat([ys, next_token.unsqueeze(1)], dim=1)
        if next_token.item() == EOS:
            break

    return ys.squeeze(0).tolist()


# =====================
# 主函数
# =====================
if __name__ == '__main__':
    model = train()

    # 测试
    xs = [5, 12, 33, 70]
    signal = torch.zeros(SEQ_LEN)
    signal[xs] = 1.0

    pred = inference(model, signal)
    print("GT:", xs)
    print("Pred:", pred)
