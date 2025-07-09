import torch
import torch.nn as nn
import torch.nn.functional as F

# === 1. Explicit Tiny vocab ===
vocab_list = list("abcdefghijklmnopqrstuvwxyz .,!?")
vocab_size = len(vocab_list)
char_to_idx = {ch: i for i, ch in enumerate(vocab_list)}
idx_to_char = {i: ch for i, ch in enumerate(vocab_list)}

# === 2. Load training and validation text ===
with open("train_tiny.txt", "r") as f:
    raw_train_text = f.read().lower()
with open("val_tiny.txt", "r") as f:
    raw_val_text = f.read().lower()

# Filter to allowed characters only
train_text = ''.join([ch if ch in char_to_idx else ' ' for ch in raw_train_text])
val_text = ''.join([ch if ch in char_to_idx else ' ' for ch in raw_val_text])

train_data = torch.tensor([char_to_idx[c] for c in train_text], dtype=torch.long)
val_data = torch.tensor([char_to_idx[c] for c in val_text], dtype=torch.long)

# === 3. TinyGPT definition ===
hidden_size = 48
num_layers = 2
num_heads = 2
seq_len = 32

class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        x = self.token_embedding(x) * (hidden_size ** 0.5)
        x = self.transformer(x)
        return self.lm_head(x)

model = TinyGPT()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# === 4. Training Loop ===
batch_size = 1  # For simplicity
epochs = 10000

for epoch in range(epochs):
    idx = torch.randint(0, len(train_data) - seq_len - 1, (batch_size,))
    seq = torch.stack([train_data[i:i+seq_len] for i in idx]).transpose(0,1)  # [seq_len, batch]
    target = torch.stack([train_data[i+1:i+seq_len+1] for i in idx]).transpose(0,1)

    optimizer.zero_grad()
    output = model(seq)
    loss = F.cross_entropy(output.view(-1, vocab_size), target.reshape(-1))
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        with torch.no_grad():
            val_idx = torch.randint(0, len(val_data) - seq_len - 1, (batch_size,))
            val_seq = torch.stack([val_data[i:i+seq_len] for i in val_idx]).transpose(0,1)
            val_target = torch.stack([val_data[i+1:i+seq_len+1] for i in val_idx]).transpose(0,1)
            val_output = model(val_seq)
            val_loss = F.cross_entropy(val_output.view(-1, vocab_size), val_target.reshape(-1))
        print(f"Epoch {epoch}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

# === 5. Quantization and Export ===
def quantize_tensor(tensor):
    tensor = tensor.clamp(-1, 1)  # safety
    tensor = (tensor * 127).round().clamp(-128, 127).to(torch.int8)
    return tensor

state_dict = model.state_dict()
with open("tiny_llm_weights.bin", "wb") as f:
    for k, v in state_dict.items():
        q = quantize_tensor(v.cpu().flatten())
        q.numpy().tofile(f)

print(f"\n✅ Export complete: tiny_llm_weights.bin ({sum(v.numel() for v in state_dict.values())} params)")
