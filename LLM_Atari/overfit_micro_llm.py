import torch
import torch.nn as nn
import torch.nn.functional as F
import json

# === Tiny test corpus ===
text = """
the cat sat.
the dog ran.
the fox jumped.
the sun rose.
the moon shone.
lila laughed.
tom smiled.
anna danced.
birds sang.
trees swayed.
""".strip().lower()

# === Vocab setup ===
vocab_list = list("abcdefghijklmnopqrstuvwxyz .,!?")
vocab_size = len(vocab_list)
char_to_idx = {ch: i for i, ch in enumerate(vocab_list)}
idx_to_char = {i: ch for i, ch in enumerate(vocab_list)}

filtered_text = ''.join([ch if ch in char_to_idx else ' ' for ch in text])
data = torch.tensor([char_to_idx[c] for c in filtered_text], dtype=torch.long)

# === Model hyperparameters ===
hidden_size = 64
num_layers = 1
num_heads = 1
seq_len = 32
epochs = 1000
lr = 3e-3

# === Model definition ===
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
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# === Overfit training loop ===
for epoch in range(epochs):
    # Always train on the same slice for deterministic overfit testing
    if len(data) < seq_len + 1:
        raise ValueError("Text too short for seq_len, reduce seq_len or use longer text.")
    seq = data[:seq_len].unsqueeze(1)      # [seq_len, 1]
    target = data[1:seq_len+1].unsqueeze(1) # [seq_len, 1]

    optimizer.zero_grad()
    output = model(seq)
    loss = F.cross_entropy(output.view(-1, vocab_size), target.view(-1))
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# === Quantize and export weights ===
def quantize_tensor(tensor):
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor * 127).round().clamp(-128, 127).to(torch.int8)
    return tensor

state_dict = model.state_dict()
with open("tiny_llm_weights.bin", "wb") as f:
    for k, v in state_dict.items():
        q = quantize_tensor(v.cpu().flatten())
        q.numpy().tofile(f)

print("\n✅ Export complete: tiny_llm_weights.bin")

# === Parameter count & size ===
param_count = sum(p.numel() for p in model.parameters())
param_size_bytes = param_count
param_size_mb = param_size_bytes / (1024 * 1024)

print(f"📊 Model parameters: {param_count:,} params")
print(f"📦 Model size: {param_size_bytes:,} bytes ({param_size_mb:.4f} MB)\n")

# === Save config ===
config = {
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "num_heads": num_heads,
    "seq_len": seq_len,
    "vocab_list": vocab_list,
    "vocab_size": vocab_size
}
with open("tiny_llm_config.json", "w") as f:
    json.dump(config, f, indent=2)

print("✅ Config saved to tiny_llm_config.json")
