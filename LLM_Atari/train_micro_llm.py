import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json

# === 1. CLI args ===
parser = argparse.ArgumentParser(description="Train a tiny character-level transformer.")
parser.add_argument("--model_name", type=str, default="tiny_llm", help="Base name for saved model files.")
parser.add_argument("--train_file", type=str, default="train_tiny.txt", help="Training file path.")
parser.add_argument("--val_file", type=str, default="val_tiny.txt", help="Validation file path.")
parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs.")
parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate.")
args = parser.parse_args()

# === 2. Hyperparams (can move to args later) ===
hidden_size = 128
num_layers = 3
num_heads = 1
seq_len = 128

# === 3. Read data ===
with open(args.train_file, 'r') as f:
    train_text = f.read().lower()
with open(args.val_file, 'r') as f:
    val_text = f.read().lower()

# === 4. Vocab ===
vocab = sorted(set(train_text + val_text))
vocab_size = len(vocab)
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}

def encode(text):
    return torch.tensor([char_to_idx.get(c, 0) for c in text], dtype=torch.long)

train_data = encode(train_text)
val_data = encode(val_text)

# === 5. Model ===
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(seq_len, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        pos = torch.arange(0, x.size(0), device=x.device).unsqueeze(1)
        x = self.token_embedding(x) + self.pos_embedding(pos % seq_len)
        x = x * (hidden_size ** 0.5)
        x = self.transformer(x)
        return self.lm_head(x)

model = TinyGPT()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# === 6. Training loop ===
for epoch in range(args.epochs):
    model.train()
    input_seq = train_data[:seq_len].unsqueeze(1)
    target_seq = train_data[1:seq_len+1].unsqueeze(1)

    optimizer.zero_grad()
    output = model(input_seq)
    loss = F.cross_entropy(output.view(-1, vocab_size), target_seq.view(-1))
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

# === 7. Quantize weights ===
def quantize_tensor(tensor):
    return (tensor.clamp(-1, 1) * 127).round().clamp(-128, 127).to(torch.int8)

with open(f"{args.model_name}_weights.bin", "wb") as f:
    for name, param in model.state_dict().items():
        q = quantize_tensor(param.cpu().flatten())
        q.numpy().tofile(f)

# === 8. Save config ===
config = {
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "num_heads": num_heads,
    "seq_len": seq_len,
    "vocab_list": vocab,
    "vocab_size": vocab_size,
    "train_file": args.train_file,
    "val_file": args.val_file,
    "learning_rate": args.lr
}
with open(f"{args.model_name}_config.json", "w") as f:
    json.dump(config, f, indent=2)

# === 9. Stats ===
param_count = sum(p.numel() for p in model.parameters())
param_size = param_count
print(f"\n✅ Training complete.")
print(f"📦 Saved: {args.model_name}_weights.bin and config.")
print(f"📊 Params: {param_count:,} | Size: {param_size / (1024 * 1024):.4f} MB")
