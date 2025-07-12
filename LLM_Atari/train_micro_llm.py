import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json

# === 0. Argument parsing ===
parser = argparse.ArgumentParser(description="Train Micro LLM with quantized export and config (with positional embeddings).")
parser.add_argument("--hidden_size", type=int, default=32, help="Hidden size (default: 32)")
parser.add_argument("--num_layers", type=int, default=1, help="Number of transformer layers (default: 1)")
parser.add_argument("--num_heads", type=int, default=1, help="Number of attention heads (default: 1)")
parser.add_argument("--seq_len", type=int, default=32, help="Sequence length (default: 32)")
parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs (default: 1000)")
parser.add_argument("--train_file", type=str, default="train_tiny.txt", help="Training text file (default: train_tiny.txt)")
parser.add_argument("--val_file", type=str, default="val_tiny.txt", help="Validation text file (default: val_tiny.txt)")
parser.add_argument("--model_name", type=str, default="tiny_llm", help="Base name for model files (default: tiny_llm)")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 0.001)")
args = parser.parse_args()

# === 1. Vocab definition ===
vocab_list = list("abcdefghijklmnopqrstuvwxyz .,!?")
vocab_size = len(vocab_list)
char_to_idx = {ch: i for i, ch in enumerate(vocab_list)}
idx_to_char = {i: ch for i, ch in enumerate(vocab_list)}

# === 2. Load data ===
with open(args.train_file, "r") as f:
    raw_train_text = f.read().lower()
with open(args.val_file, "r") as f:
    raw_val_text = f.read().lower()

train_text = ''.join([ch if ch in char_to_idx else ' ' for ch in raw_train_text])
val_text = ''.join([ch if ch in char_to_idx else ' ' for ch in raw_val_text])

train_data = torch.tensor([char_to_idx[c] for c in train_text], dtype=torch.long)
val_data = torch.tensor([char_to_idx[c] for c in val_text], dtype=torch.long)

# === 3. Model definition ===
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, args.hidden_size)
        self.pos_embedding = nn.Embedding(args.seq_len, args.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=args.hidden_size, nhead=args.num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=args.num_layers)
        self.lm_head = nn.Linear(args.hidden_size, vocab_size)

    def forward(self, x):
        positions = torch.arange(0, x.size(0), device=x.device).unsqueeze(1)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        x = x * (args.hidden_size ** 0.5)
        x = self.transformer(x)
        return self.lm_head(x)

model = TinyGPT()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

# === 4. Training loop ===
batch_size = 1
for epoch in range(args.epochs):
    idx = torch.randint(0, len(train_data) - args.seq_len - 1, (batch_size,))
    seq = torch.stack([train_data[i:i+args.seq_len] for i in idx]).transpose(0,1)
    target = torch.stack([train_data[i+1:i+args.seq_len+1] for i in idx]).transpose(0,1)

    optimizer.zero_grad()
    output = model(seq)
    loss = F.cross_entropy(output.view(-1, vocab_size), target.reshape(-1))
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        with torch.no_grad():
            val_idx = torch.randint(0, len(val_data) - args.seq_len - 1, (batch_size,))
            val_seq = torch.stack([val_data[i:i+args.seq_len] for i in val_idx]).transpose(0,1)
            val_target = torch.stack([val_data[i+1:i+args.seq_len+1] for i in val_idx]).transpose(0,1)
            val_output = model(val_seq)
            val_loss = F.cross_entropy(val_output.view(-1, vocab_size), val_target.reshape(-1))
        print(f"Epoch {epoch}, Train Loss: {loss.item():.6f}, Val Loss: {val_loss.item():.6f}")

# === 5. Quantization and export ===
def quantize_tensor(tensor):
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor * 127).round().clamp(-128, 127).to(torch.int8)
    return tensor

state_dict = model.state_dict()
with open(f"{args.model_name}_weights.bin", "wb") as f:
    for k, v in state_dict.items():
        q = quantize_tensor(v.cpu().flatten())
        q.numpy().tofile(f)

print(f"\n✅ Export complete: {args.model_name}_weights.bin")

# === Parameter count & size ===
param_count = sum(p.numel() for p in model.parameters())
param_size_bytes = param_count
param_size_mb = param_size_bytes / (1024 * 1024)

print(f"📊 Model parameters: {param_count:,}")
print(f"📦 Model size: {param_size_bytes:,} bytes ({param_size_mb:.4f} MB)")

# === 6. Save config ===
config = {
    "hidden_size": args.hidden_size,
    "num_layers": args.num_layers,
    "num_heads": args.num_heads,
    "seq_len": args.seq_len,
    "vocab_list": vocab_list,
    "vocab_size": vocab_size,
    "train_file": args.train_file,
    "val_file": args.val_file,
    "learning_rate": args.lr
}
with open(f"{args.model_name}_config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"✅ Config saved to {args.model_name}_config.json")
