# train_micro_llm_mps.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import json
import os

# === 0. Argument parsing ===
parser = argparse.ArgumentParser(description="Train Micro LLM on MPS with quantized export and config.")
parser.add_argument("--model_name", type=str, default="tiny_llm", help="Base name for saved model files.")
parser.add_argument("--train_file", type=str, default="train_tiny.txt", help="Training file (default: train_tiny.txt)")
parser.add_argument("--val_file", type=str, default="val_tiny.txt", help="Validation file (default: val_tiny.txt)")
parser.add_argument("--hidden_size", type=int, default=32, help="Hidden size (default: 32)")
parser.add_argument("--num_layers", type=int, default=1, help="Number of transformer layers (default: 1)")
parser.add_argument("--num_heads", type=int, default=1, help="Number of attention heads (default: 1)")
parser.add_argument("--seq_len", type=int, default=32, help="Sequence length (default: 32)")
parser.add_argument("--epochs", type=int, default=1000, help="Number of training epochs (default: 1000)")
parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3)")
parser.add_argument("--debug", action="store_true", help="Enable debug output during training")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size (default: 4)")
args = parser.parse_args()

# === 1. Hyperparams and device ===
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"🖥️  Using device: {device}")

model_name = args.model_name
hidden_size = args.hidden_size
num_layers = args.num_layers
num_heads = args.num_heads
seq_len = args.seq_len
epochs = args.epochs
lr = args.lr
batch_size = args.batch_size
debug = args.debug

# === 2. Load and clean data ===
def load_and_clean(path):
    with open(path, "r") as f:
        raw = f.read().lower()
    return raw

train_text = load_and_clean(args.train_file)
val_text = load_and_clean(args.val_file)

# === 3. Vocab ===
vocab = sorted(set(train_text + val_text))
vocab_size = len(vocab)
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}

def encode(text):
    return torch.tensor([char_to_idx.get(c, 0) for c in text], dtype=torch.long)

train_data = encode(train_text)
val_data = encode(val_text)

# === 4. Model ===
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(seq_len, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        seq_len_actual, batch_size = x.shape
        positions = torch.arange(0, seq_len_actual, device=x.device).unsqueeze(1).expand(seq_len_actual, batch_size)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        x = x * (hidden_size ** 0.5)
        x = self.transformer(x)
        return self.lm_head(x)

model = TinyGPT().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# === 5. Training loop ===
for epoch in range(epochs):
    model.train()

    start_idx = torch.randint(0, len(train_data) - seq_len - 1, (batch_size,))
    input_seq = torch.stack([train_data[i : i + seq_len] for i in start_idx]).transpose(0, 1)
    target_seq = torch.stack([train_data[i + 1 : i + seq_len + 1] for i in start_idx]).transpose(0, 1)

    input_seq = input_seq.to(device)
    target_seq = target_seq.to(device)

    optimizer.zero_grad()
    output = model(input_seq)
    loss = F.cross_entropy(output.view(-1, vocab_size), target_seq.reshape(-1))
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
        if debug:
            for b in range(min(2, batch_size)):
                sample_input = ''.join(idx_to_char[i.item()] for i in input_seq[:, b].cpu())
                sample_target = ''.join(idx_to_char[i.item()] for i in target_seq[:, b].cpu())
                print(f"🧪 Input sample:  {sample_input}")
                print(f"🎯 Target sample: {sample_target}")
                print(f"🔍 Embedding[0]: {model.token_embedding.weight[0][:5].detach().cpu().numpy()}")
                print()

# === 6. Quantized export ===
def quantize_tensor(tensor):
    tensor = tensor.clamp(-1, 1)
    return (tensor * 127).round().clamp(-128, 127).to(torch.int8)

weights_path = f"{model_name}_weights.bin"
with open(weights_path, "wb") as f:
    for k, v in model.state_dict().items():
        #q = quantize_tensor(v.cpu().flatten())
        #q.numpy().tofile(f)
        v.cpu().flatten().numpy().astype(np.float32).tofile(f)

# === 7. Save config ===
config = {
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "num_heads": num_heads,
    "seq_len": seq_len,
    "vocab_list": vocab,
    "vocab_size": vocab_size,
    "train_file": args.train_file,
    "val_file": args.val_file,
    "learning_rate": lr,
    "batch_size": batch_size
}
config_path = f"{model_name}_config.json"
with open(config_path, "w") as f:
    json.dump(config, f, indent=2)

# === 8. Stats ===
param_count = sum(p.numel() for p in model.parameters())
param_size_bytes = param_count
param_size_mb = param_size_bytes / (1024 * 1024)

print(f"\n✅ Training complete: {model_name}")
print(f"📦 Saved weights to {weights_path}")
print(f"🧠 Config written to {config_path}")
print(f"📊 Parameters: {param_count:,}")
print(f"📦 Model size: {param_size_bytes:,} bytes ({param_size_mb:.4f} MB)")
