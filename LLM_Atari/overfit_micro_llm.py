import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import argparse

# === Argument parsing ===
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="tiny_llm", help="Base name for model files (default: tiny_llm)")
parser.add_argument("--lr", type=float, default=3e-3, help="Learning rate (default: 0.003)")
parser.add_argument("--verbose", action="store_true", help="Enable extra logging (sample output, gradients, etc.)")
args = parser.parse_args()
model_name = args.model_name
lr = args.lr
verbose = args.verbose

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

vocab_list = list("abcdefghijklmnopqrstuvwxyz .,!?")
vocab_size = len(vocab_list)
char_to_idx = {ch: i for i, ch in enumerate(vocab_list)}
idx_to_char = {i: ch for i, ch in enumerate(vocab_list)}

filtered_text = ''.join([ch if ch in char_to_idx else ' ' for ch in text])
data = torch.tensor([char_to_idx[c] for c in filtered_text], dtype=torch.long)

hidden_size = 64
num_layers = 1
num_heads = 1
seq_len = 32
epochs = 1000

# === Model definition ===
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(seq_len, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x):
        positions = torch.arange(0, x.size(0), device=x.device).unsqueeze(1)
        x = self.token_embedding(x) + self.pos_embedding(positions)
        x = x * (hidden_size ** 0.5)
        x = self.transformer(x)
        return self.lm_head(x)

model = TinyGPT()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# === Training loop ===
for epoch in range(epochs):
    seq = data[:seq_len].unsqueeze(1)
    target = data[1:seq_len+1].unsqueeze(1)

    optimizer.zero_grad()
    output = model(seq)
    loss = F.cross_entropy(output.view(-1, vocab_size), target.view(-1))
    loss.backward()

    if verbose:
        # Gradient stats
        grad_norms = [p.grad.abs().max().item() for p in model.parameters() if p.grad is not None]
        grad_mean = sum(g for g in grad_norms) / len(grad_norms) if grad_norms else 0

    optimizer.step()

    if epoch % 10 == 0:
        log = f"Epoch {epoch:4d}, Loss: {loss.item():.6f}"
        if verbose:
            log += f", Max Grad: {max(grad_norms):.4f}, Avg Grad: {grad_mean:.4f}"
        print(log)

    if verbose and epoch % 50 == 0:
        model.eval()
        with torch.no_grad():
            prompt = "the"
            ctx = torch.tensor([char_to_idx[c] for c in prompt], dtype=torch.long).unsqueeze(1)
            for _ in range(80):
                x = ctx[-seq_len:] if ctx.shape[0] >= seq_len else \
                    torch.cat([torch.zeros(seq_len - ctx.shape[0], 1, dtype=torch.long), ctx], dim=0)
                logits = model(x)
                next_token = torch.argmax(logits[-1, 0]).item()
                ctx = torch.cat([ctx, torch.tensor([[next_token]])], dim=0)
            gen = ''.join(idx_to_char[idx.item()] for idx in ctx.squeeze())
            print("📤 Sample:", gen[-100:])
        model.train()

# === Quantize and export ===
def quantize_tensor(tensor):
    tensor = tensor.clamp(-1, 1)
    tensor = (tensor * 127).round().clamp(-128, 127).to(torch.int8)
    return tensor

state_dict = model.state_dict()
with open(f"{model_name}_weights.bin", "wb") as f:
    for k, v in state_dict.items():
        q = quantize_tensor(v.cpu().flatten())
        q.numpy().tofile(f)

print(f"\n✅ Export complete: {model_name}_weights.bin")

# === Parameter and config export ===
param_count = sum(p.numel() for p in model.parameters())
param_size_bytes = param_count
param_size_mb = param_size_bytes / (1024 * 1024)
print(f"📊 Params: {param_count:,} | Size: {param_size_bytes:,} bytes ({param_size_mb:.4f} MB)")

config = {
    "hidden_size": hidden_size,
    "num_layers": num_layers,
    "num_heads": num_heads,
    "seq_len": seq_len,
    "vocab_list": vocab_list,
    "vocab_size": vocab_size,
    "train_file": "[inline tiny corpus]",
    "val_file": "[inline tiny corpus]",
    "learning_rate": lr
}
with open(f"{model_name}_config.json", "w") as f:
    json.dump(config, f, indent=2)

print(f"✅ Config saved to {model_name}_config.json")
