import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse

# === 0. Argument parser for k ===
parser = argparse.ArgumentParser(description="Micro LLM Top-k Sampler")
parser.add_argument("--k", type=int, default=5, help="Top-k sampling value (default: 5)")
args = parser.parse_args()
top_k = args.k

# === 1. Explicit vocab and mapping ===
vocab_list = list("abcdefghijklmnopqrstuvwxyz .,!?")
vocab_size = len(vocab_list)
char_to_idx = {ch: i for i, ch in enumerate(vocab_list)}
idx_to_char = {i: ch for i, ch in enumerate(vocab_list)}

# === 2. Model definition matching your training ===
hidden_size = 32
num_layers = 1
num_heads = 1
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

# === 3. Load quantized weights ===
state_dict = model.state_dict()
with open("tiny_llm_weights.bin", "rb") as f:
    for k in state_dict.keys():
        numel = state_dict[k].numel()
        q = np.frombuffer(f.read(numel), dtype=np.int8).astype(np.float32) / 127.0
        state_dict[k] = torch.from_numpy(q.reshape(state_dict[k].shape))

model.load_state_dict(state_dict)
model.eval()

print(f"✅ Weights loaded successfully. Using top-k = {top_k}")

# === 4. Top-k Sampling Helper ===
def top_k_sampling(logits, k=5):
    values, indices = torch.topk(logits, k)
    probs = F.softmax(values, dim=0)
    idx = torch.multinomial(probs, num_samples=1)
    return indices[idx].item()

# === 5. Generation loop ===
prompt = "the cat"
num_generate = 300  # characters to generate

context = [char_to_idx.get(ch, char_to_idx[' ']) for ch in prompt.lower()]
context = torch.tensor(context, dtype=torch.long).unsqueeze(1)  # [seq_len, batch]

print(f"\n🌱 Prompt: {prompt}\n")
print(prompt, end='', flush=True)

for _ in range(num_generate):
    input_seq = context[-seq_len:] if context.shape[0] >= seq_len else \
                torch.cat([torch.zeros(seq_len - context.shape[0], 1, dtype=torch.long), context], dim=0)

    with torch.no_grad():
        logits = model(input_seq)
        next_logits = logits[-1, 0, :]
        next_idx = top_k_sampling(next_logits, k=top_k)

    context = torch.cat([context, torch.tensor([[next_idx]], dtype=torch.long)], dim=0)
    print(idx_to_char[next_idx], end='', flush=True)

print("\n\n✅ Generation complete using top-k sampling.")
