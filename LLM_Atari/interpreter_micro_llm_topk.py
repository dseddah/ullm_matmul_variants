import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import json
import os
 
# === 0. CLI arguments ===
parser = argparse.ArgumentParser(description="Micro LLM Top-k Sampler with positional generation.")
parser.add_argument("--k", type=int, default=1, help="Top-k sampling value (default: 1)")
parser.add_argument("--prompt", type=str, default="the ", help="Prompt for generation (default: 'the ')")
parser.add_argument("--generate", type=int, default=200, help="Number of characters to generate (default: 200)")
parser.add_argument("--model_name", type=str, default="tiny_llm", help="Base name of model (default: tiny_llm)")
parser.add_argument("--debug", action="store_true", help="Enable debug output during generation")
args = parser.parse_args()

top_k = args.k
prompt = args.prompt
num_generate = args.generate
model_name = args.model_name
debug = args.debug

# === 1. Load config and vocab ===
config_file = f"{model_name}_config.json"
weights_file = f"{model_name}_weights.bin"

if not os.path.exists(config_file):
    raise FileNotFoundError(f"❌ Config file not found: {config_file}")
if not os.path.exists(weights_file):
    raise FileNotFoundError(f"❌ Weights file not found: {weights_file}")

with open(config_file, "r") as f:
    config = json.load(f)

hidden_size = config["hidden_size"]
num_layers = config["num_layers"]
num_heads = config["num_heads"]
seq_len = config["seq_len"]
vocab_list = config["vocab_list"]
vocab_size = config["vocab_size"]
char_to_idx = {ch: i for i, ch in enumerate(vocab_list)}
idx_to_char = {i: ch for i, ch in enumerate(vocab_list)}

# === 2. Model definition ===
class TinyGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.pos_embedding = nn.Embedding(seq_len, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, start_pos=0):
        positions = torch.arange(start_pos, start_pos + x.size(0), device=x.device).unsqueeze(1) % seq_len
        x = self.token_embedding(x) + self.pos_embedding(positions)
        x = x * (hidden_size ** 0.5)
        x = self.transformer(x)
        return self.lm_head(x)

model = TinyGPT()

# === 3. Load quantized weights ===
state_dict = model.state_dict()
with open(weights_file, "rb") as f:
    for k in state_dict.keys():
        numel = state_dict[k].numel()
        q = np.frombuffer(f.read(numel), dtype=np.int8).astype(np.float32) / 127.0
        state_dict[k] = torch.from_numpy(q.reshape(state_dict[k].shape))

model.load_state_dict(state_dict)
model.eval()

print(f"✅ Loaded model: {model_name}")
print(f"📚 Prompt: '{prompt}' | 🧠 top-k: {top_k} | ✏️ generate: {num_generate} tokens")

# === 4. Sampling function ===
def top_k_sampling(logits, k=1):
    values, indices = torch.topk(logits, k)
    probs = F.softmax(values, dim=0)
    idx = torch.multinomial(probs, num_samples=1)
    return indices[idx].item(), indices, probs

# === 5. Encode prompt ===
context = [char_to_idx.get(ch, char_to_idx[' ']) for ch in prompt.lower()]
context = torch.tensor(context, dtype=torch.long).unsqueeze(1)  # shape: [seq, 1]

print("\n🌱 Generating:\n")
print(prompt, end='', flush=True)

# === 6. Generation loop ===
for step in range(num_generate):
    input_seq = context[-seq_len:] if context.shape[0] >= seq_len else \
                torch.cat([torch.zeros(seq_len - context.shape[0], 1, dtype=torch.long), context], dim=0)

    start_pos = context.shape[0] - seq_len

    with torch.no_grad():
        logits = model(input_seq, start_pos=start_pos)
        next_logits = logits[-1, 0, :]

        if debug:
            print(f"\n--- Step {step} ---")
            print("Start pos:", start_pos)
            print("Input seq:", ''.join(idx_to_char[i.item()] for i in input_seq.squeeze()))
            print("Logits max/min:", next_logits.max().item(), next_logits.min().item())

        next_idx, top_indices, top_probs = top_k_sampling(next_logits, k=top_k)

        if debug and step < 5:
            print(f"Top-{top_k} tokens:")
            for rank in range(top_k):
                ch = idx_to_char[top_indices[rank].item()]
                p = top_probs[rank].item()
                print(f"  {rank+1}: '{ch}' ({p:.4f})")

    context = torch.cat([context, torch.tensor([[next_idx]], dtype=torch.long)], dim=0)
    print(idx_to_char[next_idx], end='', flush=True)

print("\n\n✅ Done.")
