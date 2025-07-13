import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import argparse
import json
import os
from model_micro_llm import TinyGPT

# === 0. CLI arguments ===
parser = argparse.ArgumentParser(description="Micro LLM Top-k / Top-p Sampler with temperature.")
parser.add_argument("--k", type=int, default=0, help="Top-k sampling value (0 to disable)")
parser.add_argument("--top_p", type=float, default=0.0, help="Top-p (nucleus) sampling value (0 to disable)")
parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature (default: 1.0)")
parser.add_argument("--prompt", type=str, default="the ", help="Prompt for generation")
parser.add_argument("--generate", type=int, default=200, help="Number of characters to generate")
parser.add_argument("--model_name", type=str, default="tiny_llm", help="Base name of model files")
parser.add_argument("--debug", action="store_true", help="Enable debug output")
parser.add_argument("--debug_softmax", action="store_true", help="Print top softmax values")
args = parser.parse_args()

top_k = args.k
top_p = args.top_p
temperature = args.temperature
prompt = args.prompt
num_generate = args.generate
model_name = args.model_name
debug = args.debug
debug_softmax = args.debug_softmax

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

# === 2. Load model ===
model = TinyGPT(vocab_size, hidden_size, seq_len, num_layers, num_heads)
state_dict = model.state_dict()
with open(weights_file, "rb") as f:
    for k in state_dict:
        numel = state_dict[k].numel()
        q = np.frombuffer(f.read(numel * 4), dtype=np.float32)  # float32
        state_dict[k] = torch.from_numpy(q.reshape(state_dict[k].shape))
model.load_state_dict(state_dict)
model.eval()

print(f"✅ Loaded model: {model_name}")
print(f"📚 Prompt: '{prompt}' | 🧠 top-k: {top_k} | 🔮 top-p: {top_p} | 🌡 temp: {temperature} | ✏️ generate: {num_generate}")

# === 3. Sampling function ===
def sample_token(logits, k=0, p=0.0, temperature=1.0):
    logits = logits / temperature
    logits = logits.clone()

    if k > 0:
        values, indices = torch.topk(logits, k)
        logits[:] = float('-inf')
        logits[indices] = values

    if p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(probs, dim=-1)
        mask = cumulative > p
        if torch.any(mask):
            cutoff = mask.nonzero()[0].item()
            sorted_logits[cutoff + 1:] = float('-inf')
        logits[:] = float('-inf')
        logits[sorted_indices] = sorted_logits

    probs = F.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, 1).item()
    return idx, torch.topk(probs, 10)

# === 4. Encode prompt ===
context = [char_to_idx.get(ch, char_to_idx[' ']) for ch in prompt.lower()]
context = torch.tensor(context, dtype=torch.long).unsqueeze(1)  # [seq, 1]

print("\n🌱 Generating:\n")
print(prompt, end='', flush=True)

# === 5. Generation loop ===
for step in range(num_generate):
    input_seq = context[-seq_len:] if context.shape[0] >= seq_len else \
        torch.cat([torch.zeros(seq_len - context.shape[0], 1, dtype=torch.long), context], dim=0)

    start_pos = context.shape[0] - seq_len

    with torch.no_grad():
        logits = model(input_seq, start_pos=start_pos)
        next_logits = logits[-1, 0, :]

        next_idx, (top_probs, top_indices) = sample_token(next_logits, k=top_k, p=top_p, temperature=temperature)

        if debug_softmax:
            print("\n📊 Sampled softmax probs (top 10):")
            for i in range(10):
                print(f"  {idx_to_char[top_indices[i].item()]}: {top_probs[i].item():.4f}")

        if debug:
            print(f"\n--- Step {step} ---")
            print("Start pos:", start_pos)
            print("Input seq:", ''.join(idx_to_char[i.item()] for i in input_seq.squeeze()))
            print("Logits max/min:", next_logits.max().item(), next_logits.min().item())
            print("Logits (sample):", next_logits[:10].cpu().numpy())

    context = torch.cat([context, torch.tensor([[next_idx]], dtype=torch.long)], dim=0)
    print(idx_to_char[next_idx], end='', flush=True)

print("\n\n✅ Done.")
