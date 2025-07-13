import torch
import torch.nn.functional as F
import numpy as np
import argparse
import json
import os
from model_micro_llm import TinyGPT

# === CLI arguments ===
parser = argparse.ArgumentParser(description="Micro LLM Inference with Top-k/Top-p Sampling and Temperature.")
parser.add_argument("--k", type=int, default=0, help="Top-k sampling (0 = disable)")
parser.add_argument("--top_p", type=float, default=0.0, help="Top-p (nucleus) sampling (0.0 = disable)")
parser.add_argument("--temperature", type=float, default=1.0, help="Softmax temperature (default: 1.0)")
parser.add_argument("--prompt", type=str, default="the ", help="Prompt to start generation")
parser.add_argument("--generate", type=int, default=200, help="Number of characters to generate")
parser.add_argument("--model_name", type=str, default="tiny_llm", help="Base name of model files")
parser.add_argument("--debug", action="store_true", help="Print logits/debug info per step")
parser.add_argument("--debug_softmax", action="store_true", help="Print top softmax probabilities")
args = parser.parse_args()

# === Load model config ===
config_path = f"{args.model_name}_config.json"
weights_path = f"{args.model_name}_weights.bin"

if not os.path.exists(config_path):
    raise FileNotFoundError(f"Config file not found: {config_path}")
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Weights file not found: {weights_path}")

with open(config_path, "r") as f:
    config = json.load(f)

vocab = config["vocab_list"]
char_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_char = {i: ch for i, ch in enumerate(vocab)}

model = TinyGPT(
    vocab_size=config["vocab_size"],
    hidden_size=config["hidden_size"],
    seq_len=config["seq_len"],
    num_layers=config["num_layers"],
    num_heads=config["num_heads"]
)

# === Load weights ===
state_dict = model.state_dict()
with open(weights_path, "rb") as f:
    for k in state_dict:
        numel = state_dict[k].numel()
        arr = np.frombuffer(f.read(numel * 4), dtype=np.float32).copy()
        state_dict[k] = torch.from_numpy(arr.reshape(state_dict[k].shape))
model.load_state_dict(state_dict)
model.eval()

print(f"✅ Loaded model: {args.model_name}")
print(f"📚 Prompt: '{args.prompt}' | top-k: {args.k} | top-p: {args.top_p} | temp: {args.temperature} | gen: {args.generate}")

# === Sampling function ===
def sample_token(logits, k=0, p=0.0, temperature=1.0):
    logits = logits / temperature
    logits = logits.clone()

    if k > 0:
        top_values, top_indices = torch.topk(logits, k)
        mask = torch.full_like(logits, float('-inf'))
        mask[top_indices] = top_values
        logits = mask

    if p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)
        cutoff = (cumulative_probs > p).nonzero(as_tuple=False)
        if len(cutoff) > 0:
            cutoff_idx = cutoff[0, 0]
            sorted_logits[cutoff_idx + 1:] = float('-inf')
        logits = torch.full_like(logits, float('-inf'))
        logits[sorted_indices] = sorted_logits

    probs = F.softmax(logits, dim=-1)
    sampled_idx = torch.multinomial(probs, 1).item()
    top_probs, top_indices = torch.topk(probs, 10)
    return sampled_idx, top_probs, top_indices

# === Encode prompt ===
context = [char_to_idx.get(ch, char_to_idx[' ']) for ch in args.prompt.lower()]
context = torch.tensor(context, dtype=torch.long).unsqueeze(1)  # [seq, 1]

print("\n🌱 Generating:\n")
print(args.prompt, end='', flush=True)

# === Generation loop ===
for step in range(args.generate):
    input_seq = context[-config["seq_len"]:]
    if input_seq.shape[0] < config["seq_len"]:
        pad = torch.zeros(config["seq_len"] - input_seq.shape[0], 1, dtype=torch.long)
        input_seq = torch.cat([pad, input_seq], dim=0)

    start_pos = context.shape[0] - config["seq_len"]

    with torch.no_grad():
        logits = model(input_seq, start_pos=start_pos)
        next_logits = logits[-1, 0, :]

        next_idx, top_probs, top_indices = sample_token(
            next_logits, k=args.k, p=args.top_p, temperature=args.temperature
        )

        if args.debug_softmax:
            print("\n📊 Softmax top-10:")
            for i in range(10):
                print(f"  {idx_to_char[top_indices[i].item()]}: {top_probs[i].item():.4f}")

        if args.debug:
            print(f"\n--- Step {step} ---")
            print("Input:", ''.join(idx_to_char[i.item()] for i in input_seq.squeeze()))
            print("Logits range:", next_logits.min().item(), "to", next_logits.max().item())

    context = torch.cat([context, torch.tensor([[next_idx]], dtype=torch.long)], dim=0)
    print(idx_to_char[next_idx], end='', flush=True)

print("\n\n✅ Done.")
