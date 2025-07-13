# 🧠 Micro LLM with Positional Embeddings + Quantized Weights

> ✨ Built interactively with the help of [ChatGPT](https://openai.com/chatgpt) during an extended prototyping and debugging session.  
> All architecture, training logic, and tooling decisions were refined through a collaborative loop.


---

## 📁 Project Structure


my_micro_llm_project/
├── train_micro_llm.py # Generic training script (configurable via CLI)
├── overfit_micro_llm.py # Tiny hardcoded dataset to force overfitting
├── interpreter_micro_llm_topk.py # Inference script with top-k sampling and debug
├── micro_llm_model.py # TinyGPT model definition
├── micro_llm_utils.py # Vocab, encoding, cleaning utilities
├── Makefile # clean, train, predict, overfit targets
├── train_tiny.txt # Main training text
├── val_tiny.txt # Validation text
├── tiny_llm_weights.bin # Exported quantized weights
├── tiny_llm_config.json # Model configuration



---

## 🚀 Quickstart

### 🏗️ Train
```bash
make train ARGS="--epochs 2000 --hidden_size 64 --seq_len 64"


make overfit

### Inference

make predict ARGS="--prompt 'the ' --k 3 --generate 200"


###  Features

✅ Positional Embeddings
✅ Configurable CLI training (--hidden_size, --seq_len, etc.)
✅ Quantized int8 model export
✅ Overfit sanity check
✅ Debug mode with top-k sampling insights


### To Extend

Add dropout, layer norm, or FFN expansion
Swap to byte-level vocab
Export to Atari, FPGA, or retro platforms 😄



### For the super curious
here's the chatGPT conversation, it started to forget things right before I
asked if he was running out of context (10% before the end)

here's the conversation link
https://chatgpt.com/c/686e8b4f-2e60-8007-b5e0-18167946fe61

