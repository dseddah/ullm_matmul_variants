# === File Paths ===
TRAIN_SCRIPT = train_micro_llm.py
PREDICT_SCRIPT = interpreter_micro_llm_topk.py
WEIGHTS_FILE = tiny_llm_weights.bin
CONFIG_FILE = tiny_llm_config.json

# === Clean: remove weights and config ===
clean:
	rm -f $(WEIGHTS_FILE) $(CONFIG_FILE)

# === Train: run training script with optional ARGS ===
train:
	python $(TRAIN_SCRIPT) $(ARGS)

# === Predict: run interpreter with optional ARGS ===
predict:
	python $(PREDICT_SCRIPT) $(ARGS)
