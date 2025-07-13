# === File Paths ===
TRAIN_SCRIPT = train_micro_llm.py
PREDICT_SCRIPT = interpreter_micro_llm_topk.py
OVERFIT_SCRIPT = overfit_micro_llm.py
WEIGHTS_FILE = tiny_llm_weights.bin
CONFIG_FILE = tiny_llm_config.json

# === Clean: remove weights and config ===
clean:
	rm -f $(WEIGHTS_FILE) $(CONFIG_FILE)
	rm -f debug_weights.bin debug_config.json

# === Train: run training script with optional ARGS ===
train:
	python $(TRAIN_SCRIPT) $(ARGS)

# === Predict: run interpreter with optional ARGS ===
predict:
	python $(PREDICT_SCRIPT) $(ARGS)
	
	
full: train predict
	

# === Overfit: train tiny model and test it ===
overfit:
	@echo "🚧 Running overfit test (model: debug)..."
	python $(OVERFIT_SCRIPT) --model_name debug
	@echo "🧠 Testing debug model with prompt 'the':"
	python $(PREDICT_SCRIPT) --model_name debug --prompt "the " --k 1 --generate 100


overfit_real:
		@echo "🚧 Running overfit test from the main script (model: debug)..."
		python $(TRAIN_SCRIPT) --model_name debug --epochs 1000 --train_file overfit_tiny.txt\
		--val_file overfit_tiny_2.txt --model_name debug_overfit $(ARGS)
		@echo "🧠 Testing debug model with prompt 'the':"
		python $(PREDICT_SCRIPT) --model_name debug_overfit --k 1 --prompt "the " --generate 100	
		
overfit_real_old:
		@echo "🚧 Running overfit test from the main script (model: debug)..."
		python $(TRAIN_SCRIPT).old  --epochs 1000 --train_file overfit_tiny.txt\
		--val_file overfit_tiny_2.txt --model_name debug_overfit_old $(ARGS)
		@echo "🧠 Testing debug model with prompt 'the':"
		python $(PREDICT_SCRIPT) --model_name debug_overfit_old --k 1 --prompt "the " --generate 100	