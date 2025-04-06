# Config #######################################################################

OUT := out

SRCS := \
    $(OUT)/c-flags/lib/c-flags.c \
    $(OUT)/c-flags/lib/string-view.c \
    ullm/llama2.c \
    ullm/matmul_wrapper.mm \
    util/log.c \
    util/status.c \
    sys/file.c \
    sys/memory.c \
    sys/time.c \
    tools/ullm.c

OPT := 2

BIN := ullm

# build ########################################################################

BIN := $(OUT)/$(BIN)

OBJS := $(patsubst %.c, $(OUT)/%.o, $(filter %.c, $(SRCS))) \
        $(patsubst %.mm, $(OUT)/%.o, $(filter %.mm, $(SRCS)))

CFLAGS := \
    -I . \
    -I out/c-flags/lib \
    -std=c99 \
    -Wall \
    -O$(OPT)

MMFLAGS := \
    -std=c++17 \
    -fobjc-arc \
    -Wall \
    -O$(OPT)

LDFLAGS := \
    -lm \
    -framework Accelerate \
    -framework Metal \
    -framework Foundation

.PHONY:
all: $(BIN).elf
	size $(BIN).elf

.PHONY:
fetchdeps:
	@mkdir -p $(OUT)
	cd $(OUT); git clone https://github.com/DieTime/c-flags.git
	cd $(OUT); git clone https://github.com/karpathy/llama2.c.git
	cd $(OUT); curl -L -O https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

$(BIN).elf: ullm/matmul.metallib $(OBJS)
	clang++ $(CFLAGS) $(OBJS) -o $@ $(LDFLAGS)

# Compile C files
$(OUT)/%.o: %.c
	@mkdir -p $(dir $@)
	clang $(CFLAGS) -c $< -o $@

# Compile Objective-C++ files
$(OUT)/%.o: %.mm
	@mkdir -p $(dir $@)
	clang++ $(MMFLAGS) -c $< -o $@

# Compile Metal shader
ullm/matmul.metallib: ullm/matmul.metal
	xcrun -sdk macosx metal -c -o ullm/matmul.air $<
	xcrun -sdk macosx metallib -o $@ ullm/matmul.air

.PHONY:
clean:
	rm -rf $(OUT) ullm/*.air ullm/*.metallib

test:
	./out/ullm.elf -c out/stories15M.bin -t out/llama2.c/tokenizer.bin -p "The quick brown fox jumped. Where did he go?"
