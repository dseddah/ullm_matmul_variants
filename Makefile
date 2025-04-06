################################################################################
#
# POSIX Makefile
#
################################################################################

# Config #######################################################################

OUT := out

SRCS := \
    $(OUT)/c-flags/lib/c-flags.c \
    $(OUT)/c-flags/lib/string-view.c \
    ullm/llama2.c \
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

OBJS := $(patsubst %.c, $(OUT)/%.o, $(SRCS))

CFLAGS := \
    -I . \
    -I out/c-flags/lib \
    -std=c99 \
    -Wall \
    -O$(OPT) \
    -O3 

LDFLAGS := \
    -lm -framework Accelerate







.PHONY:
all: $(BIN).elf
	size $(BIN).elf

.PHONY:
fetchdeps:
	@mkdir -p $(OUT)
	cd $(OUT); git clone https://github.com/DieTime/c-flags.git
	cd $(OUT); git clone https://github.com/karpathy/llama2.c.git
	cd $(OUT); curl -L -O https://huggingface.co/karpathy/tinyllamas/resolve/main/stories15M.bin

$(BIN).elf: $(OBJS)
	cc $(CFLAGS) $^ $(LDFLAGS) -o $@

$(OUT)/%.o: %.c
	@mkdir -p $(dir $@)
	cc $(CFLAGS) -c $< -o $@

.PHONY:
clean:
	rm -rf $(OUT)


test:
	./out/ullm.elf -c out/stories15M.bin -t out/llama2.c/tokenizer.bin -p "The quick brown fox jumped. Where did he go?"
	
