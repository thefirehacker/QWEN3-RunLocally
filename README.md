# Models

## Qwen 3 0.6B (Pure C)

Minimal C inference for Qwen 3 0.6B on Mac (no CUDA). Code lives in [repos/qwen3.c](repos/qwen3.c) (from [thefirehacker/qwen3.c](https://github.com/thefirehacker/qwen3.c)).

### Get the code

Clone this repo with submodules so you get the qwen3.c code:

```bash
git clone --recurse-submodules <url-of-this-repo>
```

If you already cloned without submodules:

```bash
git submodule update --init
```

### Build and run (Mac)

From the repo root:

```bash
cd repos/qwen3.c
# Download FP32 model from Hugging Face (one-time)
git clone https://huggingface.co/huggit0000/Qwen3-0.6B-GGUF-FP32
mv Qwen3-0.6B-GGUF-FP32/Qwen3-0.6B-FP32.gguf ./
# Build and run
make run
./run Qwen3-0.6B-FP32.gguf
```

Faster inference with OpenMP (set threads to your core count):

```bash
make runomp
OMP_NUM_THREADS=8 ./run Qwen3-0.6B-FP32.gguf
```

Options: multi-turn chat `-m 1`, reasoning mode `-k 1`. See [qwen3.c Quick Start](https://github.com/thefirehacker/qwen3.c#quick-start) for details.