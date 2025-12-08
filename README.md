<p align="center">
<img width="300" src="assets/logo.png">
</p>

<p align="center">
<a href="https://trendshift.io/repositories/15323" target="_blank"><img src="https://trendshift.io/api/badge/repositories/15323" alt="GeeeekExplorer%2Fnano-vllm | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</p>

# Nano-vLLM (W/ best installation practice by wln)
> > NOTE: This is only a personal note!

A lightweight vLLM implementation built from scratch.

## Key Features

* 🚀 **Fast offline inference** - Comparable inference speeds to vLLM
* 📖 **Readable codebase** - Clean implementation in ~ 1,200 lines of Python code
* ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graph, etc.

## Installation & Demo, Best practice by wln

It's hard to successfully build and install `flash-attn`! After some trials I finally found out the correct way. 

**Key points:**
- Use [uv](https://docs.astral.sh/uv/) to download packages like `torch` very quickly.
- Separately build and install `flash-attn` before installing the main project.
  Before building `flash-attn`, make sure to install some packages in advance:
  - `ninja` for fast `flash-attn` building process.
  - `torch==2.8`, `psutil`, `packaging` as building dependencies. Seems that only `torch>=2.2, <=2.8` work (up to 2025.12). 
  And make sure to add `--no-build-isolation` when `pip install flash-attn`, otherwise errors like "No module named 'torch'" might occur.
- After installing flash-attn and before the installation of main project, remove the related dependencies (`torch`, `triton`, `flash-attn`) in its `pyproject.toml`, or there might be some forced reinstall and bring conflicts.


**Practice:**
When get a new machine, save the following content in `~/start.sh` and run `bash ~/start.sh` to install & verify if nano-vllm works as expected:

```bash
# ===== BEGIN INSTALLATION =====
# install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# clone nano-vllm (with modified pyproject.toml)
git clone https://github.com/GeeeekExplorer/nano-vllm.git

# create venv and activate it
cd nano-vllm
uv venv --python 3.12 --seed
source .venv/bin/activate

# first install flash-attn separately
uv pip install ninja setuptools build psutil packaging
uv pip install torch==2.8 torchvision
uv pip install flash-attn --no-build-isolation

# finally install nano-vllm with the modified pyproject.toml
rm pyproject.toml
cat > pyproject.toml <<EOF
[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "nano-vllm"
version = "0.2.0"
authors = [{ name = "Xingkai Yu" }]
license = "MIT"
license-files = ["LICENSE"]
readme = "README.md"
description = "a lightweight vLLM implementation built from scratch"
requires-python = ">=3.10,<3.13"
dependencies = [
    "transformers>=4.51.0",
    "xxhash",
]

[project.urls]
Homepage="https://github.com/GeeeekExplorer/nano-vllm"

[tool.setuptools.packages.find]
where = ["."]
include = ["nanovllm*"]
EOF

uv pip install -e .

# ===== END INSTALLATION =====

# ===== BEGIN DEMO =====
# download model
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir ~/huggingface/Qwen3-0.6B/ \
  --local-dir-use-symlinks False

# create and run demo
cat > test.py <<EOF
from nanovllm import LLM, SamplingParams
import os

model_path = os.path.join(os.path.expanduser("~"), "huggingface", "Qwen3-0.6B")
llm = LLM(model_path, enforce_eager=True, tensor_parallel_size=1, dtype="float16")
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hi, how are you today?", "你好！我是"]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])
print("="*20)
print(outputs[1]["text"])
EOF

uv run python test.py

# ===== END DEMO =====
```




## Benchmark

See `bench.py` for benchmark.

**Test Configuration:**
- Hardware: RTX 4070 Laptop (8GB)
- Model: Qwen3-0.6B
- Total Requests: 256 sequences
- Input Length: Randomly sampled between 100–1024 tokens
- Output Length: Randomly sampled between 100–1024 tokens

**Performance Results:**
| Inference Engine | Output Tokens | Time (s) | Throughput (tokens/s) |
|----------------|-------------|----------|-----------------------|
| vLLM           | 133,966     | 98.37    | 1361.84               |
| Nano-vLLM      | 133,966     | 93.41    | 1434.13               |


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=GeeeekExplorer/nano-vllm&type=Date)](https://www.star-history.com/#GeeeekExplorer/nano-vllm&Date)
