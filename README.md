# Real-Time FIM Autocomplete Editor

React + FastAPI editor prototype with Lexical ghost text and an ExLlamaV2-backed Qwen Fill-in-the-Middle (FIM) autocomplete path.

The fast path is designed for low-latency inline completion on an RTX 4090 using a small **base** code model, not an instruct/chat model.

Recommended model:
- [Qwen/Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B)

## What This Runs

- Backend: FastAPI WebSocket server at `ws://127.0.0.1:8000/ws/editor`
- Frontend: Vite React app at `http://127.0.0.1:5173`
- Editor: Lexical plain-text editor with ghost-text autocomplete
- Autocomplete prompt format:

```text
<|fim_prefix|>{text_before_cursor}<|fim_suffix|>{text_after_cursor}<|fim_middle|>
```

## Ubuntu Setup

### 1. Install system tools

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip nodejs npm git
```

Use Python 3.10+ and Node 18+ if possible.

### 2. Clone and enter the repo

```bash
git clone <repo-url>
cd <repo-dir>
```

### 3. Create the Python environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

For real GPU inference, install ExLlamaV2 in the same venv:

```bash
pip install exllamav2
```

If your CUDA/PyTorch environment needs a specific ExLlamaV2 wheel, install the wheel recommended for your CUDA and PyTorch versions instead of the generic PyPI package.

### 4. Install frontend dependencies

```bash
cd frontend
npm install
cd ..
```

### 5. Download the model

Download a Qwen2.5-Coder **base** EXL2 quant into a local model directory. Start with 1.5B for the Phase 5 latency target.

Model card:
- [Qwen/Qwen2.5-Coder-1.5B](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B)

Set the path to your downloaded EXL2 model directory in `.env`.

## Configuration

Create `.env` from the example:

```bash
cp .env.example .env
```

For mock CPU development:

```env
USE_MOCK_ENGINE=true
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
```

For real ExLlamaV2 inference:

```env
USE_MOCK_ENGINE=false
EXLLAMA_MODEL_DIR=/absolute/path/to/Qwen2.5-Coder-1.5B-EXL2
EDITOR_TOKENIZER_MODEL=/absolute/path/to/Qwen2.5-Coder-1.5B-EXL2
SERVER_HOST=127.0.0.1
SERVER_PORT=8000
GENERATION_TEMPERATURE=0.2
GENERATION_TOP_P=0.9
GENERATION_TOP_K=40
GENERATION_REPETITION_PENALTY=1.05
```

If the EXL2 folder does not include tokenizer files, set `EDITOR_TOKENIZER_MODEL` to a compatible Hugging Face tokenizer source, such as `Qwen/Qwen2.5-Coder-1.5B`.

## Run The Program

Open two terminals.

### Terminal 1: backend

```bash
source venv/bin/activate
python server.py
```

Expected backend URL:

```text
http://127.0.0.1:8000
```

### Terminal 2: frontend

```bash
cd frontend
npm run dev -- --host 0.0.0.0
```

Open:

```text
http://127.0.0.1:5173
```

Start typing in the editor and pause briefly. The frontend sends debounced `edit` and `autocomplete` WebSocket payloads. Token chunks stream back as ghost text.

## Remote / Tunnel Notes

If the frontend is accessed through a tunnel, browser `localhost` means the browser's machine, not the GPU machine.

Preferred setup:
- Run backend on the GPU machine at port `8000`
- Run Vite on the GPU machine at port `5173`
- Tunnel the frontend port
- Let Vite proxy `/ws/editor` to the backend

If using separate frontend and backend tunnels, start the frontend with:

```bash
VITE_EDITOR_WS_URL=wss://<backend-tunnel-host>/ws/editor npm run dev -- --host 0.0.0.0
```

## Tests

Run backend tests:

```bash
source venv/bin/activate
pytest test_fim_prompt.py test_editor_state_manager.py
```

Run frontend checks:

```bash
cd frontend
npm run lint
npm run build
```

## TabbyAPI A/B Probe

Use this to compare TabbyAPI against the custom backend with the exact same FIM prompt builder.

Start TabbyAPI with the same Qwen2.5-Coder base EXL2 model, then run:

```bash
source venv/bin/activate
python tabby_ab_test.py --prefix $'def add(a, b):\n    ' --suffix $'\n'
```

Optional environment variables:

```bash
export TABBY_API_URL=http://127.0.0.1:5000/v1/completions
export TABBY_MODEL=<model-name-if-required>
```

Interpretation:
- Tabby good, custom backend bad: debug ExLlama integration, cache, tokenization, or stopping.
- Both bad: model, quant, or sampling settings are the likely cause.
- Both good: the editor bridge is healthy.

## Key Files

- `server.py`: FastAPI WebSocket server and generation task orchestration
- `engine.py`: mock engine, real ExLlamaV2 engine, FIM prompt builder
- `editor_state_manager.py`: character edit to token/cache state manager
- `frontend/src/Editor.tsx`: Lexical editor and WebSocket wiring
- `frontend/src/AutocompletePlugin.tsx`: ghost text insertion, accept, dismiss
- `tabby_ab_test.py`: OpenAI-compatible TabbyAPI comparison probe

## Current Model Guidance

Use a base code-completion model for Phase 5:

1. First choice: `Qwen2.5-Coder-1.5B` base EXL2
2. Fallback: `Qwen2.5-Coder-3B` base EXL2 if quality is too weak and TTFT remains acceptable
3. Avoid 7B/14B for Phase 5 fast ghost text unless latency budget changes
4. Avoid instruct/chat models for inline autocomplete; they tend to emit assistant-style responses

Heavy rewrite and reasoning paths belong in a later phase with a larger model.

## Debugging Environment

Use these commands on the GPU machine when inference quality, CUDA loading, or ExLlamaV2 behavior looks suspicious.

### Check NVIDIA driver and CUDA runtime

```bash
nvidia-smi
nvcc --version
```

`nvidia-smi` shows the installed driver and the maximum CUDA runtime supported by that driver. `nvcc --version` only exists if the CUDA toolkit is installed; it is not required for every runtime setup.

### Check Python, PyTorch, and CUDA visibility

```bash
source venv/bin/activate
python - <<'PY'
import torch

print("torch:", torch.__version__)
print("torch cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device 0:", torch.cuda.get_device_name(0))
    print("capability:", torch.cuda.get_device_capability(0))
PY
```

For an RTX 4090, CUDA should be visible and the device name should include `4090`.

### Check ExLlamaV2 import

```bash
source venv/bin/activate
python - <<'PY'
import exllamav2
print("exllamav2:", getattr(exllamav2, "__version__", "unknown"))

from exllamav2 import ExLlamaV2, ExLlamaV2Tokenizer, ExLlamaV2Config
from exllamav2.generator import ExLlamaV2Sampler, ExLlamaV2StreamingGenerator
print("imports: ok")
PY
```

If this fails, install a wheel that matches your Python, PyTorch, and CUDA stack. ExLlamaV2 wheels are sensitive to those versions.

### Recommended stack notes

- GPU: NVIDIA RTX 4090
- Python: 3.10 or 3.11 is usually safest for CUDA inference packages
- PyTorch: install a CUDA build, not CPU-only PyTorch
- CUDA: match PyTorch and ExLlamaV2 wheel expectations
- Model: use a base Qwen2.5-Coder EXL2 quant for Phase 5 autocomplete

### Common symptoms

- `torch.cuda.is_available() == False`: wrong PyTorch build, missing driver, or container does not expose the GPU.
- ExLlamaV2 import/build errors: Python/PyTorch/CUDA wheel mismatch.
- `KeyError: 'cuda:0'`: cache/model split or device registration issue; verify model loading and ExLlamaV2 version.
- Repeated nonsense tokens: test the same FIM prompt through TabbyAPI; if Tabby also fails, suspect model/quant/settings.
- `Assistant:` or `Dear user` in output: you are probably using an instruct/chat model or a chat prompt path, not FIM with a base Coder model.
- Good end-of-line results but bad midline results: confirm the frontend sends the real cursor index and the backend builds a FIM prompt with both prefix and suffix.
