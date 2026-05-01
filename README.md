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

## Testing Instructions

Run these checks before changing the editor protocol, inference engine, or frontend socket flow.

### Backend Unit Tests

```bash
source venv/bin/activate
pytest test_fim_prompt.py test_server_payload.py test_editor_state_manager.py
```

What these cover:
- `test_fim_prompt.py`: Qwen FIM prompt construction for empty text, cursor-at-end, mid-document insertion, multiline code, and suffix preservation.
- `test_server_payload.py`: WebSocket payload parsing and routing safety for `edit`, `autocomplete`, and `rewrite`.
- `test_editor_state_manager.py`: editor text/cache mutation behavior, truncation boundaries, and recovery cases.

### Python Compile Check

```bash
source venv/bin/activate
python -m py_compile engine.py server.py editor_state_manager.py test_fim_prompt.py test_server_payload.py
```

Use this after touching backend imports, ExLlamaV2 integration, or schema code.

### Frontend Checks

```bash
cd frontend
npm run lint
npm run build
```

### Manual Browser Tests

Start the backend and frontend, then open the editor in Chrome.

```bash
source venv/bin/activate
python server.py
```

```bash
cd frontend
npm run dev -- --host 0.0.0.0
```

Run these manual checks:

1. Type text at the end of a line, pause, and confirm ghost text appears inline.
2. Press `Tab` while ghost text is visible and confirm it commits as normal editable text.
3. Type any character while ghost text is visible and confirm the ghost text disappears before the typed character lands.
4. Click to a different line while generation is streaming and confirm the old ghost text disappears.
5. Select all text with `Ctrl+A`, press `Backspace`, and confirm the backend recovers without an exception.
6. Highlight a phrase, use the floating rewrite toolbar, and confirm rewrite chunks replace only the selected text.
7. Cancel the rewrite toolbar and confirm the selection clears without sending a rewrite request.

### WebSocket Payload Checks

Open browser DevTools and watch the WebSocket frames.

Expected fast-path payload:

```json
{"action":"autocomplete","new_text":"example text","edit_char_index":12}
```

Expected rewrite payload:

```json
{"action":"rewrite","text":"selected text","prompt":"Make this sound more professional"}
```

Expected stream responses:

```json
{"type":"token","chunk":"..."}
{"type":"done"}
```

### GPU Telemetry Checks

When using the real engine, every completed request should print a terminal log.

Fast path:

```text
[FAST-PATH] TTFT: 38ms | TPS: 45.2 | Total Time: 210ms | Tokens: 12
```

Heavy path:

```text
[HEAVY-PATH] TTFT: 120ms | TPS: 32.5 | Draft Acceptance: 78% | Total Time: 1850ms | Tokens: 128
```

Acceptance targets for Phase 6:
- Fast-path TTFT should stay near the sub-40ms target on an RTX 4090 with the 1.5B Qwen model.
- Rewrite should stream steadily without blocking the WebSocket.
- Draft acceptance should be non-zero when ExLlamaV2 exposes speculative stats for the dynamic generator.
- No `Assistant`, `Dear user`, chat-role leakage, repeated-character collapse, or FIM token leakage should appear in ghost text.

### TabbyAPI A/B Test

Use this when autocomplete quality is suspicious. It sends the same FIM prompt to TabbyAPI so you can compare model/runtime behavior against the custom backend.

```bash
source venv/bin/activate
python tabby_ab_test.py --prefix $'def add(a, b):\n    ' --suffix $'\n'
```

Interpretation:
- Tabby good, custom backend bad: debug ExLlama integration, cache, tokenization, stopping, or sampling.
- Both bad: model, quant, or generation settings are the likely cause.
- Both good: frontend/backend wiring is probably healthy.

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

For an RTX 3090/4090, CUDA should be visible and the device name should include the GPU model.

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

### Check Flash Attention for speculative rewrite

ExLlamaV2's dynamic/speculative generator requires paged attention. Install Flash Attention 2.5.7+ in the same venv and verify paged attention support:

```bash
source venv/bin/activate
pip install -U packaging ninja
pip install "flash-attn>=2.5.7" --no-build-isolation

python - <<'PY'
import flash_attn
from exllamav2 import attn

print("flash_attn:", flash_attn.__version__)
print("paged attention available:", attn.has_flash_attn_with_paged)
PY
```

Expected:

```text
paged attention available: True
```

If installation is slow or fails, use a prebuilt wheel matching your Python, PyTorch, and CUDA versions.

### Cache mode notes

- Qwen autocomplete uses an 8-bit KV cache to keep the fast path small.
- Llama rewrite target/draft use Q4 KV caches because ExLlamaV2's dynamic/speculative generator rejects 8-bit caches.
- Speculative rewrite uses the synchronous dynamic job queue: `ExLlamaV2DynamicJob`, `generator.enqueue(job)`, and `generator.iterate()`.
- If startup fails with `Dynamic generator does not currently work with 8-bit cache`, make sure the code is using the rewrite Q4 cache path and restart the backend.

### Recommended stack notes

- GPU: NVIDIA RTX 3090 or RTX 4090
- Python: 3.10 or 3.11 is usually safest for CUDA inference packages
- PyTorch: install a CUDA build, not CPU-only PyTorch
- CUDA: match PyTorch and ExLlamaV2 wheel expectations
- Known working baseline from a Vast.ai/Salad-style RTX 3090 instance:
  - PyTorch: `2.5.1+cu121`
  - Torch CUDA: `12.1`
  - GPU: `NVIDIA GeForce RTX 3090`
  - `torch.cuda.is_available()`: `True`
- Flash Attention: `2.5.7+` is required for ExLlamaV2 dynamic/speculative rewrite
- Model: use a base Qwen2.5-Coder EXL2 quant for Phase 5 autocomplete

### Common symptoms

- `torch.cuda.is_available() == False`: wrong PyTorch build, missing driver, or container does not expose the GPU.
- ExLlamaV2 import/build errors: Python/PyTorch/CUDA wheel mismatch.
- `Paged attention required Flash Attention 2.5.7 or later`: install or upgrade `flash-attn` inside the active venv.
- `Dynamic generator does not currently work with 8-bit cache`: rewrite target/draft must use Q4 or FP16 cache; the current backend uses Q4 for rewrite.
- `ExLlamaV2DynamicJobAsync object has no attribute prepare_for_queue`: use the synchronous dynamic job queue path; async jobs can be incompatible with the installed ExLlamaV2 release.
- `KeyError: 'cuda:0'`: cache/model split or device registration issue; verify model loading and ExLlamaV2 version.
- Repeated nonsense tokens: test the same FIM prompt through TabbyAPI; if Tabby also fails, suspect model/quant/settings.
- `Assistant:` or `Dear user` in output: you are probably using an instruct/chat model or a chat prompt path, not FIM with a base Coder model.
- Good end-of-line results but bad midline results: confirm the frontend sends the real cursor index and the backend builds a FIM prompt with both prefix and suffix.
