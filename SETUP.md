# SETUP.md — Quick Environment Setup

## Prerequisites
- Python 3.11+ (tested on 3.14)
- Windows / Linux / macOS
- Git

## 1. Create Virtual Environment (Isolated — Safe for your System)

```bash
# From project root
python -m venv venv_mappo

# Activate (Windows)
venv_mappo\Scripts\activate

# Activate (Linux/macOS)
source venv_mappo/bin/activate
```

## 2. Install Core Dependencies

```bash
pip install numpy tqdm matplotlib trueskill pytest scipy
```

## 3. Install PyTorch

**CPU only (always works, no GPU required):**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

**GPU (CUDA 12.x) — if you have an NVIDIA GPU:**
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Verify:
```bash
python -c "import torch; print(torch.__version__, '| CUDA:', torch.cuda.is_available())"
```

## 4. Verify Installation

```bash
python -m pytest tests/ -q
# Expected: 70 passed in ~5-7s
```

## 5. Activate Environment (reminder for every new terminal session)

```bash
# Windows
venv_mappo\Scripts\activate

# Linux/macOS
source venv_mappo/bin/activate
```

## Notes

- `venv_mappo/` is created inside the project directory — fully isolated.
- **Pygame** fails to install on Python 3.14 (no pre-built wheel). This only affects the GIF visualizer, not training or agent submission.
- The evaluation server uses CPU-only. The submission agent is always CPU-safe.
- All training scripts auto-detect GPU via `device="auto"`.
