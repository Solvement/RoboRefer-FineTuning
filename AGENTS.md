# AGENTS.md

## Cursor Cloud specific instructions

### Project overview

RoboRefer-FineTuning is a Python ML/research codebase built on the VILA v2.0.0 framework (by NVIDIA). It fine-tunes a Vision-Language Model (NVILA-Lite-2B / Qwen2 backbone) on ScanNet indoor scene data for 3D spatial reasoning. The Python package is named `vila` (installed from `pyproject.toml`) but the source code lives under the `llava/` directory.

### Known issue: missing VILA base files

Several core files from the VILA framework are **not committed** to this repository:
- `llava/__init__.py`
- `llava/conversation.py`
- `llava/media.py`
- `llava/constants.py`
- `llava/mm_utils.py`

This means `vila-infer`, the Flask API server (`API/api.py`), training scripts, and evaluation scripts that import from these modules will fail at import time. The `vila-run` and `vila-eval` CLI entry points work since they don't depend on these missing modules at the help/CLI level.

### Dependencies

Install with `pip install -e ".[train,eval]"` from the repo root. This installs ~330 pinned dependencies including PyTorch 2.5.1 (CUDA 12.4), vLLM, xformers, DeepSpeed, etc. All packages install on CPU-only environments but GPU operations require NVIDIA GPUs.

### Linting

- `python3 -m black --check llava/` (line length 120, configured in `pyproject.toml`)
- `python3 -m isort --check-only llava/` (profile "black", configured in `pyproject.toml`)
- `python3 -m flake8 llava/ --max-line-length 120`

The codebase has existing formatting issues (64 files flagged by black, 374 flake8 warnings). These are pre-existing.

### Testing

No formal pytest test suite exists. The repo has evaluation scripts (`test_fiveframes_crossview.py`, `Evaluation/test_benchmark.py`) that require model weights and datasets at runtime.

### CLI entry points

- `vila-run` — Slurm job launcher
- `vila-eval` — Model evaluation runner
- `vila-infer` — Interactive inference (requires missing VILA base files)
- `vila-upload` — HuggingFace Hub uploader

Add `/home/ubuntu/.local/bin` to `PATH` to access these commands.

### GPU requirement

Training, inference, and evaluation require NVIDIA GPUs (designed for RTX 6000 Ada with 48GB VRAM). The Flask API (`API/api.py`) and all model-loading code require CUDA. CPU-only environments can install dependencies, run linting, and exercise data processing utilities but cannot run model inference.
