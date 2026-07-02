# AGENTS.md

## Commands

| Action | Command |
|---|---|
| Run app | `source env313/bin/activate && streamlit run app.py` |
| Test model (no UI) | `python debug_model.py` |

## Key gotchas

- **Label swap**: Model trained `REAL=1, FAKE=0`. `INVERT_PROBABILITY = True` in `src/config.py` corrects at inference. After inversion: `≥ 0.5 → FAKE`, `< 0.5 → REAL`. Metrics use `Real=0, Fake=1`.
- **`debug_model.py` is standalone**: It does **not** import from `src/` and does **not** apply `INVERT_PROBABILITY`. Its output is the raw (non-inverted) prediction.
- **CPU only + 8 GB RAM**: `MAX_BATCH = 100` per session. All inference is on CPU.
- **Preprocessing**: No face detection, mean subtraction, or TTA. Only `RGB → resize 256×256 → /255.0`. Placeholder code gated behind `False` flags in `src/config.py`.
- **Model load dry-run**: `load_model()` (`src/model_utils.py:15`) runs a dummy forward pass to build the graph. Required for `model.input` to exist.
- **Grad-CAM**: Walks **all layers** sequentially inside `GradientTape` (`src/model_utils.py:67`), not just the last conv. Last `Conv2D` auto-detected by name.

## Structure

- `app.py` — Streamlit entrypoint. Pages auto-discovered from `pages/`. App uses `env313/` virtual env (not `.venv/`).
- `src/` — 5 modules: `config.py`, `preprocessing.py`, `model_utils.py`, `metrics.py`, `visualization.py`.
- `model/mesonet_deepfake.h5` — pre-trained MesoNet-4. **Not** in `.gitignore`.
- No tests, no linter, no typechecker. No CI.
