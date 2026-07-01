# AGENTS.md

## Commands

| Action | Command |
|---|---|
| Run app | `source env313/bin/activate && streamlit run app.py` |
| Test model (no UI) | `python debug_model.py` |

## Key gotchas

- **Label swap**: Model was trained with `REAL=1, FAKE=0`. Config flag `INVERT_PROBABILITY = True` corrects this at inference. Metrics treat `Real=0, Fake=1` (binary convention after inversion).
- **CPU only + 8 GB RAM**: Batch size capped at 100 (`MAX_BATCH`). All inference is on CPU.
- **Preprocessing**: No face detection, no mean subtraction, no TTA. Only `RGB convert → resize 256×256 → /255.0`. Placeholder code exists but is gated behind `False` flags in `src/config.py`.
- **Model load dry-run**: `load_model()` runs a dummy forward pass to build the graph (`model(dummy, training=False)`). Required for `model.input` to exist.
- **Grad-CAM**: Walks **all layers** sequentially inside `GradientTape`, not just the last conv. Last `Conv2D` layer is auto-detected by name.

## Structure

- `app.py` — Streamlit entrypoint. Pages auto-discovered from `pages/`.
- `src/` — 5 modules: `config.py`, `preprocessing.py`, `model_utils.py`, `metrics.py`, `visualization.py`.
- `model/mesonet_deepfake.h5` — pre-trained MesoNet-4. **Not** in `.gitignore`.
- No tests, no linter, no typechecker. No CI.
