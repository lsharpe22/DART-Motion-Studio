# Reproduction guide — DART Motion Studio

## 1. What was validated

The student demo ran on a Runpod NVIDIA A40 with Python 3.8.20, torch 2.1.0+cu118, CUDA available, persistent worker port 8011, FastAPI port 8012, and a browser over a temporary HTTPS tunnel. Model startup observed after migration: 8.27 s (single run). One previous API test produced ~42 frames for a first prompt and ~40 new frames for continuation; timings are not a paper-comparable benchmark.

**This bundle has been statically inspected and its helper files syntax-checked, but a full clean-room install has not been executed as part of packaging.** Verify the full procedure on a new Pod before claiming end-to-end reproducibility.

## 2. Upstream environment (GPU required)

- Linux x86-64 with NVIDIA GPU and compatible driver; sufficient GPU memory and disk space.
- Install conda. Clone `https://github.com/zkf1997/DART` and follow the **exact upstream** `README.md`, `environment.yml`, checkpoint/model download and data preparation instructions.
- Python 3.8.20 and PyTorch 2.1.0+cu118 were validated in the demonstration environment; the upstream conda environment is detailed and may require resolving aging package pins.
- From the GitHub project root, overlay all eight `backend_patch/` files onto the upstream repository, preserving relative paths:

```bash
# Example: PROJECT is this GitHub project, DART_HOME the upstream checkout
cp -a "$PROJECT/backend_patch/." "$DART_HOME/"
```

- Install additional HTTP dependencies into the working **DART** Python environment:

```bash
python -m pip install -r "$PROJECT/requirements-service.txt"
```

- Verify `python -c 'import torch; print(torch.__version__, torch.cuda.is_available())'` and `python -c 'import mld.rollout_mld'` from the DART checkout.

## 3. Checkpoint/data layout

The custom worker expects:

```text
$DART_HOME/mld_denoiser/mld_fps_clip_repeat_euler/{checkpoint_300000.pt,args.yaml}
$DART_HOME/mvae/mvae_fps_clip/{checkpoint_200000.pt,args.yaml}
$DART_HOME/data/stand.pkl
$DART_HOME/data/seq_data_zero_male/mean_std_h2_f8.pkl
$DART_HOME/body_models/smplx/SMPLX_MALE.npz
```

Other data and SMPL-X assets can be needed by the saved YAML config. Obtain them from authorized upstream sources. `config_files/data_paths.py` is set to `body_model_dir = './body_models'` because downstream SMPL-X utilities add the `smplx` subdirectory. Keep CLIP ViT-B/32 downloadable/available from the dependency's cache. If the cache is relocated, restore its path before startup.

## 4. Historical absolute paths

Current source uses `/content/DART` and `/content/dart-env`. Those paths arose in Colab and were kept for the demonstrated deployment. Create paths safely (do **not** overwrite a real directory):

```bash
mkdir -p /content  # root permissions may be required
# Ensure targets do NOT already exist as real directories
ln -s /absolute/path/to/upstream/DART /content/DART
ln -s /absolute/path/to/python-environment /content/dart-env
```

If using a **migrated conda environment** rather than installing afresh, you may also need to configure `PYTHONHOME=/content/dart-env` and standard-library paths. A fresh upstream install is preferable; archived conda environments are not guaranteed to be relocatable.

## 5. Start the worker and API

Set a private key in your shell, run `bash scripts/check_assets.sh`, then run `bash scripts/start_backend.sh`. These scripts expect `$DART_HOME` (defaults to `/content/DART`) and `$DART_PYTHON` (defaults to `/content/dart-env/bin/python`) and create logs in the checkout's `motion_service/outputs/service_logs/`. The start script waits for the worker to load, then checks the API and reports failures from logs. An existing unrelated server on ports 8011/8012 causes the script to stop rather than overwrite it.

```bash
export DART_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
export DART_HOME=/content/DART
export DART_PYTHON=/content/dart-env/bin/python
bash scripts/start_backend.sh
```

Service only binds to localhost. For a **temporary demo** you can use the downloaded `cloudflared` executable or equivalent authenticated HTTPS reverse proxy:

```bash
cloudflared tunnel --no-autoupdate --url http://127.0.0.1:8012
```

Read its generated `https://...trycloudflare.com` URL from terminal. Set that URL and your key in browser Settings. Quick Tunnels are ephemeral; use a stable authenticated HTTPS service for long-term accessibility.

## 6. Test authenticated generation and continuation

```bash
python scripts/smoke_test.py --url http://127.0.0.1:8012 --api-key "$DART_API_KEY"
python scripts/smoke_test.py --url http://127.0.0.1:8012 --api-key "$DART_API_KEY" --continue-prompt 'turn left'
```

The test prints metadata (never keys or full joint arrays). A successful response has 22 joints, positive frames, and a non-empty session ID. Repeated runs may differ; report hardware, model checkpoint, number of primitives, timing, and whether the model had already loaded.

## 7. Troubleshooting

- **Missing `encodings`/`urllib.parse`/torch internals:** environment is incomplete or its prefix is wrong; repair or recreate it. `python --version` alone is not a valid environment test.
- **Worker not ready:** check `motion_service/outputs/service_logs/worker.log`, GPU, assets, CLIP cache, and paths. First load may require more than 15 seconds.
- **API /health works but generate fails:** check worker `8011/health`, key, session state, and logs. API `/health` is liveness only.
- **401 Unauthorized:** use the same `DART_API_KEY` in the running API process and frontend `X-API-Key` header. Generate a new secret rather than sharing it.
- **429 Busy:** only one inference request at a time.
- **Browser CORS failure:** ensure the website origin matches the allowlist and HTTPS tunnel is reachable.
- **Stiff motion / lateral shimmy:** known qualitative limitations; no fix assumed in this reproduction.

## 8. What a grader should do

Check CUDA and worker health, run smoke test, open website on browser, generate `walk forward`, continue `turn left`, inspect returned 22-joint frames and recorded demo evidence. Compare timings with the research paper only when checkpoint, evaluation method, GPU, and sampling settings match; do not treat app latency and paper model-only FPS as directly equivalent.
