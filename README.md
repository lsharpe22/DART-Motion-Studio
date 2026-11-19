# DART Motion Studio

Interactive text-to-3D-human-motion demo powered by **pretrained DART (DartControl)** inference. The browser-based Studio sends prompts to a GPU-hosted Python service, receives 22-joint motion sequences, and animates a skeleton or rigged Mixamo human. Follow-up prompts can continue an existing motion session.

- **Web demo:** https://lsharpe22.github.io/DART-Motion-Studio/
- **Original research/model:** [zkf1997/DART](https://github.com/zkf1997/DART) — *A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control*.
- **This project:** browser frontend, motion-service API, persistent GPU inference, continuation/session integration, cloud deployment, rendering and recording features, and associated evaluation.

> This repository's patch files are **not the complete pretrained DART implementation**. The original DART repository, checkpoints, processed data, and licensed SMPL-X body-model assets are needed. Weights and body-model files are not included in this source package.

## Repository files

```text
docs/index.html                 # Existing website in the GitHub repo (not included in this source-only ZIP)
backend_patch/                 # Eight project source/patch files; overlay onto upstream DART
scripts/check_assets.sh       # Checks environment, files and GPU readiness
scripts/start_backend.sh      # Starts worker and FastAPI, checks health
scripts/stop_backend.sh       # Stops only processes started by the start script
scripts/smoke_test.py         # Authenticated generate + optional continuation test
requirements-service.txt      # Additional service dependencies
REPRODUCIBILITY.md            # Detailed setup and limitations
EVALUATION_TEMPLATE.md        # Evaluation protocol and results template
.gitignore                    # Avoid accidentally committing private/large assets
```

## Quick start

1. Use a Linux machine or cloud Pod with an NVIDIA CUDA GPU. The successful demonstration used **NVIDIA A40**, Linux, **Python 3.8.20**, and **PyTorch 2.1.0+cu118**.
2. Install Miniconda/Miniforge, clone the [upstream DART repository](https://github.com/zkf1997/DART), and follow its documented environment and data/model setup. Its `environment.yml` is the starting point for the GPU dependencies; simply installing `requirements-service.txt` is **not sufficient**. See [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
3. Overlay the files in `backend_patch/` into the root of the upstream DART checkout, keeping their relative paths.
4. Put the official model checkpoints, config `args.yaml`, preprocessed data, stand seed, and appropriately licensed SMPL-X assets at the paths described below. Do not upload restricted body-model assets into public GitHub.
5. Ensure your environment can run DART and CUDA. Install the API-only packages:

   ```bash
   python -m pip install -r requirements-service.txt
   ```

6. The custom code currently uses fixed historical Colab paths. On the cloud host, arrange:

   ```bash
   sudo mkdir -p /content
   sudo ln -sfn /absolute/path/to/upstream/DART /content/DART
   sudo ln -sfn /absolute/path/to/working/python-environment /content/dart-env
   ```

   If `/content/DART` or `/content/dart-env` already exists as a **real directory**, inspect and move it safely before creating a symlink; `ln -sfn` alone will not replace a real directory.

7. On the machine hosting the service, set a new strong private API key; **never commit or post it**:

   ```bash
   export DART_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
   ```

8. Run these commands from the GitHub project root after ensuring the upstream DART is checked out and patched:

   ```bash
   bash scripts/check_assets.sh
   DART_API_KEY="$DART_API_KEY" bash scripts/start_backend.sh
   python scripts/smoke_test.py --url http://127.0.0.1:8012 --api-key "$DART_API_KEY"
   ```

9. To use the GitHub Pages frontend remotely, establish an **HTTPS reverse tunnel or trusted HTTPS deployment** to port `8012`, then enter its HTTPS URL and your private API key in the website's Settings. If using Cloudflare Quick Tunnel, its random URL changes on restart. **Do not expose the unauthenticated worker at port 8011**.

## Required model/data locations (relative to upstream DART)

```text
mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt
mld_denoiser/mld_fps_clip_repeat_euler/args.yaml
mvae/mvae_fps_clip/checkpoint_200000.pt
mvae/mvae_fps_clip/args.yaml
data/stand.pkl
data/seq_data_zero_male/mean_std_h2_f8.pkl
body_models/smplx/SMPLX_MALE.npz            # additional gender files as needed
```

Checkpoint YAML files may reference more preprocessed data/config paths; follow the upstream instructions and validate them. The cached OpenAI CLIP ViT-B/32 model may be downloaded by the upstream `clip` dependency on first run (or provided in the library's standard cache). **Do not confuse the unrelated PyPI package `clip` with the OpenAI CLIP implementation.**

## API

- `GET /health`: service liveness; this response **does not prove the worker is healthy**.
- `GET /engine`: shows persistent inference adapter is selected.
- `POST /generate`: `X-API-Key` header; JSON `{"prompt":"walk forward","repetitions":5}`.
- `POST /continue`: same authentication; JSON `{"prompt":"turn left","repetitions":5,"session_id":"<returned ID>"}`.
- Worker health: `GET http://127.0.0.1:8011/health` from **inside the server**; look for `model_loaded: true`.

Generated responses contain a `session_id`, `frames`, `joint_count`, `fps` (30), and a list of 3-D joint positions. Sessions are held in process memory; restarting the API invalidates previous sessions. Single-user/demo operation only: only one GPU generation is accepted at a time.

## Limitations and security

The demonstrated system uses prepaid GPU compute, Python environment and model assets that need substantial storage, and session state files. Known motion artifacts include visually stiff Mixamo retargeting and occasional lateral wobble on walking. Mobile usability is not confirmed until actually tested on a phone. A Cloudflare Quick Tunnel is a **temporary demonstration endpoint**, not a permanent deployment strategy. The browser's authorized origin is hard-coded to `https://lsharpe22.github.io` in `backend_patch/motion_service/backend/api.py`; adapt CORS only for a domain you control. Do not publish API keys, SMPL-X downloads, session-state files, or model weights that lack distribution rights.

## Attribution

DART research, pretrained neural architectures and model assets belong to their respective authors and licensors. This student project adapts inference to an interactive service and 3D browser viewer; it does **not** claim to have trained the pretrained DART model.
