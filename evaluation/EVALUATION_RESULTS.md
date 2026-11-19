# DART Motion Studio — Evaluation Results

**Date:** November 19, 2026  
**Project:** DART Motion Studio — interactive text-driven human motion generation  
**Reference paper:** Zhao et al., *DartControl: A Diffusion-Based Autoregressive Motion Model for Real-Time Text-Driven Motion Control* (ICLR 2025), https://arxiv.org/abs/2410.05260  
**Original project:** https://github.com/zkf1997/DART  
**Official project page:** https://zkf1997.github.io/DART/

## 1. Objective

Evaluate whether the deployed system reliably generates and continues 3D human motion from text prompts, measure response latency on the deployed NVIDIA A40, and compare the results carefully with the original paper's reported speed. This is a **deployment-level functional/latency evaluation**, not a full reproduction of the paper's standardized motion-quality benchmarks.

## 2. Environment and protocol

- **Hardware:** single NVIDIA A40 (Runpod; 46,068 MiB reported VRAM).
- **Software:** Python 3.8.20, PyTorch 2.1.0+cu118; persistent DART model worker on localhost:8011, FastAPI application on localhost:8012.
- **Model assets:** original DART denoiser `checkpoint_300000.pt`, motion VAE `checkpoint_200000.pt`, required SMPL-X/data assets.
- **Requests:** `POST /generate` with prompt `walk forward`, `repetitions=5`; `POST /continue` with prompt `turn left`, `repetitions=5`, using the returned `session_id`.
- **Trials:** five sequential paired requests against an already-running, warmed-up worker, after an earlier successful smoke test. No worker reload between trials.
- **Timing method:** Python `time.perf_counter()` around each HTTP request through the project's `scripts/smoke_test.py`, measuring **local API round-trip time**. This is not isolated GPU kernel/denoiser time, and it does not include a public internet or Cloudflare round trip.
- **Output:** 22 joints per frame; 30 FPS playback metadata; 42 frames per initial generation and 40 new frames per continuation.
- **Functional assertions:** success response, session ID, positive frame count, 22 joints, and returned joint-array length matching frame count. These assertions do not independently assess motion quality.

### Exact reproduction command (on an already-running backend)

Do not commit real API keys. The sample reads the locally stored private key, and all test calls stay on localhost:

```bash
cd /workspace
for i in 1 2 3 4 5; do
  echo "=== Trial $i ==="
  PYTHONHOME=/content/dart-env \
  /content/dart-env/bin/python /tmp/dart_smoke_test.py \
    --url http://127.0.0.1:8012 \
    --api-key "$(cat /workspace/.dart_api_key)" \
    --prompt "walk forward" \
    --repetitions 5 \
    --continue-prompt "turn left"
done
```

For a fresh setup, first follow `REPRODUCIBILITY.md` and use the repository version of `scripts/smoke_test.py` in place of `/tmp/dart_smoke_test.py`. The authentication key must be configured privately; do not commit it.

## 3. Trial measurements

| Trial | Walk forward (42 frames), s | Turn left continuation (40 frames), s | Test |
|---:|---:|---:|---|
| 1 | 0.499 | 0.522 | PASS |
| 2 | 0.513 | 0.559 | PASS |
| 3 | 0.546 | 0.579 | PASS |
| 4 | 0.552 | 0.581 | PASS |
| 5 | 0.498 | 0.495 | PASS |
| **Mean** | **0.5216** | **0.5472** | **5/5 paired tests passed** |
| **Sample standard deviation** | **0.0258** | **0.0376** | |
| **Minimum–maximum** | **0.498–0.552** | **0.495–0.581** | |
| **Frames / mean wall-clock seconds** | **80.5 frames/s** | **73.1 frames/s** | |

The rate is calculated as `returned frames / mean API response time` and is **not** the animation playback FPS. At 30 FPS playback, 42 frames represent about 1.40 seconds of animation and 40 frames about 1.33 seconds. Both are produced in less than a second on average in this warmed-up local test.

A separate preceding smoke test returned 0.760 seconds for `/generate` and 0.568 seconds for `/continue`; it was not included in the five-trial mean. The worker reported startup time of 8.27 seconds after the Runpod migration; startup is not included in measured API latencies.

## 4. Comparison with original research

The official research project page reports **over 300 generated frames per second on a single RTX 4090 GPU** for DART motion generation (https://zkf1997.github.io/DART/). This is a **published author-reported result**, not a value independently replicated in this project.

| Dimension | Original DART reported result | This project | Comparable? |
|---|---|---|---|
| Hardware | NVIDIA RTX 4090 | NVIDIA A40 | **No:** different GPUs |
| Speed | More than 300 frames/s (author-reported motion-generation throughput) | 80.5 frames/s initial; 73.1 frames/s continuation **including local API round trip** | **No:** speed definitions and setup differ |
| Model | DART diffusion autoregressive motion generator | Released pretrained denoiser and motion VAE checkpoints | Same model family; no proof of exact evaluation configuration |
| Application | Model inference and interactive text-driven motion control | Persistent GPU worker, FastAPI, browser/Three.js UI, prompt continuation | Application integration is additional |
| Motion quality | Paper's quantitative/qualitative evaluations | Visual inspection only | **No:** published datasets and standardized metrics not reproduced |

**Interpretation:** These results confirm working low-latency cloud inference for the prompts tested. They do **not** establish a hardware-normalized speed deficit versus the paper, nor do they reproduce the paper's motion-quality statistics. Avoid stating that the model reproduces the paper's >300 FPS result or its standardized quality metrics.

## 5. Observations and limitations

1. **Functional success:** initial generation and text-prompt continuation passed all five paired trials with valid 22-joint sequences and session IDs.
2. **Rendering:** the frontend plays generated 3D joint motion and can retarget it onto a Mixamo character.
3. **Visual artifacts:** `walk forward` can show sideways/lateral shimmy even in the skeleton; the rigged human looks somewhat stiff. Their origins (model output versus coordinate conversion/retargeting) have **not been isolated**, so do not categorically attribute them to DART.
4. **No standardized quality replication:** no controlled FID, R-precision, motion-composition metrics, human ratings, or paper-dataset protocol were run here.
5. **Warm vs cold:** measurements use a preloaded worker. They exclude startup and should not be described as cold-start latencies.
6. **Deployment dependence:** the demo uses a rented GPU and temporary Cloudflare tunnel; the public endpoint changes when restarted.
7. **Reproducibility scope:** `check_assets.sh` and the authenticated smoke test passed on the existing deployed environment. A full clean-environment setup by an independent user is still unverified.

## 6. Outcome and possible follow-up

**Outcome:** Functional inference deployment verified; 10/10 generation/continuation requests successful; mean API response times 0.5216 s and 0.5472 s on the NVIDIA A40. Evidence supports a working model-as-a-service demonstration, not a full research benchmark reproduction.

If additional time is available: independently test a clean installation; run varied prompts/seeds; inspect root-motion direction numerically; and implement the original paper's published evaluation protocol if its dataset, scripts, and metrics are accessible.
