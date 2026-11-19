
import os
import sys
import json
import time
import random
import secrets
import re
import traceback
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler

import numpy as np
import torch

DART = Path("/content/DART")
BACKEND = DART / "motion_service/backend"

sys.path.insert(0, str(DART))
sys.path.insert(0, str(BACKEND))

import mld.rollout_mld as dart
from inference import build_timeline_prompt

CHECKPOINT = (
    DART /
    "mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt"
)

OUTPUT_DIR = (
    DART / "motion_service/outputs/persistent"
)

STATE_DIR = (
    DART / "motion_service/outputs/session_states"
)

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
STATE_DIR.mkdir(parents=True, exist_ok=True)

started_at = time.monotonic()

print("Initializing persistent DART worker...", flush=True)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is unavailable. Select a T4 GPU.")

device = torch.device("cuda")

# Match the existing rollout command configuration.
args = dart.RolloutArgs()

args.denoiser_checkpoint = str(CHECKPOINT)
args.text_prompt = "walk forward*3"
args.dataset = "babel"
args.device = device
args.batch_size = 1
args.guidance_param = 1.0
args.export_smpl = False
args.use_predicted_joints = True
args.save_dir = OUTPUT_DIR

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

torch.set_default_dtype(torch.float32)

torch.backends.cudnn.deterministic = (
    args.torch_deterministic
)

# Load neural networks ONCE.
denoiser_args, denoiser_model, vae_args, vae_model = (
    dart.load_mld(args.denoiser_checkpoint, device)
)

# Initialize diffusion ONCE.
diffusion_args = denoiser_args.diffusion_args
diffusion_args.respacing = args.respacing

diffusion = dart.create_gaussian_diffusion(
    diffusion_args
)

# Initialize the dataset, body model and CLIP encoder ONCE.
dataset = dart.SinglePrimitiveDataset(
    cfg_path=vae_args.data_args.cfg_path,
    dataset_path=vae_args.data_args.data_dir,
    body_type=vae_args.data_args.body_type,
    sequence_path="./data/stand.pkl",
    batch_size=1,
    device=device,
    enforce_gender="male",
    enforce_zero_beta=1,
)

initialization_seconds = time.monotonic() - started_at

print(
    "PERSISTENT_DART_READY",
    "startup_seconds=" + str(round(initialization_seconds, 2)),
    "gpu=" + torch.cuda.get_device_name(0),
    flush=True
)


def allowed_state_path(value):
    if not value:
        return None

    resolved = Path(value).resolve()

    # Only accept states from our dedicated session directory.
    if resolved.parent != STATE_DIR.resolve():
        raise ValueError("Invalid motion state path.")

    return resolved


def generate(request):
    prompt = str(request.get("prompt", "")).strip()
    repetitions = int(request.get("repetitions", 5))

    if not re.fullmatch(r"[A-Za-z0-9 ,.'-]+", prompt):
        raise ValueError("Invalid motion prompt.")

    if not 1 <= repetitions <= 20:
        raise ValueError("Repetitions must be 1–20.")

    full_prompt = build_timeline_prompt(
        prompt, repetitions
    )

    state_in = allowed_state_path(
        request.get("state_in")
    )

    if state_in and not state_in.is_file():
        raise FileNotFoundError("Input motion state missing.")

    state_out = allowed_state_path(
        request.get("state_out")
    )

    if state_out is None:
        state_out = (
            STATE_DIR /
            ("worker_" + secrets.token_hex(12) + ".pt")
        )

    # Unique directory avoids collisions between requests.
    request_dir = (
        OUTPUT_DIR / ("job_" + secrets.token_hex(12))
    )

    request_dir.mkdir(parents=True)

    args.save_dir = request_dir
    args.text_prompt = full_prompt

    # The existing DART rollout reads these environment
    # variables to restore/save motion continuation state.
    os.environ.pop("DART_MOTION_STATE_IN", None)
    os.environ["DART_MOTION_STATE_OUT"] = str(state_out)

    if state_in:
        os.environ["DART_MOTION_STATE_IN"] = str(state_in)

    start = time.monotonic()

    try:
        with torch.no_grad():
            dart.rollout(
                full_prompt,
                denoiser_args,
                denoiser_model,
                vae_args,
                vae_model,
                diffusion,
                dataset,
                args
            )

        results = list(request_dir.rglob("sample_0.pkl"))

        if len(results) != 1:
            raise RuntimeError(
                "Expected one DART output, got " +
                str(len(results))
            )

        if not state_out.is_file():
            raise RuntimeError(
                "DART failed to save continuation state."
            )

        seconds = time.monotonic() - start

        print(
            "GENERATED",
            prompt,
            "repetitions=", repetitions,
            "seconds=", round(seconds, 2),
            flush=True
        )

        return {
            "status": "success",
            "path": str(results[0]),
            "state_path": str(state_out),
            "seconds": round(seconds, 3),
            "prompt": prompt
        }

    finally:
        os.environ.pop("DART_MOTION_STATE_IN", None)
        os.environ.pop("DART_MOTION_STATE_OUT", None)


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status, value):
        body = json.dumps(value).encode("utf-8")

        self.send_response(status)
        self.send_header(
            "Content-Type",
            "application/json"
        )
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path != "/health":
            return self.send_json(404, {"error": "Not found"})

        self.send_json(200, {
            "status": "ok",
            "model_loaded": True,
            "gpu": torch.cuda.get_device_name(0),
            "startup_seconds": round(
                initialization_seconds, 2
            ),
            "uptime_seconds": round(
                time.monotonic() - started_at, 2
            )
        })

    def do_POST(self):
        if self.path != "/generate":
            return self.send_json(404, {"error": "Not found"})

        try:
            size = int(self.headers.get("Content-Length", "0"))

            if size <= 0 or size > 4096:
                return self.send_json(
                    400, {"error": "Invalid request size"}
                )

            request = json.loads(
                self.rfile.read(size).decode("utf-8")
            )

            result = generate(request)
            self.send_json(200, result)

        except (ValueError, FileNotFoundError) as error:
            self.send_json(400, {"error": str(error)})

        except Exception as error:
            traceback.print_exc()
            self.send_json(500, {"error": str(error)})

    def log_message(self, format, *args):
        pass


# Only accessible from this Colab runtime.
server = HTTPServer(
    ("127.0.0.1", 8011),
    Handler
)

print("Worker listening on 127.0.0.1:8011", flush=True)
server.serve_forever()
