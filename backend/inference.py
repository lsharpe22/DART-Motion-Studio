
import subprocess
import shutil
import uuid
import re
from pathlib import Path

DART_DIR = Path("/content/DART")
PYTHON = "/content/dart-env/bin/python"

CHECKPOINT = (
    DART_DIR /
    "mld_denoiser/mld_fps_clip_repeat_euler/checkpoint_300000.pt"
)

OUTPUT_DIR = DART_DIR / "motion_service/outputs"

def generate_motion(prompt, repetitions=10):
    """
    Generate motion using pretrained DART.

    Returns the path to the generated pickle file.
    """

    prompt = prompt.strip()

    if not prompt:
        raise ValueError("Please enter a motion prompt.")

    if not 1 <= repetitions <= 20:
        raise ValueError("Repetitions must be between 1 and 20.")

    # Restrict input to a simple action description.
    # DART uses * to specify the number of motion primitives.
    if not re.fullmatch(r"[A-Za-z0-9 ,.'-]+", prompt):
        raise ValueError("Use a simple text prompt without special characters.")

    full_prompt = f"{prompt}*{repetitions}"

    command = [
        PYTHON,
        "-m",
        "mld.rollout_mld",
        "--denoiser-checkpoint",
        str(CHECKPOINT),
        "--text-prompt",
        full_prompt,
        "--dataset",
        "babel",
        "--device",
        "cuda",
        "--batch-size",
        "1",
        "--guidance-param",
        "1.0",
        "--export-smpl",
        "0",
        "--use-predicted-joints",
        "1"
    ]

    print("Generating:", full_prompt)

    result = subprocess.run(
        command,
        cwd=str(DART_DIR),
        capture_output=True,
        text=True,
        timeout=300
    )

    if result.returncode != 0:
        raise RuntimeError(
            "DART inference failed:\n" + result.stderr[-3000:]
        )

    # Match DART's output directory naming convention.
    rollout_root = (
        CHECKPOINT.parent /
        CHECKPOINT.stem /
        "rollout"
    )

    folder_name = (
        "use_pred_joints_" +
        full_prompt[:40].replace(" ", "_").replace(".", "") +
        "_guidance1.0_seed0"
    )

    generated_file = rollout_root / folder_name / "sample_0.pkl"

    if not generated_file.exists():
        raise FileNotFoundError(
            f"Generated file not found: {generated_file}\n"
            f"Process output:\n{result.stdout[-2000:]}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    request_id = uuid.uuid4().hex[:8]
    destination = OUTPUT_DIR / f"motion_{request_id}.pkl"

    shutil.copy2(generated_file, destination)

    print("Motion generated successfully:", destination)

    return str(destination)
