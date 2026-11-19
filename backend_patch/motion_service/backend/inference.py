import os

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


def build_timeline_prompt(prompt, repetitions):
    """
    Convert comma-separated actions into DART's timeline format.

    Example:
    walk forward, turn left, wave both arms

    Becomes:
    walk forward*7,turn left*7,wave both arms*6
    when repetitions=20.
    """

    actions = [
        re.sub(r"(?i)^then\\s+", "", part.strip()).strip()
        for part in prompt.split(",")
    ]

    if any(not action for action in actions):
        raise ValueError("Each action must have a description.")

    if len(actions) > 4:
        raise ValueError("Use a maximum of four actions.")

    if repetitions < len(actions):
        raise ValueError(
            "Choose a longer motion for this number of actions."
        )

    # Distribute the requested primitives across actions.
    base = repetitions // len(actions)
    remainder = repetitions % len(actions)

    counts = [
        base + (1 if i < remainder else 0)
        for i in range(len(actions))
    ]

    return ",".join(
        f"{action}*{count}"
        for action, count in zip(actions, counts)
    )


def generate_motion(prompt, repetitions=10, state_in=None, state_out=None):
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

    full_prompt = build_timeline_prompt(prompt, repetitions)

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


    # Use a separate environment for every inference request.
    # This prevents one session's state from leaking into another.
    process_env = os.environ.copy()
    process_env.pop("DART_MOTION_STATE_IN", None)
    process_env.pop("DART_MOTION_STATE_OUT", None)

    if state_in is not None:
        state_in = Path(state_in).resolve()

        if not state_in.is_file():
            raise FileNotFoundError(
                f"Previous motion state not found: {state_in}"
            )

        process_env["DART_MOTION_STATE_IN"] = str(state_in)

    if state_out is not None:
        state_out = Path(state_out).resolve()
        state_out.parent.mkdir(parents=True, exist_ok=True)
        process_env["DART_MOTION_STATE_OUT"] = str(state_out)

    result = subprocess.run(
        command,
        cwd=str(DART_DIR),
        env=process_env,
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


    if state_out is not None and not state_out.is_file():
        raise FileNotFoundError(
            f"DART did not save its final motion state: {state_out}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    request_id = uuid.uuid4().hex[:8]
    destination = OUTPUT_DIR / f"motion_{request_id}.pkl"

    shutil.copy2(generated_file, destination)

    print("Motion generated successfully:", destination)

    return str(destination)
