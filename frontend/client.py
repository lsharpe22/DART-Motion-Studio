
import sys
import pickle
import tempfile
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation, FFMpegWriter

import gradio as gr

APP_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DIR))

import requests
import torch

BONES = [
    (0, 1), (0, 2), (0, 3),
    (1, 4), (2, 5), (3, 6),
    (4, 7), (5, 8), (6, 9),
    (7, 10), (8, 11), (9, 12),
    (9, 13), (9, 14),
    (12, 15),
    (13, 16), (14, 17),
    (16, 18), (17, 19),
    (18, 20), (19, 21)
]

def render_motion(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    joints = data["joints"]
    if hasattr(joints, "detach"):
        joints = joints.detach().cpu().numpy()

    joints = np.asarray(joints)

    center = (
        joints.min(axis=(0, 1)) +
        joints.max(axis=(0, 1))
    ) / 2

    span = max(np.ptp(joints, axis=(0, 1)).max(), 1.0)
    half = span / 2 + 0.2

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    def update(frame):
        ax.cla()

        for a, b in BONES:
            ax.plot(
                joints[frame, [a, b], 0],
                joints[frame, [a, b], 1],
                joints[frame, [a, b], 2],
                linewidth=2
            )

        ax.scatter(
            joints[frame, :, 0],
            joints[frame, :, 1],
            joints[frame, :, 2],
            s=12
        )

        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)

        ax.set_title(f"DART Motion | Frame {frame + 1}")
        return []

    animation = FuncAnimation(
        fig,
        update,
        frames=len(joints),
        interval=33
    )

    video_path = Path(tempfile.mkdtemp()) / "motion.mp4"

    animation.save(
        str(video_path),
        writer=FFMpegWriter(fps=30),
        dpi=80
    )

    plt.close(fig)

    return str(video_path), len(joints)



import os

API_URL = os.environ.get(
    "DART_API_URL",
    "http://127.0.0.1:8000"
).rstrip("/")

def run_generation(prompt, repetitions):
    try:
        response = requests.post(
            f"{API_URL}/generate",
            headers={
                "X-API-Key": os.environ["DART_API_KEY"]
            },
            json={
                "prompt": prompt,
                "repetitions": int(repetitions)
            },
            timeout=300
        )

        response.raise_for_status()
        data = response.json()

        # Adapt the JSON response for our existing renderer.
        # This is a temporary compatibility format.
        import tempfile
        import pickle

        temp_dir = Path(tempfile.mkdtemp())
        motion_path = temp_dir / "generated_motion.pkl"

        motion = {
            "joints": torch.tensor(
                data["joints"],
                dtype=torch.float32
            ),
            "texts": [data["prompt"]]
        }

        with open(motion_path, "wb") as file:
            pickle.dump(motion, file)

        video_path, frames = render_motion(str(motion_path))

        return (
            video_path,
            str(motion_path),
            f"Generated {frames} frames using the DART API."
        )

    except requests.exceptions.RequestException as error:
        raise gr.Error(f"API request failed: {error}")
    except Exception as error:
        raise gr.Error(str(error))

CUSTOM_CSS = """
.gradio-container {
    background: #101827 !important;
    color: #F2F5FA !important;
    max-width: 1100px !important;
    margin: auto !important;
}

#app-title {
    font-size: 34px;
    font-weight: 750;
    color: #F2F5FA;
    margin-bottom: 5px;
}

#subtitle {
    color: #A9B8CF;
    font-size: 15px;
}

#generate-btn {
    background: #D97732 !important;
    color: white !important;
    font-size: 18px !important;
    font-weight: 700 !important;
    min-height: 54px !important;
    border: none !important;
}

#generate-btn:hover {
    background: #EB8A45 !important;
}

"""

theme = gr.themes.Base(
    primary_hue="orange",
    neutral_hue="slate",
)

with gr.Blocks(
    title="DART Motion Studio",
    theme=theme,
    css=CUSTOM_CSS
) as demo:

    gr.HTML(
        '<div id="app-title">DART Motion Studio</div>'
        '<div id="subtitle">'
        'Generate 3D human motion using pretrained diffusion AI.'
        '</div>'
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Generate a motion")

            prompt = gr.Textbox(
                label="Describe a movement",
                value="walk forward",
                placeholder="Example: walk in circles",
                lines=2
            )

            repetitions = gr.Radio(
                choices=[
                    ("Short (5 primitives)", 5),
                    ("Medium (10 primitives)", 10),
                    ("Long (20 primitives)", 20)
                ],
                value=10,
                label="Motion length"
            )

            generate_button = gr.Button(
                "✦ Generate Motion",
                variant="primary",
                elem_id="generate-btn"
            )

            status = gr.Textbox(
                label="Generation status",
                interactive=False
            )

        with gr.Column(scale=2):
            gr.Markdown("### Motion preview")

            video = gr.Video(
                label="Generated animation",
                height=430
            )

            motion_file = gr.File(
                label="Download motion data"
            )

    generate_button.click(
        fn=run_generation,
        inputs=[prompt, repetitions],
        outputs=[video, motion_file, status],
        show_progress="full"
    )

if __name__ == "__main__":
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=True,
        show_error=True
    )
