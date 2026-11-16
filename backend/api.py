
import pickle
import os
import secrets
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

from backend.inference import generate_motion

app = FastAPI(title="DART Motion API", version="0.1.0")

generation_lock = threading.Lock()

class MotionRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=120)
    repetitions: int = Field(default=5, ge=1, le=20)

@app.get("/health")
def health():
    return {"status": "ok", "model": "DART"}

@app.post("/generate")
def generate(
    request: MotionRequest,
    x_api_key: str = Header(default="")
):
    expected_key = os.environ.get("DART_API_KEY", "")
    if not expected_key or not secrets.compare_digest(
        x_api_key, expected_key
    ):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )

    # Prevent simultaneous requests from overwriting DART output.
    if not generation_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Model is busy generating another motion."
        )

    try:
        path = generate_motion(
            request.prompt,
            request.repetitions
        )

        # Load only files created by our own backend.
        with open(path, "rb") as file:
            motion = pickle.load(file)

        joints = motion["joints"]
        if hasattr(joints, "detach"):
            joints = joints.detach().cpu().numpy()

        return {
            "status": "success",
            "prompt": request.prompt,
            "frames": len(joints),
            "joint_count": joints.shape[1],
            "fps": 30,
            "joints": joints.tolist()
        }

    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        print("Inference error:", repr(error))
        raise HTTPException(
            status_code=500,
            detail="Motion generation failed. Check server logs."
        )
    finally:
        generation_lock.release()
