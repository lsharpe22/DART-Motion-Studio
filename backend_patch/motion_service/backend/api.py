
import os
import pickle
import secrets
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from motion_service.backend.inference import generate_motion


app = FastAPI(title="DART Motion API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://lsharpe22.github.io"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)

generation_lock = threading.Lock()

# Sessions stay in memory while the backend runs.
sessions = {}

STATE_DIR = Path(
    "/content/DART/motion_service/outputs/session_states"
)
STATE_DIR.mkdir(parents=True, exist_ok=True)

MAX_SESSIONS = 20


class MotionRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=120)
    repetitions: int = Field(default=5, ge=1, le=20)


class ContinueRequest(MotionRequest):
    session_id: str = Field(min_length=16, max_length=128)


def check_api_key(x_api_key):
    expected = os.environ.get("DART_API_KEY", "")

    if not expected or not secrets.compare_digest(
        x_api_key, expected
    ):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )


def run_generation(request, session_id=None):
    # Only one GPU generation at a time.
    if not generation_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=429,
            detail="Model is busy generating another motion."
        )

    new_state = None

    try:
        continuing = session_id is not None

        if continuing:
            previous_state = sessions.get(session_id)

            if previous_state is None or not previous_state.is_file():
                raise HTTPException(
                    status_code=404,
                    detail="Motion session not found. Start a new motion."
                )
        else:
            session_id = secrets.token_urlsafe(24)
            previous_state = None

        # Never use client-supplied paths.
        new_state = STATE_DIR / (
            session_id + "_" + secrets.token_hex(8) + ".pt"
        )

        path = generate_motion(
            request.prompt,
            request.repetitions,
            state_in=previous_state,
            state_out=new_state
        )

        with open(path, "rb") as file:
            motion = pickle.load(file)

        joints = motion["joints"]

        if hasattr(joints, "detach"):
            joints = joints.detach().cpu().numpy()

        # Each continuation includes the previous 2 history
        # frames. Remove them before appending on the website.
        if continuing:
            joints = joints[2:]

        # Limit how many active sessions are retained.
        if not continuing and len(sessions) >= MAX_SESSIONS:
            oldest_id = next(iter(sessions))
            oldest_state = sessions.pop(oldest_id)
            oldest_state.unlink(missing_ok=True)

        sessions[session_id] = new_state

        # The previous state is no longer needed.
        if continuing:
            previous_state.unlink(missing_ok=True)

        return {
            "status": "success",
            "session_id": session_id,
            "continued": continuing,
            "prompt": request.prompt,
            "frames": len(joints),
            "joint_count": joints.shape[1],
            "fps": 30,
            "joints": joints.tolist()
        }

    except HTTPException:
        raise

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:
        print("Inference error:", repr(error))

        if new_state is not None:
            new_state.unlink(missing_ok=True)

        raise HTTPException(
            status_code=500,
            detail="Motion generation failed. Check server logs."
        )

    finally:
        generation_lock.release()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "DART",
        "session_support": True
    }


@app.post("/generate")
def generate(
    request: MotionRequest,
    x_api_key: str = Header(default="")
):
    check_api_key(x_api_key)
    return run_generation(request)


@app.post("/continue")
def continue_motion(
    request: ContinueRequest,
    x_api_key: str = Header(default="")
):
    check_api_key(x_api_key)
    return run_generation(request, session_id=request.session_id)
