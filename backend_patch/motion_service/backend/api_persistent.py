
import api
from persistent_inference import generate_motion

# The original API still handles authentication,
# session IDs, joint conversion, and continuation.
# Only its inference function is replaced.

assert "generate_motion" in (
    api.run_generation.__code__.co_names
), "Original API inference hook not found."

api.generate_motion = generate_motion

app = api.app


@app.get("/engine")
def engine():
    return {
        "engine": "persistent",
        "model": "DART",
        "session_support": True
    }
