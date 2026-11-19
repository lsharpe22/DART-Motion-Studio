
import json
import urllib.request
import urllib.error
from pathlib import Path

WORKER_URL = "http://127.0.0.1:8011"

OUTPUT_ROOT = Path(
    "/content/DART/motion_service/outputs/persistent"
).resolve()


def generate_motion(
    prompt,
    repetitions=10,
    state_in=None,
    state_out=None
):
    """Use the already-loaded DART worker."""

    payload = {
        "prompt": prompt,
        "repetitions": repetitions
    }

    if state_in is not None:
        payload["state_in"] = str(Path(state_in).resolve())

    if state_out is not None:
        payload["state_out"] = str(Path(state_out).resolve())

    request = urllib.request.Request(
        WORKER_URL + "/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(
            request, timeout=300
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as error:
        body = error.read().decode(
            "utf-8", errors="replace"
        )

        try:
            detail = json.loads(body).get("error", body)
        except ValueError:
            detail = body

        if error.code == 400:
            raise ValueError(detail) from error

        raise RuntimeError(
            "Persistent worker failed: " + detail
        ) from error

    except urllib.error.URLError as error:
        raise RuntimeError(
            "Persistent worker unavailable."
        ) from error

    output = Path(result["path"]).resolve()

    # Only accept generated files within the worker folder.
    try:
        output.relative_to(OUTPUT_ROOT)
    except ValueError:
        raise RuntimeError(
            "Worker returned an invalid output path."
        )

    if not output.is_file():
        raise FileNotFoundError(
            "Worker motion output missing."
        )

    if state_out is not None:
        expected = Path(state_out).resolve()

        if not expected.is_file():
            raise FileNotFoundError(
                "Worker continuation state missing."
            )

        if Path(result["state_path"]).resolve() != expected:
            raise RuntimeError(
                "Worker returned the wrong continuation state."
            )

    return str(output)
