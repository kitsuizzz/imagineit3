"""Per-user Kaggle API client.

A user pastes their own Kaggle API token (``KGAT_...``); the app uses it as a
bearer to read the notebook, re-push it, and start a run.  Disconnect cannot
use the API (Kaggle has no stop endpoint), so it asks the worker inside the
notebook to exit instead.

Wire format note: the push endpoint wants **JSON**, not multipart.  Sending
multipart fails with ``Error reading JToken from JsonReader``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

BASE_URL = "https://www.kaggle.com/api/v1"
DEFAULT_KERNEL = "wssdaas/imagineitv3"
TIMEOUT_SECONDS = 120
GPU_SHAPE = "NvidiaTeslaT4"
TEMPLATE_FILE = Path(__file__).resolve().parents[2] / "templates" / "imagineitv3.ipynb"


def load_local_template() -> str | None:
    """Read the bundled notebook used when the target kernel is missing."""
    path = Path(os.getenv("KAGGLE_TEMPLATE_FILE", str(TEMPLATE_FILE)))
    try:
        return path.read_text(encoding="utf-8")
    except OSError:
        return None


class KaggleApiError(RuntimeError):
    """Upstream Kaggle call failed."""


def _request(
    token: str,
    method: str,
    path: str,
    *,
    body: bytes | None = None,
    content_type: str = "application/json",
) -> tuple[int, bytes]:
    token = (token or "").strip()
    if not token:
        raise KaggleApiError("no Kaggle API key configured")

    headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = content_type

    request = Request(f"{BASE_URL}{path}", data=body, headers=headers, method=method)
    try:
        with urlopen(request, timeout=TIMEOUT_SECONDS) as response:
            return response.status, response.read()
    except HTTPError as exc:
        return exc.code, exc.read()
    except URLError as exc:
        raise KaggleApiError(f"could not reach kaggle: {exc.reason}") from exc


def detect_username(token: str) -> str | None:
    """Resolve the token's owner from their own kernels or datasets.

    Kaggle has no whoami endpoint, but ``group=profile`` lists the
    authenticated user's items, and the author field carries the username.
    """
    for path in (
        "/kernels/list?group=profile&pageSize=1",
        "/datasets/list?group=profile&pageSize=1",
    ):
        code, body = _request(token, "GET", path)
        if code != 200:
            continue
        try:
            items = json.loads(body)
        except json.JSONDecodeError:
            continue
        if not isinstance(items, list) or not items:
            continue
        item = items[0]
        if not isinstance(item, dict):
            continue
        author = item.get("author")
        if not isinstance(author, str) or not author:
            ref = item.get("ref")
            if isinstance(ref, str) and "/" in ref:
                author = ref.split("/", 1)[0]
        if author:
            return author
    return None


def split_ref(ref: str | None = None) -> tuple[str, str]:
    value = (ref or os.getenv("KAGGLE_KERNEL") or DEFAULT_KERNEL).strip()
    if "/" not in value:
        raise KaggleApiError(f"kernel ref must look like user/slug, got {value!r}")
    user, slug = value.split("/", 1)
    return user, slug


def kernel_status(token: str, user: str, slug: str) -> str:
    """Return a status string, or ``idle`` when Kaggle has no run for the kernel."""
    code, body = _request(
        token, "GET", f"/kernels/status?userName={user}&kernelSlug={slug}"
    )

    if code == 404:
        return "idle"
    if code != 200:
        raise KaggleApiError(
            f"status failed ({code}): {body[:300].decode(errors='replace')}"
        )

    text = body.decode(errors="replace").strip().strip('"')
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return text or "queued"

    if isinstance(parsed, str):
        return parsed
    if isinstance(parsed, dict):
        for key in ("status", "kernelWorkerStatus", "state"):
            value = parsed.get(key)
            if isinstance(value, str):
                return value
    return "queued"


RUNNING_STATES = {"queued", "starting", "running", "cancelrequested"}


def is_running(status: str) -> bool:
    """True only for states that actually hold a session.

    ``cancelAcknowledged`` and ``complete`` mean the run is over, and a push is
    allowed again.
    """
    return status.strip().lower() in RUNNING_STATES


def pull_kernel(token: str, user: str, slug: str) -> tuple[dict, str]:
    code, body = _request(
        token, "GET", f"/kernels/pull?userName={user}&kernelSlug={slug}"
    )
    if code != 200:
        raise KaggleApiError(
            f"pull failed ({code}): {body[:300].decode(errors='replace')}"
        )

    payload = json.loads(body)
    metadata = payload.get("metadata") or {}
    source = (payload.get("blob") or {}).get("source")
    if not source:
        raise KaggleApiError("kaggle returned no notebook source")
    return metadata, source


def _clean_notebook(source: str) -> str:
    """Strip cell outputs and flatten sources, exactly like the official client."""
    try:
        notebook = json.loads(source)
    except json.JSONDecodeError:
        return source

    for cell in notebook.get("cells", []):
        if not isinstance(cell, dict):
            continue
        if cell.get("cell_type") == "code":
            cell.pop("outputs", None)
            cell["execution_count"] = None
        lines = cell.get("source")
        if isinstance(lines, list):
            cell["source"] = "".join(lines)

    return json.dumps(notebook)


def _refs(items: list | None) -> list[str]:
    refs: list[str] = []
    for item in items or []:
        if isinstance(item, dict):
            value = item.get("ref")
            if isinstance(value, str):
                refs.append(value)
        elif isinstance(item, str):
            refs.append(item)
    return refs


def inject_worker_key(source: str, key_payload: dict) -> str:
    """Replace the notebook's Kaggle-secret cell with this account's key.

    The pushed notebook is private to the user, so embedding their revocable
    worker key is acceptable and removes the need to configure a secret.
    """
    try:
        notebook = json.loads(source)
    except json.JSONDecodeError:
        return source
    cells = notebook.get("cells")
    if not isinstance(cells, list):
        return source

    code = (
        "import json, pathlib\n"
        "p = pathlib.Path('/root/.imagineit')\n"
        "p.mkdir(parents=True, exist_ok=True)\n"
        "f = p / 'gpu-kaggle.worker.json'\n"
        f"f.write_text(json.dumps({key_payload!r}))\n"
        "f.chmod(0o600)\n"
    )

    target = None
    for cell in cells:
        if not isinstance(cell, dict):
            continue
        text = "".join(cell.get("source") or [])
        if "IMAGINEIT_WORKER_KEY" in text or "gpu-kaggle.worker.json" in text:
            target = cell
            break

    replacement = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [code],
    }
    if target is not None:
        target.clear()
        target.update(replacement)
    else:
        cells.insert(0, replacement)
    return json.dumps(notebook)


def _strip_key_cells(source: str) -> list[tuple[str, str]]:
    """Notebook shape ignoring the worker-key cell, which changes per account."""
    try:
        notebook = json.loads(source)
    except json.JSONDecodeError:
        return []
    cells = []
    for cell in notebook.get("cells", []):
        if not isinstance(cell, dict):
            continue
        text = "".join(cell.get("source") or [])
        if "gpu-kaggle.worker.json" in text or "IMAGINEIT_WORKER_KEY" in text:
            continue
        cells.append((cell.get("cell_type", ""), text.strip()))
    return cells


def notebooks_equivalent(left: str, right: str) -> bool:
    return _strip_key_cells(left) == _strip_key_cells(right)


def build_push(
    token: str,
    ref: str | None = None,
    *,
    enable_gpu: bool = True,
    worker_key: dict | None = None,
) -> dict:
    """Pull the notebook and build the push body, without sending it.

    If the target notebook does not exist under this user yet, pull the app's
    template with the server key and push it as theirs, which creates it.
    """
    user, slug = split_ref(ref)
    metadata: dict = {}
    target_source: str | None = None
    try:
        metadata, target_source = pull_kernel(token, user, slug)
    except KaggleApiError as exc:
        message = str(exc)
        if "pull failed (403)" not in message and "pull failed (404)" not in message:
            raise

    # The app's notebook is canonical: an existing kernel is overwritten when
    # its content drifted, so updates ship on the next connect.
    template = load_local_template()
    if template is None:
        template_token = (os.getenv("KAGGLE_KEY") or "").strip()
        if not template_token:
            raise KaggleApiError("notebook missing and no template is configured")
        template_user, template_slug = split_ref(
            os.getenv("KAGGLE_TEMPLATE", DEFAULT_KERNEL)
        )
        _, template = pull_kernel(template_token, template_user, template_slug)

    changed = target_source is None or not notebooks_equivalent(template, target_source)
    source = (
        inject_worker_key(template, worker_key) if worker_key is not None else template
    )

    # `/kernels/push` wants `id` as the integer kernel id and `slug` as the
    # full "user/slug". For a new kernel only the slug is sent.
    kernel_id = metadata.get("id")
    payload: dict = {
        "slug": f"{user}/{slug}",
        "newTitle": metadata.get("title") or slug,
        "text": _clean_notebook(source),
        "language": metadata.get("language") or "python",
        "kernelType": metadata.get("kernelType") or "notebook",
        "isPrivate": bool(metadata.get("isPrivate", True)),
        "enableGpu": enable_gpu,
        "enableTpu": False,
        "enableInternet": bool(metadata.get("enableInternet", True)),
        "datasetDataSources": _refs(metadata.get("datasetDataSources")),
        "competitionDataSources": _refs(metadata.get("competitionDataSources")),
        "kernelDataSources": _refs(metadata.get("kernelDataSources")),
        "modelDataSources": _refs(metadata.get("modelDataSources")),
        "categoryIds": metadata.get("categoryIds") or [],
    }

    if isinstance(kernel_id, int):
        payload["id"] = kernel_id

    if enable_gpu:
        payload["machineShape"] = metadata.get("machineShape") or GPU_SHAPE

    docker_image = metadata.get("dockerImage")
    if docker_image:
        payload["dockerImage"] = docker_image

    return payload, changed


def start_kernel(
    token: str,
    ref: str | None = None,
    *,
    enable_gpu: bool = True,
    worker_key: dict | None = None,
) -> dict:
    """Push the existing notebook, which starts a run."""
    user, slug = split_ref(ref)
    payload, changed = build_push(
        token, ref, enable_gpu=enable_gpu, worker_key=worker_key
    )

    code, response = _request(
        token,
        "POST",
        "/kernels/push",
        body=json.dumps(payload).encode("utf-8"),
    )
    if code not in (200, 201):
        raise KaggleApiError(
            f"push failed ({code}): {response[:400].decode(errors='replace')}"
        )

    saved = json.loads(response) if response.strip() else {}
    return {
        "ref": f"{user}/{slug}",
        "enableGpu": bool(payload.get("enableGpu")),
        "machineShape": payload.get("machineShape"),
        "versionNumber": saved.get("versionNumber"),
        "status": kernel_status(token, user, slug),
        "updated": changed,
    }
