"""Stage-agnostic Gemini Batch API driver: submit a JSONL, poll, download.

A batch job runs asynchronously (target 24h, usually faster). Job names/state are
recorded to outputs/batch_jobs.json so a run can be resumed across invocations,
and a chunk whose results already exist on disk is skipped.
"""
import json
import time
from pathlib import Path

from config_pipeline import BATCH_JOBS_FILE, RAW_DIR
from gemini_client import client

TERMINAL = {"JOB_STATE_SUCCEEDED", "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED", "JOB_STATE_EXPIRED"}


# ---------------------------------------------------------------------------
# batch_jobs.json bookkeeping
# ---------------------------------------------------------------------------
def load_jobs() -> dict:
    if BATCH_JOBS_FILE.exists():
        return json.loads(BATCH_JOBS_FILE.read_text())
    return {}


def save_jobs(jobs: dict):
    BATCH_JOBS_FILE.parent.mkdir(parents=True, exist_ok=True)
    BATCH_JOBS_FILE.write_text(json.dumps(jobs, indent=2))


def record(chunk_id: str, **fields):
    jobs = load_jobs()
    jobs.setdefault(chunk_id, {}).update(fields)
    save_jobs(jobs)


# ---------------------------------------------------------------------------
# Submit / poll / download
# ---------------------------------------------------------------------------
def submit(chunk_id: str, jsonl_path: Path, model: str) -> str:
    """Upload the JSONL and create a batch job for `model`. Returns the job name.
    `model` is passed explicitly so this runner stays stage-agnostic."""
    from google.genai import types
    up = client().files.upload(
        file=str(jsonl_path),
        config=types.UploadFileConfig(display_name=f"{chunk_id}-requests",
                                      mime_type="jsonl"),
    )
    job = client().batches.create(
        model=model, src=up.name,
        config=types.CreateBatchJobConfig(display_name=f"stage1-{chunk_id}"),
    )
    st = _state_name(job)
    record(chunk_id, job_name=job.name, state=st,
           jsonl=str(jsonl_path), input_file=up.name)
    print(f"  submitted {chunk_id}: {job.name} ({st})")
    return job.name or ""


def _state_name(job) -> str:
    """job.state is Optional per the SDK type; normalize to a plain string."""
    return job.state.name if job.state else "JOB_STATE_UNSPECIFIED"


def state(job_name: str) -> str:
    return _state_name(client().batches.get(name=job_name))


def download_results(job_name: str, out_path: Path) -> bool:
    """If the job succeeded, download its results JSONL to out_path. Returns True
    on success. Raises on FAILED/CANCELLED/EXPIRED with the reported error."""
    job = client().batches.get(name=job_name)
    st = _state_name(job)
    if st != "JOB_STATE_SUCCEEDED":
        if st in TERMINAL:
            raise RuntimeError(f"batch {job_name} ended {st}: {getattr(job, 'error', None)}")
        return False  # still running
    if job.dest is None or not job.dest.file_name:
        raise RuntimeError(f"batch {job_name} succeeded but has no result file")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    data = client().files.download(file=job.dest.file_name)
    out_path.write_bytes(data)
    return True


def wait(job_name: str, out_path: Path, poll_secs: int = 30, max_wait: int = 26 * 3600):
    """Block until the job reaches a terminal state, downloading results on
    success. Returns out_path. Prints state transitions."""
    waited, last = 0, None
    while True:
        st = state(job_name)
        if st != last:
            print(f"  {job_name}: {st}")
            last = st
        if st == "JOB_STATE_SUCCEEDED":
            download_results(job_name, out_path)
            return out_path
        if st in TERMINAL:
            raise RuntimeError(f"batch {job_name} ended {st}")
        if waited >= max_wait:
            raise TimeoutError(f"batch {job_name} still {st} after {waited}s")
        time.sleep(poll_secs)
        waited += poll_secs


def results_path(chunk_id: str) -> Path:
    return RAW_DIR / f"{chunk_id}_results.jsonl"
