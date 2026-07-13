#!/usr/bin/env python3
"""
Shared utilities for the luxury-scoring model-comparison experiment.

Replicates the reference implementation in Research/llm_image_rank_test/score.py:
same prompt construction (system prompt + anchor grid + target image), same
schema (reasoning, score, level, confidence, valid) extended with the model's
own room_type judgment, ported to the OpenAI, Gemini, and Anthropic SDKs with
each provider's strict structured-output mechanism.

API keys come from config.Config at the repo root (never from a local .env).
"""

import base64
import json
import sys
import threading
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from config import Config  # noqa: E402

# ---------------------------------------------------------------------------
# Rooms and assets (reference prompts + anchor grids from the earlier experiment)
# ---------------------------------------------------------------------------
ASSETS_ROOT = REPO_ROOT / "Research" / "llm_image_rank_test"

ROOM_TYPES = ["kitchen", "bathroom", "bedroom", "living_room"]
CLASSIFY_LABELS = ROOM_TYPES + ["other"]

# kitchen anchors go 1-7; the other rooms go 1-8 (per each prompt.txt)
ROOMS = {
    "kitchen":     {"assets_dir": ASSETS_ROOT / "kitchen",  "max_level": 7},
    "bathroom":    {"assets_dir": ASSETS_ROOT / "bathroom", "max_level": 8},
    "bedroom":     {"assets_dir": ASSETS_ROOT / "bedroom",  "max_level": 8},
    "living_room": {"assets_dir": ASSETS_ROOT / "living",   "max_level": 8},
}

VALID_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}

# ---------------------------------------------------------------------------
# Models under comparison
# ---------------------------------------------------------------------------
MODELS = {
    "gpt-4o":            {"provider": "openai"},
    "gemini-3.5-flash":  {"provider": "gemini"},
    "gemini-2.5-flash":  {"provider": "gemini"},
    "claude-haiku-4-5":  {"provider": "anthropic"},
    "claude-sonnet-4-6": {"provider": "anthropic"},
}

CLASSIFIER_MODEL = "gpt-4o"  # Stage 1 uses this ONLY, image detail "low"

# $ per 1M tokens, standard tier, as of 2026-07. Batch = 50% off for all five.
# cached_input = price for cache-read tokens; cache_write = Anthropic's 1.25x
# write premium (OpenAI/Gemini have no write premium).
PRICING = {
    "gpt-4o":            {"input": 2.50, "output": 10.00, "cached_input": 1.25,  "cache_write": 2.50},
    "gemini-3.5-flash":  {"input": 1.50, "output": 9.00,  "cached_input": 0.15,  "cache_write": 1.50},
    "gemini-2.5-flash":  {"input": 0.30, "output": 2.50,  "cached_input": 0.075, "cache_write": 0.30},
    "claude-haiku-4-5":  {"input": 1.00, "output": 5.00,  "cached_input": 0.10,  "cache_write": 1.25},
    "claude-sonnet-4-6": {"input": 3.00, "output": 15.00, "cached_input": 0.30,  "cache_write": 3.75},
}
BATCH_DISCOUNT = 0.5

MAX_RETRIES = 4
IMAGE_DETAIL_SCORING = "high"   # matches score.py
IMAGE_DETAIL_CLASSIFY = "low"   # per experiment spec


class RateLimiter:
    """Thread-safe token bucket for pacing calls under a tokens-per-minute (TPM)
    budget. Refills at tpm/60 tokens/sec, capacity = tpm. acquire() reserves an
    estimate before a call; charge() reconciles with the actual token count
    (positive = consumed more, negative = refund)."""

    def __init__(self, tpm: float):
        self.rate = tpm / 60.0
        self.capacity = float(tpm)
        self.tokens = float(tpm)
        self.ts = time.monotonic()
        self.lock = threading.Lock()

    def _refill(self):
        now = time.monotonic()
        self.tokens = min(self.capacity, self.tokens + (now - self.ts) * self.rate)
        self.ts = now

    def acquire(self, need: float):
        need = min(need, self.capacity)  # a single call can't need more than the bucket holds
        while True:
            with self.lock:
                self._refill()
                if self.tokens >= need:
                    self.tokens -= need
                    return
                wait = (need - self.tokens) / self.rate
            time.sleep(wait)

    def charge(self, delta: float):
        with self.lock:
            self.tokens = min(self.capacity, self.tokens - delta)


# Optional per-provider limiters, off unless a stage sets one (e.g. Stage 1 on a
# low-TPM OpenAI tier). Keyed by provider name.
_limiters = {}


def set_rate_limit(provider: str, tpm):
    """tpm=None/0 disables. Estimate tokens per call are reconciled from usage."""
    _limiters[provider] = RateLimiter(tpm) if tpm else None


def compute_cost(model: str, input_tokens: int, output_tokens: int,
                 cached_input_tokens: int = 0, cache_write_tokens: int = 0,
                 batch: bool = False) -> float:
    """input_tokens is the TOTAL prompt size (providers are normalized to this
    in the _call_* functions); cached/write portions bill at their own rates."""
    p = PRICING[model]
    fresh = max(input_tokens - cached_input_tokens - cache_write_tokens, 0)
    cost = (fresh * p["input"]
            + cached_input_tokens * p["cached_input"]
            + cache_write_tokens * p["cache_write"]
            + output_tokens * p["output"]) / 1_000_000
    return cost * BATCH_DISCOUNT if batch else cost


# ---------------------------------------------------------------------------
# Image helpers (as in score.py)
# ---------------------------------------------------------------------------
def mime_for(path: Path) -> str:
    ext = path.suffix.lower()
    return {
        ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".png": "image/png", ".webp": "image/webp", ".gif": "image/gif",
    }.get(ext, "image/jpeg")


def to_data_url(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:{mime_for(path)};base64,{b64}"


def to_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def load_room_assets(room_type: str):
    """Return (prompt_text, grid_path) for a room type."""
    cfg = ROOMS[room_type]
    prompt = (cfg["assets_dir"] / "prompt.txt").read_text(encoding="utf-8")
    grid = cfg["assets_dir"] / "grid.png"
    return prompt, grid


# ---------------------------------------------------------------------------
# Schemas — score.py's LUXURY_SCHEMA + the model's own room-type judgment.
# The level enum is parameterized per room (kitchen 1-7, others 1-8).
# ---------------------------------------------------------------------------
def luxury_schema(room_type: str) -> dict:
    levels = list(range(1, ROOMS[room_type]["max_level"] + 1))
    # living_room scores are integer-only (no decimals); other rooms keep the
    # continuous 1.0-8.0 score.
    score_field = {"type": "integer", "enum": levels} if room_type == "living_room" \
        else {"type": "number"}
    return {
        "type": "object",
        "properties": {
            "reasoning":  {"type": "string"},
            "score":      score_field,
            "level":      {"type": "integer", "enum": levels},
            "confidence": {"type": "number"},
            "valid":      {"type": "boolean"},
            "room_type":  {"type": "string", "enum": CLASSIFY_LABELS},
        },
        "required": ["reasoning", "score", "level", "confidence", "valid", "room_type"],
        "additionalProperties": False,
    }


CLASSIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "room_type": {"type": "string", "enum": CLASSIFY_LABELS},
        "other_label": {
            "type": ["string", "null"],
            "description": 'When room_type is "other": a short lowercase label for what '
                           'the image actually shows, e.g. "dining room", "floorplan", '
                           '"exterior", "hallway", "garage", "backyard". Null otherwise.',
        },
    },
    "required": ["room_type", "other_label"],
    "additionalProperties": False,
}

CLASSIFY_PROMPT = (
    "Classify the room shown in this real-estate listing photo. "
    "Set room_type to exactly one of: kitchen, bathroom, bedroom, living_room, other. "
    "Use \"other\" for exteriors, floor plans, hallways, dining rooms, offices, "
    "garages, yards, closets, laundry rooms, or anything that is not clearly one "
    "of the four listed room types — and in that case also set other_label to a "
    "short label describing what the image shows; otherwise set other_label to null."
)

# Extra scoring instruction so every model reports its own room-type judgment
# (score.py's prompt predates this field).
ROOM_JUDGMENT_NOTE = (
    '\n\nAdditionally, set "room_type" to your own judgment of what room is shown '
    "in the image to score (one of: kitchen, bathroom, bedroom, living_room, other), "
    "regardless of what room type the rubric assumes."
)


def _gemini_schema(schema: dict) -> dict:
    """Gemini responseSchema doesn't accept integer enums; use min/max instead."""
    out = json.loads(json.dumps(schema))  # deep copy
    props = out.get("properties", {})
    for prop in props.values():
        if prop.get("type") == "integer" and "enum" in prop:
            levels = prop.pop("enum")
            prop["minimum"] = min(levels)
            prop["maximum"] = max(levels)
    out.pop("additionalProperties", None)
    return out


# ---------------------------------------------------------------------------
# Provider clients (lazy singletons)
# ---------------------------------------------------------------------------
_clients = {}


def _openai():
    if "openai" not in _clients:
        from openai import OpenAI
        if not Config.OPENAI_API_KEY:
            sys.exit("ERROR: OPENAI_API_KEY is empty — set it where config.py reads it from.")
        _clients["openai"] = OpenAI(api_key=Config.OPENAI_API_KEY)
    return _clients["openai"]


def _gemini():
    if "gemini" not in _clients:
        from google import genai
        if not Config.GEMINI_API_KEY:
            sys.exit("ERROR: GEMINI_API_KEY is empty — set it where config.py reads it from.")
        _clients["gemini"] = genai.Client(api_key=Config.GEMINI_API_KEY)
    return _clients["gemini"]


def _anthropic():
    if "anthropic" not in _clients:
        import anthropic
        if not Config.CLAUDE_API_KEY:
            sys.exit("ERROR: CLAUDE_API_KEY is empty — set it where config.py reads it from.")
        _clients["anthropic"] = anthropic.Anthropic(api_key=Config.CLAUDE_API_KEY)
    return _clients["anthropic"]


# ---------------------------------------------------------------------------
# One scoring / classification call per provider.
# All return a dict:
#   parsed          dict per schema (or raises on hard failure)
#   raw             full response JSON (string)
#   response_model  exact model/snapshot string reported by the API
#   input_tokens / output_tokens / cached_input_tokens / thought_tokens
#   latency_s
# ---------------------------------------------------------------------------
def _call_openai(model, system_text, image_parts, schema, schema_name, detail, cache_key):
    client = _openai()
    content = []
    for kind, val in image_parts:
        if kind == "text":
            content.append({"type": "text", "text": val})
        else:  # image path
            content.append({"type": "image_url",
                            "image_url": {"url": to_data_url(val), "detail": detail}})
    # Pace under the OpenAI TPM budget if a limiter is set; reserve an estimate
    # (bigger for the high-detail scoring prompt), reconcile with actual usage.
    limiter = _limiters.get("openai")
    est = 1200 if detail == IMAGE_DETAIL_CLASSIFY else 3200
    if limiter is not None:
        limiter.acquire(est)
    t0 = time.monotonic()
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_text},
                {"role": "user", "content": content},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {"name": schema_name, "strict": True, "schema": schema},
            },
            # OpenAI prompt caching is automatic (>=1024-token prefixes); a stable
            # key routes identical-prefix requests to the same cache shard.
            prompt_cache_key=cache_key,
        )
    except Exception:
        if limiter is not None:
            limiter.charge(-est)  # refund the reservation on failure
        raise
    latency = time.monotonic() - t0
    parsed = json.loads(resp.choices[0].message.content)
    usage = resp.usage
    if limiter is not None:  # reconcile estimate against real token spend (TPM counts both)
        limiter.charge((usage.prompt_tokens + usage.completion_tokens) - est)
    cached = getattr(getattr(usage, "prompt_tokens_details", None), "cached_tokens", 0) or 0
    return {
        "parsed": parsed,
        "raw": resp.model_dump_json(),
        "response_model": resp.model,
        "input_tokens": usage.prompt_tokens,  # includes cached tokens
        "output_tokens": usage.completion_tokens,
        "cached_input_tokens": cached,
        "cache_write_tokens": 0,
        "thought_tokens": 0,
        "latency_s": latency,
    }


def _call_gemini(model, system_text, image_parts, schema, schema_name, detail, cache_key):
    # Gemini implicit caching is automatic on 2.5+/3.x models for repeated
    # prefixes; the static prefix (system instruction + grid) already comes
    # first, which is the only requirement. Hits show up in
    # cached_content_token_count.
    from google.genai import types
    client = _gemini()
    contents = []
    for kind, val in image_parts:
        if kind == "text":
            contents.append(val)
        else:
            contents.append(types.Part.from_bytes(data=val.read_bytes(), mime_type=mime_for(val)))
    t0 = time.monotonic()
    resp = client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            system_instruction=system_text,
            temperature=0,
            max_output_tokens=8192,
            response_mime_type="application/json",
            response_schema=_gemini_schema(schema),
        ),
    )
    latency = time.monotonic() - t0
    parsed = json.loads(resp.text)
    um = resp.usage_metadata
    thought = getattr(um, "thoughts_token_count", 0) or 0
    out_tok = (um.candidates_token_count or 0) + thought  # thinking bills as output
    return {
        "parsed": parsed,
        "raw": resp.model_dump_json(),
        "response_model": getattr(resp, "model_version", None) or model,
        "input_tokens": um.prompt_token_count or 0,  # includes cached tokens
        "output_tokens": out_tok,
        "cached_input_tokens": getattr(um, "cached_content_token_count", 0) or 0,
        "cache_write_tokens": 0,
        "thought_tokens": thought,
        "latency_s": latency,
    }


def _call_anthropic(model, system_text, image_parts, schema, schema_name, detail, cache_key):
    client = _anthropic()
    content = []
    for kind, val in image_parts:
        if kind == "text":
            content.append({"type": "text", "text": val})
        else:
            content.append({"type": "image", "source": {
                "type": "base64", "media_type": mime_for(val), "data": to_b64(val)}})
    # Anthropic caching is explicit. Two breakpoints: one on the last static
    # block (tools + system + the room's grid prefix — shared by every image
    # of that room type), and one on the final block so the 5 back-to-back
    # replicates of the same image read the target image from cache too.
    # Silently no-ops below the model's minimum cacheable prefix size.
    if len(content) >= 2:
        content[-2]["cache_control"] = {"type": "ephemeral"}
    content[-1]["cache_control"] = {"type": "ephemeral"}
    tool = {
        "name": schema_name,
        "description": "Record the structured result. Always call this tool.",
        "strict": True,
        "input_schema": schema,
    }
    t0 = time.monotonic()
    resp = client.messages.create(
        model=model,
        max_tokens=1024,
        temperature=0,
        system=system_text,
        messages=[{"role": "user", "content": content}],
        tools=[tool],
        tool_choice={"type": "tool", "name": schema_name},
    )
    latency = time.monotonic() - t0
    tool_use = next(b for b in resp.content if b.type == "tool_use")
    cache_read = getattr(resp.usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(resp.usage, "cache_creation_input_tokens", 0) or 0
    return {
        "parsed": tool_use.input,
        "raw": resp.to_json(),
        # Anthropic's input_tokens EXCLUDES cache reads/writes; normalize to total
        "response_model": resp.model,
        "input_tokens": resp.usage.input_tokens + cache_read + cache_write,
        "output_tokens": resp.usage.output_tokens,
        "cached_input_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "thought_tokens": 0,
        "latency_s": latency,
    }


_PROVIDER_FNS = {"openai": _call_openai, "gemini": _call_gemini, "anthropic": _call_anthropic}


def call_with_retry(model, system_text, image_parts, schema, schema_name,
                    detail=IMAGE_DETAIL_SCORING, max_retries=MAX_RETRIES,
                    cache_key=None):
    """One structured-output call with exponential-backoff retry (as score.py)."""
    fn = _PROVIDER_FNS[MODELS.get(model, {"provider": "openai"})["provider"]]
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(model, system_text, image_parts, schema, schema_name, detail, cache_key)
        except Exception as e:
            last_err = e
            wait = 2 ** attempt
            print(f"    call failed (attempt {attempt}/{max_retries}): {e} -- retrying in {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"giving up after {max_retries} attempts: {last_err}")


# ---------------------------------------------------------------------------
# High-level operations
# ---------------------------------------------------------------------------
def classify_image(image_path: Path) -> dict:
    """Stage 1: one gpt-4o classification call, image detail low, strict enum."""
    return call_with_retry(
        CLASSIFIER_MODEL,
        CLASSIFY_PROMPT,
        [("text", "Photo to classify:"), ("image", image_path)],
        CLASSIFY_SCHEMA,
        "room_classification",
        detail=IMAGE_DETAIL_CLASSIFY,
        cache_key="compedge-room-classify-v1",
    )


def score_image(model: str, room_type: str, image_path: Path) -> dict:
    """
    Stage 3: one scoring call replicating score.py's message structure:
    system prompt, then user content = [grid intro text, anchor grid image,
    "to score" text, target image].
    """
    prompt_text, grid_path = load_room_assets(room_type)
    max_level = ROOMS[room_type]["max_level"]
    label = room_type.replace("_", " ").upper()
    parts = [
        ("text", f"REFERENCE GRID (anchor examples, Level 1 = lowest ... Level {max_level} = highest):"),
        ("image", grid_path),
        ("text", f"{label} TO SCORE:"),
        ("image", image_path),
    ]
    return call_with_retry(
        model,
        prompt_text + ROOM_JUDGMENT_NOTE,
        parts,
        luxury_schema(room_type),
        "luxury_score",
        detail=IMAGE_DETAIL_SCORING,
        cache_key=f"compedge-luxury-score-{room_type}-v1",
    )
