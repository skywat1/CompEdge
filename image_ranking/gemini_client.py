"""Gemini plumbing for Stage-1 classification.

Self-contained port of the pieces used in
Research/model_comparison/common.py (message layout, schema, cost model),
extended with media_resolution + a batch-JSONL request builder. Nothing is
imported from Research/. API key comes from the repo-root config.py.
"""
import base64
import io
import json
import threading
from pathlib import Path

from PIL import Image

from config_pipeline import (LABELS, STAGE1_MEDIA_RESOLUTION, STAGE1_MODEL,
                             PRICE_INPUT, PRICE_OUTPUT, BATCH_DISCOUNT,
                             STAGE1_RESIZE_MAX_DIM)

# repo-root config.py holds the keys
import sys
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from config import Config  # noqa: E402

# ---------------------------------------------------------------------------
# Prompt + schema (floorplan is now a first-class label)
# ---------------------------------------------------------------------------
CLASSIFY_PROMPT = (
    "Classify the room shown in this real-estate listing photo. "
    "Set room_type to exactly one of: kitchen, bathroom, bedroom, living_room, "
    "floorplan, other. "
    'Use "floorplan" for architectural floor plans or layout diagrams. '
    'Use "other" for exteriors, hallways, dining rooms, offices, garages, yards, '
    "closets, laundry rooms, or anything not clearly one of the five listed types "
    '— and in that case also set other_label to a short lowercase label (e.g. '
    '"dining room", "exterior", "hallway"); otherwise set other_label to null.'
)


def classify_schema() -> dict:
    """Strict response schema in Gemini's responseSchema shape (nullable string,
    no additionalProperties)."""
    return {
        "type": "object",
        "properties": {
            "room_type": {"type": "string", "enum": LABELS},
            "other_label": {"type": "string", "nullable": True},
        },
        "required": ["room_type", "other_label"],
    }


# ---------------------------------------------------------------------------
# Client (lazy singleton)
# ---------------------------------------------------------------------------
_client = None
_client_lock = threading.Lock()


def client():
    """Thread-safe lazy singleton — a race here creates multiple httpx clients and
    can leave threads holding a closed one ("client has been closed")."""
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                from google import genai
                if not Config.GEMINI_API_KEY:
                    raise SystemExit("GEMINI_API_KEY is empty — set it where config.py reads it.")
                _client = genai.Client(api_key=Config.GEMINI_API_KEY)
    return _client


# ---------------------------------------------------------------------------
# Image encoding (resize is transport-only; media_resolution sets the token cost)
# ---------------------------------------------------------------------------
def encode_image(path: Path, max_dim: int = STAGE1_RESIZE_MAX_DIM) -> str:
    """Return base64 JPEG bytes, downscaled so batch files stay small."""
    im = Image.open(path).convert("RGB")
    im.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Cost
# ---------------------------------------------------------------------------
def compute_cost(input_tokens: int, output_tokens: int, batch: bool = True) -> float:
    cost = (input_tokens * PRICE_INPUT + output_tokens * PRICE_OUTPUT) / 1_000_000
    return cost * BATCH_DISCOUNT if batch else cost


# ---------------------------------------------------------------------------
# Live call (for the --live-check logic proof; default thinking, LOW resolution)
# ---------------------------------------------------------------------------
def classify_live(path: Path) -> dict:
    from google.genai import types
    r = client().models.generate_content(
        model=STAGE1_MODEL,
        contents=[types.Part.from_bytes(data=Path(path).read_bytes(),
                                        mime_type="image/jpeg"),
                  "Photo to classify:"],
        config=types.GenerateContentConfig(
            system_instruction=CLASSIFY_PROMPT,
            temperature=0,
            media_resolution=getattr(types.MediaResolution, STAGE1_MEDIA_RESOLUTION),
            response_mime_type="application/json",
            response_schema=classify_schema(),
        ),
    )
    um = r.usage_metadata  # Optional per SDK type; getattr keeps this None-safe
    parsed = json.loads(r.text or "")
    return {"room_type": parsed["room_type"], "other_label": parsed.get("other_label"),
            "response_model": getattr(r, "model_version", None) or STAGE1_MODEL,
            "input_tokens": getattr(um, "prompt_token_count", 0) or 0,
            "output_tokens": getattr(um, "candidates_token_count", 0) or 0,
            "thinking_tokens": getattr(um, "thoughts_token_count", 0) or 0}


# ---------------------------------------------------------------------------
# Batch JSONL: one request line per image, REST GenerateContentRequest shape.
# Casing verified against the SDK's by-alias serialization.
# ---------------------------------------------------------------------------
def build_jsonl_line(image_path: str) -> dict:
    """{"key": <image_path>, "request": <GenerateContentRequest>} for a batch file."""
    b64 = encode_image(_REPO / image_path)
    return {
        "key": image_path,
        "request": {
            "contents": [{
                "role": "user",
                "parts": [
                    {"inlineData": {"mimeType": "image/jpeg", "data": b64}},
                    {"text": "Photo to classify:"},
                ],
            }],
            "systemInstruction": {"parts": [{"text": CLASSIFY_PROMPT}]},
            "generationConfig": {
                "temperature": 0,
                "mediaResolution": STAGE1_MEDIA_RESOLUTION,
                "responseMimeType": "application/json",
                "responseSchema": classify_schema(),
            },
        },
    }


# ---------------------------------------------------------------------------
# Parse one result line from the downloaded results JSONL (REST camelCase).
# ---------------------------------------------------------------------------
def parse_result_obj(obj: dict) -> dict:
    """Return {key, room_type, other_label, response_model, input_tokens,
    output_tokens, thinking_tokens, error}. output_tokens is answer-only;
    thinking_tokens is separate (both bill as output). On any failure, error is
    set and room_type is None."""
    key = obj.get("key", "")
    resp = obj.get("response")
    fail = {"key": key, "room_type": None, "other_label": "", "response_model": "",
            "input_tokens": 0, "output_tokens": 0, "thinking_tokens": 0}
    if not resp or obj.get("error"):
        return {**fail,
                "error": json.dumps(obj.get("error") or obj.get("status") or "no response")}
    try:
        cands = resp.get("candidates") or []
        parts = cands[0]["content"]["parts"]
        text = "".join(p.get("text", "") for p in parts)
        parsed = json.loads(text)
        um = resp.get("usageMetadata", {}) or {}
        return {"key": key, "room_type": parsed["room_type"],
                "other_label": parsed.get("other_label") or "",
                "response_model": resp.get("modelVersion", "") or "",
                "input_tokens": um.get("promptTokenCount", 0) or 0,
                "output_tokens": um.get("candidatesTokenCount", 0) or 0,
                "thinking_tokens": um.get("thoughtsTokenCount", 0) or 0,
                "error": ""}
    except Exception as e:  # noqa: BLE001
        return {**fail, "error": f"{type(e).__name__}: {e}"}
