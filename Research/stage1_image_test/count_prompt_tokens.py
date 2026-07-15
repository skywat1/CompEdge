"""Count Stage-2 prefix tokens (room prompt + grid) against the Gemini tokenizer,
to decide whether the ~4096-token cache minimum (3.5-flash) is reachable."""
import sys
from pathlib import Path

REPO = Path("/Users/skyler/local/CompEdge")
sys.path.insert(0, str(REPO))
from config import Config

from google import genai
from google.genai import types

client = genai.Client(api_key=Config.GEMINI_API_KEY)
MODEL = "gemini-3.5-flash"
ROOMS_DIR = REPO / "Research" / "gemini_image_test" / "rooms"

print(f"{'room':<12} {'prompt_tok':>10} {'+grid HIGH':>11} {'clears 4096?':>13}")
GRID_HIGH = 1120  # media_resolution HIGH caps one image at 1120 tokens
for room in ["kitchen", "bathroom", "bedroom", "living_room"]:
    ptxt = (ROOMS_DIR / room / "prompt.txt").read_text(encoding="utf-8")
    n = client.models.count_tokens(model=MODEL, contents=ptxt).total_tokens
    total = n + GRID_HIGH
    print(f"{room:<12} {n:>10} {total:>11} {'YES' if total >= 4096 else 'no':>13}")
