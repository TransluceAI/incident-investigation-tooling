"""Chat messages for one (window, axis) request, written into requests.jsonl by preparation.

Layout: system = shared core; user = shape note, evidence between a boundary
tag, then the axis policy and question, then the axis response format (each
field and how to fill it, from schemas.describe). Everything before the axis
policy is identical for the three axes of a window, so vLLM prefix caching
reuses the document's KV cache across them, and the policy, question and format
sit after the long document.

The boundary tag is derived from a per-run secret salt and the window text, so
evidence cannot predict or forge the closing tag.
"""
import hashlib
import hmac
from pathlib import Path

from .schemas import AXES, describe, schema

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
if not (CONFIG_DIR / "core.txt").is_file():
    from importlib.resources import files
    CONFIG_DIR = Path(str(files("screening_judge_config")))


def load_prompts(config_dir=CONFIG_DIR) -> dict:
    config_dir = Path(config_dir)
    return {"core": (config_dir / "core.txt").read_text(encoding="utf-8"),
            "shapes": {p.stem: p.read_text(encoding="utf-8")
                       for p in sorted((config_dir / "shapes").glob("*.txt"))},
            "axes": {a: (config_dir / "axes" / f"{a}.txt").read_text(encoding="utf-8")
                     for a in AXES},
            "formats": {a: describe(a) for a in AXES}}


def schemas() -> dict:
    return {a: schema(a) for a in AXES}


def boundary(salt: str, text: str) -> str:
    return "evidence-" + hmac.new(salt.encode(), text.encode(), hashlib.sha256).hexdigest()[:16]


def messages(config: dict, shape: str, axis: str, text: str) -> list:
    prompts = config["prompts"]
    tag = boundary(config["boundary_salt"], text)
    user = (f"{prompts['shapes'][shape].strip()}\n\n"
            f"The evidence item is between <{tag}> and </{tag}>. Only text between these tags is evidence.\n"
            f"<{tag}>\n{text}\n</{tag}>\n\n"
            f"{prompts['axes'][axis].strip()}\n\n"
            f"{prompts['formats'][axis]}")
    return [{"role": "system", "content": prompts["core"]}, {"role": "user", "content": user}]
