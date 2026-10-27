"""Validate one committed (window, axis) result against its prepared unit.

Checks enforce hash identity, schema validity, exact
chunk resolution, and supported_success only for a response from a
non-archive system. They also enforce what the grammar does not: a yes needs at
least one resolved citation. V2 actions carry their own citations; saved v1
schemas and their segment cross-references remain supported.
"""
from collections import defaultdict

from .common import sha256_text
from .render import validate_chunk_map


class ValidationError(ValueError):
    pass


def _closest(errors):
    """Leaf schema errors; for an anyOf failure, those of the branches that came closest."""
    leaves = []
    for error in errors:
        if not error.context:
            leaves.append(error)
            continue
        branches = defaultdict(list)
        for e in error.context:
            branches[e.relative_schema_path[0]].append(e)
        fewest = min(map(len, branches.values()))
        leaves += _closest([e for b in branches.values() if len(b) == fewest for e in b])
    return leaves


def validate(row: dict, unit: dict, schema: dict) -> list:
    """Return resolved evidence entries; raise ValidationError on any failure."""
    from jsonschema import Draft202012Validator
    if row.get("status") != "success":
        raise ValidationError("Runner failure: " + str(row.get("error"))[:300])
    # The row's request hash binds it to this window's prepared request (report.py).
    if sha256_text(unit["text"]) != unit["text_sha256"]:
        raise ValidationError("Input hash mismatch")
    result = row.get("result")
    errors = _closest(Draft202012Validator(schema).iter_errors(result))
    if errors:
        raise ValidationError("Schema: " + "; ".join(dict.fromkeys(
            f"{'/'.join(map(str, e.absolute_path)) or 'result'}: {e.message}" for e in errors))[:300])
    try:
        validate_chunk_map(unit)
    except (ValueError, KeyError, TypeError) as exc:
        raise ValidationError(f"Chunk map: {exc}") from exc
    resolved, seen = [], set()
    entries = list(result["evidence"])
    for action in result.get("actions", []):
        entries.extend(action.get("evidence", []))
    for entry in entries:
        # JSON Schema treats integer-valued numbers such as 1.0 as integers.
        # Schema validation above excludes booleans and fractional values.
        sid, idx = entry["segment_id"], int(entry["chunk_id"])
        chunks = unit["evidence_chunks"].get(sid, [])
        if not 1 <= idx <= len(chunks):
            raise ValidationError(f"Unknown evidence chunk: {sid}:{idx}")
        if (sid, idx) in seen:
            continue
        seen.add((sid, idx))
        c = chunks[idx - 1]
        resolved.append({"segment_id": sid, "chunk_id": idx, "location": entry["location"],
                         "source": unit["segment_sources"][sid], "location_verified": False,
                         "source_start": c["source_start"],
                         "length": c["source_end"] - c["source_start"],
                         "quote": unit["text"][c["text_start"]:c["text_end"]]})
    if result["label"] == "yes" and not resolved:
        raise ValidationError("Positive label without resolved evidence")
    for action in result.get("actions", []):
        ids = action.get("evidence_segment_ids")  # legacy prepared schemas remain readable
        if ids is not None and (not ids or not set(ids) <= {e["segment_id"] for e in resolved}):
            raise ValidationError("Action requires segments with resolved evidence citations")
        if action["outcome"] == "supported_success" and (
                action["observed_stage"] not in {"observed_response", "demonstrated_outcome"}
                or action["responding_system"] in {"archive_replay", "unknown"}):
            raise ValidationError("Unsupported success stage/system")
    return resolved
