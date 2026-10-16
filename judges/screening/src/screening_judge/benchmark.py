"""Adapt neutral benchmark inputs without opening labels or review provenance.

The recorded benchmark contract is id/document_path/text_sha256/metadata.
Only source_family and format are carried into prompts. Case IDs and document
paths remain outside prompt provenance; they can encode review/label hints.
"""
import json
from pathlib import Path

from .common import sha256_text
from .records import iter_source_records


def build_items(root, manifest, gaps):
    root = Path(root).resolve()
    path = Path(manifest)
    if not path.is_absolute():
        path = root / path
    seen, items = set(), []
    for line, row, error in iter_source_records(path):
        where = {"source": "benchmark", "file": str(path), "line": line}
        if (error or not isinstance(row, dict)
                or any(not isinstance(row.get(k), str) or not row[k]
                       for k in ("id", "document_path", "text_sha256"))
                or not isinstance(row.get("metadata", {}), dict)):
            gaps.append({**where, "gap": "unreadable_benchmark_input", "error": error})
            continue
        if row["id"] in seen:
            raise ValueError("Duplicate benchmark case ID: " + row["id"])
        seen.add(row["id"])
        meta = {key: value for key, value in row.get("metadata", {}).items()
                if key in {"source_family", "format"} and isinstance(value, str)}
        shape = "wiki" if meta.get("source_family") == "wiki" else "text"
        iid = "benchmark-" + sha256_text(row["id"])[:24]
        items.append({"item_id": iid, "benchmark_case_id": row["id"], "source": "benchmark",
                      "shape": shape, "kinds": {"benchmark_input"}, "time_basis": set(),
                      "observations": [{"observation_id": row["id"], "kind": "benchmark_input",
                                        "line": json.dumps(meta, sort_keys=True, ensure_ascii=False)}],
                      "request_urls": [], "documents": [(row["document_path"], row["text_sha256"])],
                      "redirects": []})
    return items
