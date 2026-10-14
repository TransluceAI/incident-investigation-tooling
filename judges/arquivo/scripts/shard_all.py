"""Assign every prepared corpus unit once; no reserved pilot inputs or controls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from arquivo_judge.common import read_jsonl, sha256_file, sha256_text, write_jsonl
from arquivo_judge.evidence import validate_chunk_map
from arquivo_judge.pipeline import save_json

p = argparse.ArgumentParser()
p.add_argument('--run', required=True, type=Path)
p.add_argument('--out', required=True, type=Path)
a = p.parse_args()
units = read_jsonl(a.run / 'units.jsonl')
ids = {u['unit_id'] for u in units}
assert units and len(ids) == len(units), 'Empty or duplicate canonical inputs'
assert not any(uid.startswith('control-') for uid in ids)
for u in units:
    assert sha256_text(u['text']) == u['text_sha256']
    assert u['prompt_tokens'] <= 160000
    validate_chunk_map(u)
shards, loads = [[] for _ in range(8)], [0] * 8
for u in sorted(units, key=lambda u: (-u['prompt_tokens'], u['unit_id'])):
    n = min(range(8), key=lambda i: loads[i])
    shards[n].append(u)
    loads[n] += u['prompt_tokens']
a.out.mkdir(parents=True, exist_ok=False)
for i, rows in enumerate(shards):
    write_jsonl(a.out / f'part-{i}.jsonl', rows)
# Read the committed files back: verify IDs and full records, not just counts.
canonical = {u['unit_id']: u for u in units}
assigned = []
for i in range(8):
    for u in read_jsonl(a.out / f'part-{i}.jsonl'):
        assert u == canonical[u['unit_id']]
        assigned.append(u['unit_id'])
assert len(assigned) == len(set(assigned)) == len(units)
assert set(assigned) == ids
summary = {'units': len(units), 'all_corpus_units_assigned_once': True,
           'reserved_or_excluded_corpus_units': 0, 'synthetic_controls': 0,
           'shard_units': [len(s) for s in shards], 'shard_prompt_tokens': loads,
           'total_prompt_tokens': sum(loads),
           'config_sha256': sha256_file(a.run / 'config.json'),
           'canonical_manifest_sha256': sha256_file(a.run / 'units.jsonl'),
           'shard_sha256': {f'part-{i}.jsonl': sha256_file(a.out / f'part-{i}.jsonl') for i in range(8)}}
save_json(a.run / 'full-assignment.json', summary)
print(json.dumps(summary, indent=2))
