"""Independent coverage and shard audit. Run after shard_all.py, before inference."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from arquivo_judge.evidence import validate_chunk_map


def check(condition, message):
    # Explicit checks, unlike assert, still run under python -O.
    if not condition:
        raise SystemExit('Audit failed: ' + message)


run = Path(sys.argv[1])
def rows(path):
    with path.open(encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

budget = json.loads((run / 'config.json').read_text()).get('window_budget', 160000)
units = {u['unit_id']: u for u in rows(run / 'units.jsonl')}
check(len(units) == sum(1 for _ in rows(run / 'units.jsonl')), 'duplicate unit IDs')
for u in units.values():
    check(hashlib.sha256(u['text'].encode()).hexdigest() == u['text_sha256'], 'text hash mismatch: ' + u['unit_id'])
    check(u['prompt_tokens'] <= budget, 'window over budget: ' + u['unit_id'])
    if 'evidence_chunks' in u:
        validate_chunk_map(u)
for event in rows(run / 'coverage.jsonl'):
    for sid, length in event['segment_lengths'].items():
        ranges = sorted(units[uid]['source_offsets'][sid] for uid in event['units'] if sid in units[uid]['source_offsets'])
        cursor = 0
        for start, end in ranges:
            check(0 <= start <= cursor <= length and start <= end <= length, f'bad or missing range: {sid} {ranges}')
            cursor = max(cursor, end)
        check(cursor == length, f'uncovered characters: {sid} ({cursor} of {length})')
# Audit the assignment the launcher runs, not prepare's pilot/shard-N files.
parts = sorted((run / 'full-shards').glob('part-*.jsonl'))
check(parts, 'no full-shards/part-*.jsonl; run scripts/shard_all.py first')
assigned = []
for part in parts:
    for u in rows(part):
        check(u == units.get(u['unit_id']), 'shard record differs from units.jsonl: ' + u['unit_id'])
        assigned.append(u['unit_id'])
check(len(assigned) == len(set(assigned)), 'a unit is assigned more than once')
check(set(assigned) == set(units), 'shards do not cover every unit')
observations = list(rows(run / 'observation-map.jsonl'))
check(len({o['observation_id'] for o in observations}) == len(observations), 'duplicate observation IDs')
unmapped = sorted(o['observation_id'] for o in observations
                  if not o['event_ids'] and not o.get('screening_event_ids') and o['disposition'] != 'index_container')
summary = {'units': len(units), 'observations': len(observations), 'all_segment_characters_covered': True,
           'shards': len(parts), 'all_units_assigned_once': True, 'input_hashes_valid': True,
           'unmapped_observations': len(unmapped), 'unmapped_observation_ids': unmapped[:100]}
(run / 'coverage-audit.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
