"""Independent coverage and shard audit. Run after preparation, before inference."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from arquivo_judge.evidence import validate_chunk_map

run = Path(sys.argv[1])
def rows(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)

units = {u['unit_id']: u for u in rows(run / 'units.jsonl')}
assert len(units) == sum(1 for _ in rows(run / 'units.jsonl'))
for u in units.values():
    assert hashlib.sha256(u['text'].encode()).hexdigest() == u['text_sha256']
    assert u['prompt_tokens'] <= 160000
    if 'evidence_chunks' in u:
        validate_chunk_map(u)
for event in rows(run / 'coverage.jsonl'):
    for sid, length in event['segment_lengths'].items():
        ranges = sorted(units[uid]['source_offsets'][sid] for uid in event['units'] if sid in units[uid]['source_offsets'])
        cursor = 0
        for start, end in ranges:
            assert 0 <= start <= cursor <= length and start <= end <= length, (event['event_id'], sid, ranges)
            cursor = max(cursor, end)
        assert cursor == length, (sid, cursor, length)
assigned = [u['unit_id'] for u in rows(run / 'pilot.jsonl') if not u['unit_id'].startswith('control-')]
for n in range(8):
    assigned.extend(u['unit_id'] for u in rows(run / f'shard-{n}.jsonl'))
assert len(assigned) == len(set(assigned)) == len(units)
assert set(assigned) == set(units)
observations = list(rows(run / 'observation-map.jsonl'))
assert len({o['observation_id'] for o in observations}) == len(observations)
summary = {'units': len(units), 'observations': len(observations), 'all_segment_characters_covered': True,
           'disjoint_pilot_and_shards': True, 'input_hashes_valid': True,
           'unmapped_observations': sum(not o['event_ids'] and not o.get('screening_event_ids') and o['disposition'] != 'index_container' for o in observations)}
(run / 'coverage-audit.json').write_text(json.dumps(summary, indent=2) + '\n')
print(json.dumps(summary, indent=2))
