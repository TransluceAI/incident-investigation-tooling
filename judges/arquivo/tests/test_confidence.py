"""Confidence validation, review ordering, and historical-run compatibility."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from jsonschema import Draft202012Validator
from arquivo_judge.common import iter_jsonl, write_jsonl
from arquivo_judge.pipeline import make_units, report

SCHEMA = json.loads((Path(__file__).resolve().parents[1] / 'config/schema.json').read_text())
MISSING = object()


def judgment(score=MISSING, assessment='possible', chunk_id=1):
    hacking = {'assessment': assessment, 'categories': [], 'reason': 'Synthetic test',
               'boundary': 'unknown', 'authorization': 'unknown'}
    if score is not MISSING:
        hacking['confidence'] = score
    return {
        'hacking_attempt': hacking,
        'agentic_activity': {'assessment': 'insufficient_evidence', 'reason': 'Unknown'},
        'coordination': {'assessment': 'no_evidence', 'reason': 'None'},
        'attribution': {'claimed_actor': 'unknown', 'provider_attribution': 'not_established', 'basis': 'None'},
        'action_evidence': [],
        'evidence': [{'segment_id': 'source', 'chunk_id': chunk_id, 'location': 'Synthetic input'}],
        'missing_evidence': [],
    }


class ConfidenceTests(unittest.TestCase):
    def test_score_is_required_bounded_number(self):
        validator = Draft202012Validator(SCHEMA)
        for score in (0, 0.5, 1):
            with self.subTest(score=score):
                self.assertTrue(validator.is_valid(judgment(score)))
        for score in (MISSING, -0.01, 1.01, '0.8', None, True):
            with self.subTest(score=score):
                self.assertFalse(validator.is_valid(judgment(score)))

    def run_report(self, root, judgments, schema=SCHEMA):
        prepared = root / 'prepared'
        prepared.mkdir()
        (prepared / 'config.json').write_text(json.dumps({'schema': schema}))
        (prepared / 'control-labels.json').write_text('{}')
        units, rows = [], []
        for event_id, result in judgments:
            unit = next(make_units(event_id, [{'segment_id': 'source', 'kind': 'URL',
                                              'source': 'synthetic', 'text': 'Synthetic evidence'}], len))
            units.append(unit)
            rows.append({'unit_id': unit['unit_id'], 'text_sha256': unit['text_sha256'],
                         'status': 'success', 'result': result})
        write_jsonl(prepared / 'units.jsonl', units)
        worker = root / 'outputs/worker-0'
        worker.mkdir(parents=True)
        write_jsonl(worker / 'part-0.jsonl', rows)
        dest = root / 'report'
        with contextlib.redirect_stdout(io.StringIO()):
            report(SimpleNamespace(run=prepared, manifest=None, outputs=[worker.parent], out=dest))
        return dest

    def test_queue_sorts_scores_and_preserves_validation_status(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = self.run_report(Path(temp), [
                ('low', judgment(0.3)), ('high', judgment(0.9)),
                ('unverified', judgment(0.99, chunk_id=99)),
                ('negative', judgment(0.99, assessment='no_evidence')),
                ('zero', judgment(0)),
            ])
            queue = list(iter_jsonl(dest / 'review-queue.jsonl'))
            self.assertEqual([r['event_id'] for r in queue], ['unverified', 'high', 'low', 'zero'])
            self.assertEqual([r['hacking_confidence'] for r in queue], [0.99, 0.9, 0.3, 0])
            self.assertEqual(queue[0]['validation_status'], 'unverified_candidate')
            events = {r['event_id']: r for r in iter_jsonl(dest / 'event-results.jsonl')}
            self.assertFalse(events['unverified']['complete'])
            self.assertEqual(events['unverified']['assessment'], 'incomplete')

    def test_missing_score_in_new_run_is_failure_not_negative(self):
        with tempfile.TemporaryDirectory() as temp:
            dest = self.run_report(Path(temp), [('missing', judgment())])
            self.assertEqual(len(list(iter_jsonl(dest / 'failures.jsonl'))), 1)
            self.assertEqual(list(iter_jsonl(dest / 'review-queue.jsonl')), [])
            self.assertEqual(next(iter_jsonl(dest / 'event-results.jsonl'))['assessment'], 'incomplete')

    def test_historical_schema_keeps_unscored_results_reportable(self):
        old_schema = copy.deepcopy(SCHEMA)
        hacking = old_schema['properties']['hacking_attempt']
        hacking['required'].remove('confidence')
        del hacking['properties']['confidence']
        with tempfile.TemporaryDirectory() as temp:
            dest = self.run_report(Path(temp), [('legacy', judgment())], schema=old_schema)
            queue = list(iter_jsonl(dest / 'review-queue.jsonl'))
            self.assertEqual(len(queue), 1)
            self.assertIsNone(queue[0]['hacking_confidence'])
            self.assertEqual(queue[0]['validation_status'], 'validated')
            self.assertNotIn('confidence', queue[0]['result']['hacking_attempt'])


if __name__ == '__main__':
    unittest.main()
