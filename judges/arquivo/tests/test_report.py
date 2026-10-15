"""Report: every axis reaches the outputs, outputs must come from this run, claims need evidence."""
import contextlib
import copy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

from arquivo_judge.common import iter_jsonl, write_jsonl
from arquivo_judge.pipeline import make_units, report, save_json
from support import write_worker

TOOL = Path(__file__).resolve().parents[1]
SCHEMA = json.loads((TOOL / 'config/schema.json').read_text())
SEGMENTS = [{'segment_id': 'url', 'kind': 'URL', 'source': 'ledger.original_url', 'text': 'https://example.org/a?q=1'},
            {'segment_id': 'body', 'kind': 'BODY', 'source': 'document:a.txt', 'text': 'Response text'}]


def judgment(hacking='no_evidence', agentic='no_evidence', coordination='no_evidence',
             attribution='not_established', basis='', cite=('url',), actions=(), confidence=0.1):
    return {
        'hacking_attempt': {'assessment': hacking, 'confidence': confidence, 'categories': [],
                            'reason': 'Synthetic test', 'boundary': 'unknown', 'authorization': 'unknown'},
        'agentic_activity': {'assessment': agentic, 'reason': 'Synthetic'},
        'coordination': {'assessment': coordination, 'reason': 'Synthetic'},
        'attribution': {'claimed_actor': 'unknown', 'provider_attribution': attribution, 'basis': basis},
        'action_evidence': [{'target': 'example.org', 'action': 'request', 'observed_stage': 'recorded_request',
                             'responding_system': 'unknown', 'outcome': 'unknown', 'evidence_ids': list(ids)}
                            for ids in actions],
        'evidence': [{'segment_id': sid, 'chunk_id': 1, 'location': 'Synthetic'} for sid in cite],
        'missing_evidence': [],
    }


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def prepare(self, results):
        prepared = self.root / 'prepared'
        prepared.mkdir()
        cfg = {'schema': SCHEMA, 'system_prompt': 'rubric'}
        save_json(prepared / 'config.json', cfg)
        save_json(prepared / 'control-labels.json', {})
        write_jsonl(prepared / 'gaps.jsonl', [{'observation_id': 'x', 'gap': 'unknown_kind'}])
        units, rows = [], []
        for event_id, result in results:
            segs = [{**s, 'segment_id': f'{event_id}:{s["segment_id"]}'} for s in SEGMENTS]
            unit = next(make_units(event_id, segs, len))
            units.append(unit)
            rows.append({'unit_id': unit['unit_id'], 'text_sha256': unit['text_sha256'],
                         'status': 'success', 'result': result and prefix(result, event_id)})
        write_jsonl(prepared / 'units.jsonl', units)
        return prepared, cfg, rows

    def report(self, prepared):
        dest = self.root / 'report'
        with contextlib.redirect_stdout(io.StringIO()):
            report(SimpleNamespace(run=prepared, manifest=None, outputs=[self.root / 'outputs'], out=dest))
        out = {name: list(iter_jsonl(dest / f'{name}.jsonl'))
               for name in ('event-results', 'review-queue', 'failures', 'unverified-candidates', 'predictions')}
        out['summary'] = json.loads((dest / 'summary.json').read_text())
        return out

    def run_report(self, results):
        prepared, cfg, rows = self.prepare(results)
        write_worker(self.root / 'outputs', cfg, rows)
        return self.report(prepared)

    def test_secondary_axes_reach_events_queue_and_summary(self):
        out = self.run_report([('agent', judgment(agentic='likely')),
                               ('coord', judgment(coordination='possible', cite=('nonexistent',))),
                               ('quiet', judgment())])
        events = {e['event_id']: e for e in out['event-results']}
        self.assertEqual(events['agent']['assessment'], 'no_evidence')
        self.assertEqual(events['agent']['assessments']['agentic_activity'], 'likely')
        self.assertEqual(events['agent']['positive_axes'], ['agentic_activity'])
        self.assertEqual(events['quiet']['positive_axes'], [])
        queue = {r['event_id']: r for r in out['review-queue']}
        self.assertEqual(queue['agent']['positive_axes'], ['agentic_activity'])
        self.assertEqual(queue['coord']['validation_status'], 'unverified_candidate')
        self.assertNotIn('quiet', queue)
        self.assertEqual([r['event_id'] for r in out['unverified-candidates']], ['coord'])
        by_axis = out['summary']['event_assessments_by_axis']
        self.assertEqual(by_axis['agentic_activity'], {'likely': 1, 'no_evidence': 1, 'incomplete': 1})
        self.assertEqual(out['summary']['preparation_gaps'], {'unknown_kind': 1})

    def test_supported_attribution_needs_citation_and_basis(self):
        out = self.run_report([
            ('uncited', judgment(attribution='supported_by_supplied_evidence', basis='Signed note', cite=())),
            ('no-basis', judgment(attribution='supported_by_supplied_evidence', basis='  ')),
            ('ok', judgment(attribution='supported_by_supplied_evidence', basis='Signed note')),
        ])
        errors = {f['event_id']: f['validation_error'] for f in out['failures']}
        self.assertIn('without evidence', errors['uncited'])
        self.assertIn('without a basis', errors['no-basis'])
        events = {e['event_id']: e for e in out['event-results']}
        self.assertEqual(events['ok']['assessments']['provider_attribution'], 'supported_by_supplied_evidence')
        self.assertEqual(events['ok']['positive_axes'], ['provider_attribution'])

    def test_action_support_marks_uncited_segments(self):
        out = self.run_report([('mixed', judgment(hacking='possible', actions=[('url',), ('url', 'body')]))])
        refs = out['predictions'][0]['action_evidence_sources']
        self.assertEqual([s['support'] for s in refs[0]['segments']], ['cited'])
        self.assertEqual(refs[0]['segments'][0]['cited_chunk_ids'], [1])
        self.assertEqual([s['support'] for s in refs[1]['segments']], ['cited', 'segment_reference_only'])
        self.assertEqual(out['summary']['windows_with_uncited_action_segments'], 1)

    def test_outputs_from_another_config_are_rejected(self):
        prepared, cfg, rows = self.prepare([('a', judgment())])
        write_worker(self.root / 'outputs', {**cfg, 'system_prompt': 'another rubric'}, rows)
        with self.assertRaisesRegex(ValueError, 'different prepared config'):
            self.report(prepared)

    def test_rows_from_another_run_are_rejected(self):
        prepared, cfg, rows = self.prepare([('a', judgment())])
        write_worker(self.root / 'outputs', cfg, [{**rows[0], 'run_identity': 'f' * 64}])
        with self.assertRaisesRegex(ValueError, 'another run'):
            self.report(prepared)

    def test_altered_or_missing_identity_is_rejected(self):
        prepared, cfg, rows = self.prepare([('a', judgment())])
        worker = write_worker(self.root / 'outputs', cfg, rows)
        saved = json.loads((worker / 'identity.json').read_text())
        save_json(worker / 'identity.json', {**saved, 'manifest_hash': 'e' * 64})
        with self.assertRaisesRegex(ValueError, 'does not match its recorded inputs'):
            self.report(prepared)
        (worker / 'identity.json').unlink()
        with self.assertRaisesRegex(ValueError, 'Missing runner identity'):
            self.report(prepared)

    def test_unreadable_capture_keeps_event_incomplete(self):
        prepared, cfg, rows = self.prepare([('a', judgment()), ('b', judgment())])
        write_jsonl(prepared / 'events.jsonl', [
            {'event_id': 'a', 'body_gaps': [{'observation_id': 'cap', 'gap': 'document_unreadable'}]},
            {'event_id': 'b', 'body_gaps': []}])
        write_worker(self.root / 'outputs', cfg, rows)
        events = {e['event_id']: e for e in self.report(prepared)['event-results']}
        self.assertEqual((events['a']['complete'], events['a']['assessment'], events['a']['unreadable_captures']),
                         (False, 'incomplete', 1))
        self.assertEqual((events['b']['complete'], events['b']['assessment']), (True, 'no_evidence'))

    def test_identity_written_by_inference_runner_is_accepted(self):
        prepared, cfg, rows = self.prepare([('a', judgment())])
        # A non-ASCII config value checks encoding on both sides.
        cfg = {**cfg, 'model': 'local-model', 'note': 'non-ASCII \u2014 text'}
        save_json(prepared / 'config.json', cfg)
        fake = self.root / 'fake'
        fake.mkdir()
        (fake / 'transformers.py').write_text(
            'class AutoTokenizer:\n    @staticmethod\n    def from_pretrained(*a, **k):\n        return None\n')
        (self.root / 'empty.jsonl').write_text('')
        # With nothing left to run, the real runner records its identity without loading a model.
        result = subprocess.run(
            [sys.executable, str(TOOL / 'scripts/inference_runner.py'), '--config', str(prepared / 'config.json'),
             '--input-manifest', str(self.root / 'empty.jsonl'), '--output-dir', str(self.root / 'outputs'),
             '--worker-id', '0'], capture_output=True, text=True, env={'PYTHONPATH': str(fake)})
        self.assertEqual(result.returncode, 0, result.stderr)
        worker = self.root / 'outputs/worker-0'
        identity = json.loads((worker / 'identity.json').read_text(encoding='utf-8'))['identity']
        write_jsonl(worker / 'part-000000.jsonl', [{'run_identity': identity, **rows[0]}])
        self.assertEqual(len(self.report(prepared)['predictions']), 1)


def prefix(result, event_id):
    """Point synthetic citations at this event's segment IDs."""
    result = copy.deepcopy(result)
    for quote in result['evidence']:
        quote['segment_id'] = f'{event_id}:{quote["segment_id"]}'
    for action in result['action_evidence']:
        action['evidence_ids'] = [f'{event_id}:{sid}' for sid in action['evidence_ids']]
    return result


if __name__ == '__main__':
    unittest.main()
