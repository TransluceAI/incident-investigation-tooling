"""Preparation: malformed inputs as gaps, window size from the engine settings, controls."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest

from arquivo_judge.common import iter_jsonl
from arquivo_judge.pipeline import controls, prepare
from support import install_fake_transformers

TOOL = Path(__file__).resolve().parents[1]


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class ControlTests(unittest.TestCase):
    def test_missing_body_controls_render_without_a_body(self):
        units, labels = controls(len)
        for name in ('sql', 'traversal', 'config'):
            text = next(u['text'] for u in units if u['unit_id'] == f'control-{name}:w0')
            self.assertIn('body: missing', text)
            self.assertIn('BODY: missing (no exported capture', text)


class PrepareTests(unittest.TestCase):
    def setUp(self):
        install_fake_transformers()
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.corpus = self.root / 'corpus'
        shutil.copytree(TOOL / 'examples/corpus', self.corpus)
        self.engine = json.loads((TOOL / 'config/engine.example.json').read_text())

    def tearDown(self):
        self.temp.cleanup()

    def add_records(self, *lines):
        with (self.corpus / 'resources/arquivo.pt/records.jsonl').open('a') as out:
            for line in lines:
                out.write((line if isinstance(line, str) else json.dumps(line)) + '\n')

    def prepare(self, engine=None):
        (self.root / 'engine.json').write_text(json.dumps(engine or self.engine))
        out = self.root / 'prepared'
        with contextlib.redirect_stdout(io.StringIO()):
            prepare(SimpleNamespace(data_root=self.corpus, engine=self.root / 'engine.json',
                                    rubric=TOOL / 'config/rubric.txt', schema=TOOL / 'config/schema.json', out=out))
        return out

    def test_malformed_observations_become_gaps(self):
        (self.corpus / 'documents/ok.txt').write_text('fine')
        self.add_records(
            'not json',
            {'kind': 'arquivo_capture'},
            {'observation_id': 'no-hash', 'kind': 'arquivo_capture', 'document_path': 'documents/ok.txt',
             'metadata': {'original_url': 'https://a.example/', 'capture_timestamp': '20260101000000'}},
            {'observation_id': 'gone', 'kind': 'arquivo_capture', 'document_path': 'documents/missing.txt',
             'text_sha256': sha('x'), 'metadata': {'original_url': 'https://a.example/', 'capture_timestamp': '20260101000000'}},
            {'observation_id': 'no-doc', 'kind': 'arquivo_capture',
             'metadata': {'original_url': 'https://a.example/', 'capture_timestamp': '20260101000000'}},
            {'observation_id': 'null-original', 'kind': 'arquivo_capture', 'original_record': None,
             'document_path': 'documents/ok.txt', 'text_sha256': sha('fine')},
            {'observation_id': 'empty-index', 'kind': 'arquivo_index'},
            {'observation_id': 'empty-redirect', 'kind': 'provider_redirect'},
            {'observation_id': 'bad-types', 'kind': ['arquivo_capture'], 'source_url': 5, 'original_record': 'x'},
        )
        out = self.prepare()
        gaps = {(g['observation_id'], g['gap']) for g in iter_jsonl(out / 'gaps.jsonl')}
        self.assertTrue({(None, 'record_parse_error'), (None, 'missing_observation_id'),
                         ('no-hash', 'missing_text_sha256'), ('gone', 'document_unreadable'),
                         ('no-doc', 'capture_missing_document'), ('null-original', 'capture_missing_metadata'),
                         ('empty-index', 'index_missing_document'), ('empty-redirect', 'redirect_empty'),
                         ('bad-types', 'invalid_field_type')} <= gaps)
        mapped = {o['observation_id']: o for o in iter_jsonl(out / 'observation-map.jsonl')}
        self.assertEqual(mapped['bad-types']['disposition'], 'parse_error')
        # Captures that exist but cannot be read keep their event, marked unreadable.
        for oid in ('no-hash', 'gone', 'no-doc'):
            self.assertEqual(mapped[oid]['disposition'], 'unreadable_capture')
        event = next(e for e in iter_jsonl(out / 'events.jsonl') if e['original_url'] == 'https://a.example/')
        self.assertEqual({g['observation_id'] for g in event['body_gaps']}, {'no-hash', 'gone', 'no-doc'})
        unit = next(u for u in iter_jsonl(out / 'units.jsonl') if u['event_id'] == event['event_id'])
        self.assertIn('body: unreadable', unit['text'])
        self.assertIn('BODY: unreadable', unit['text'])
        self.assertEqual(json.loads((out / 'preparation.json').read_text())['units'], 3)

    def test_hash_mismatch_still_stops_preparation(self):
        (self.corpus / 'documents/population.csv').write_text('altered')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            self.prepare()

    def test_empty_corpus_prepares_nothing(self):
        (self.corpus / 'resources/arquivo.pt/records.jsonl').write_text('')
        summary = json.loads((self.prepare() / 'preparation.json').read_text())
        self.assertEqual((summary['units'], summary['max_prompt_tokens']), (0, 0))

    def test_window_size_follows_model_context(self):
        engine = {**self.engine, 'engine': {**self.engine['engine'], 'max_model_len': 20000}}
        (self.corpus / 'documents/population.csv').write_text('year,population\n' * 3000)
        records = list(iter_jsonl(self.corpus / 'resources/arquivo.pt/records.jsonl'))
        records[1]['text_sha256'] = sha('year,population\n' * 3000)
        (self.corpus / 'resources/arquivo.pt/records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
        out = self.prepare(engine)
        cfg = json.loads((out / 'config.json').read_text())
        self.assertEqual(cfg['window_budget'], 20000 - self.engine['sampling']['max_tokens'])
        units = list(iter_jsonl(out / 'units.jsonl'))
        self.assertGreater(len(units), 2)
        self.assertTrue(all(u['prompt_tokens'] <= cfg['window_budget'] for u in units))


if __name__ == '__main__':
    unittest.main()
