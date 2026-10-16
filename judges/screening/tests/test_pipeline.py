import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

from screening_judge import controls, records, scanner
from screening_judge.common import request_sha256, sha256_text, write_jsonl, iter_jsonl
from screening_judge.prepare import prepare
from screening_judge.prompts import load_prompts, schemas
from screening_judge.render import render_citable, windows, validate_chunk_map
from screening_judge.report import reconcile, review_queue, write_report
from screening_judge.validate import validate, ValidationError

ROOT = Path(__file__).resolve().parents[1]
# A resolved inference/qwen runner profile (the judge no longer ships engine configs).
ENGINE = {
    'model': '/path/to/local/Qwen3.8-27B-FP8',
    'model_revision': '017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',
    'engine': {'tensor_parallel_size': 1, 'max_model_len': 262144, 'max_num_seqs': 16,
               'enable_prefix_caching': True, 'seed': 17},
    'sampling': {'temperature': 0, 'max_tokens': 2048},
    'chat_template_kwargs': {'enable_thinking': False},
    'host_batch_records': 32, 'host_batch_tokens': 400000,
}


def config():
    return copy.deepcopy(ENGINE)


def answered(run, uid):
    """The runner's request fields for one prepared request."""
    body = next(r['body'] for r in iter_jsonl(run / 'requests.jsonl') if r['custom_id'] == uid)
    return {'unit_id': uid, 'custom_id': uid, 'request_sha256': request_sha256(body)}


def result(axis, label='no'):
    r = {'evidence': [], 'policy_basis': 'test', 'rationale': 'Test fixture.',
         'label': label, 'confidence': 0.9 if label == 'yes' else 0.1,
         'injection_detected': False, 'missing_evidence': []}
    if axis == 'hacking':
        r.update(categories=[], boundary='', authorization='', actions=[])
    elif axis == 'agentic_activity':
        r.update(claimed_actor='', provider_attribution='not_established')
    else:
        r.update(mechanism='none')
    return r


def write_worker(run, rows, cfg=None):
    from screening_judge.common import sha256_file
    cfg = cfg or json.loads((run / 'engine.json').read_text())
    ident = {'config': cfg, 'manifest_hash': sha256_file(run / 'requests.jsonl'),
             'runner_hash': 'test-runner', 'max_records': None}
    digest = sha256_text(json.dumps(ident, sort_keys=True))
    worker = run / 'outputs/worker-0'
    worker.mkdir(parents=True, exist_ok=True)
    (worker / 'identity.json').write_text(json.dumps({'identity': digest, **ident}))
    write_jsonl(worker / 'part-000000.jsonl', [{**r, 'run_identity': digest} for r in rows])
    return worker


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root / 'data'
        (self.data / 'resources/sample').mkdir(parents=True)

    def prepare(self, rows):
        write_jsonl(self.data / 'resources/sample/records.jsonl', rows)
        self.run = self.root / 'run'
        return prepare(self.data, self.run, lambda chat: sum(len(m['content']) for m in chat),
                       15000, config(), with_controls=False)

    def record(self, oid='one', body='ordinary text', **kw):
        p = self.data / f'{oid}.txt'
        p.write_text(body)
        return {'observation_id': oid, 'kind': 'raw_paste', 'document_path': p.name,
                'text_sha256': sha256_text(body), **kw}

    def test_unreadable_evidence_remains_incomplete_in_report(self):
        corrupt = self.record('corrupt', text_sha256='bad')
        missing = {'observation_id': 'missing', 'kind': 'raw_paste',
                   'document_path': 'missing.txt', 'text_sha256': 'a'*64}
        self.prepare([corrupt, missing])
        parts = reconcile(self.run, [])
        self.assertEqual(len(parts[1]), 2)
        for item in parts[1]:
            for axis in item['axes'].values():
                self.assertEqual(axis['assessment'], 'incomplete')
                self.assertFalse(axis['complete'])
        self.assertEqual({g['gap'] for g in iter_jsonl(self.run/'gaps.jsonl')},
                         {'document_hash_mismatch', 'document_unreadable'})

    def test_control_harness_identity_stays_out_of_model_evidence(self):
        write_jsonl(self.data / 'resources/sample/records.jsonl', [self.record()])
        run = self.root / 'controls'
        prepare(self.data, run, lambda chat: sum(len(m['content']) for m in chat) // 4,
                15000, config(), with_controls=True)
        items = list(iter_jsonl(run / 'items.jsonl'))
        self.assertEqual(sum(i['source'] == controls.SOURCE for i in items), len(controls.CONTROLS))
        requests = [r for r in iter_jsonl(run / 'requests.jsonl') if r['custom_id'].startswith('control-')]
        self.assertEqual(len(requests), 3 * len(controls.CONTROLS))
        for request in requests:
            content = '\n'.join(m['content'] for m in request['body']['messages'])
            for marker in ('synthetic-control', 'synthetic_', 'observation_id=synthetic', 'time_basis: synthetic'):
                self.assertNotIn(marker, content)
            self.assertNotIn(request['custom_id'].split(':')[0], content)
        sql = next(r for r in requests if r['custom_id'] == 'control-sql-url:w0:hacking')
        self.assertIn('UNION', sql['body']['messages'][-1]['content'])
        item = {'source': 'real-source'}
        self.assertIs(controls.prompt_item(item), item)

    def test_all_axes_counted_for_window_budget(self):
        rec = self.record(body='x'*9000)
        write_jsonl(self.data / 'resources/sample/records.jsonl', [rec])
        coordination = load_prompts()['formats']['coordination']
        def count(chat):
            return len(chat[-1]['content']) + (3500 if chat[-1]['content'].endswith(coordination) else 0)
        run = self.root / 'run'
        prepare(self.data, run, count, 14000, config(), with_controls=False)
        rows = list(iter_jsonl(run/'requests.jsonl'))
        self.assertGreater(len(rows), 3)  # the body needs more than one window
        self.assertTrue(any(r['custom_id'].endswith(':coordination') for r in rows))
        for row in rows:
            self.assertLessEqual(count(row['body']['messages']), 14000)

    def test_budget_is_measured_once_with_the_heaviest_axis(self):
        rec = self.record(body='x'*9000)
        write_jsonl(self.data / 'resources/sample/records.jsonl', [rec])
        formats = load_prompts()['formats']
        calls = []
        def count(chat):
            calls.append(next(a for a, f in formats.items() if chat[-1]['content'].endswith(f)))
            return len(chat[-1]['content'])
        prepare(self.data, self.root / 'run', count, 14000, config(), with_controls=False)
        windows_written = sum(1 for _ in iter_jsonl(self.root/'run/windows.jsonl'))
        self.assertGreater(windows_written, 1)
        # One probe per axis finds the heaviest prompt; every window measurement then uses that axis only.
        heaviest = max(formats, key=lambda a: len(formats[a]) + len(load_prompts()['axes'][a]))
        for axis in formats:
            if axis != heaviest:
                self.assertEqual(calls.count(axis), 1, axis)
        self.assertGreaterEqual(calls.count(heaviest), windows_written + 1)

    def test_boundary_tag_is_a_per_run_nonce(self):
        from screening_judge.prompts import messages
        cfg = {'prompts': load_prompts(), 'boundary_salt': 'a' * 32}
        tags = {messages(cfg, 'text', 'hacking', text)[-1]['content'].split('<')[1].split('>')[0]
                for text in ('first window', 'second window')}
        self.assertEqual(len(tags), 1)
        tag, = tags
        self.assertIn('a' * 32, tag)
        other = messages({**cfg, 'boundary_salt': 'b' * 32}, 'text', 'hacking', 'first window')[-1]['content']
        self.assertNotIn(tag, other)

    def test_requests_are_self_contained_runner_rows(self):
        self.prepare([self.record()])
        cfg = json.loads((self.run/'config.json').read_text())
        w = next(iter_jsonl(self.run/'windows.jsonl'))
        rows = list(iter_jsonl(self.run/'requests.jsonl'))
        self.assertEqual([r['custom_id'] for r in rows], [f"{w['window_id']}:{a}" for a in schemas()])
        from screening_judge.prompts import messages
        for row, axis in zip(rows, schemas()):
            self.assertEqual(set(row), {'custom_id', 'body'})
            body = row['body']
            self.assertEqual(body['messages'], messages(cfg, w['shape'], axis, w['text']))
            from screening_judge.schemas import citation_schema
            self.assertEqual(body['response_format'], {'type': 'json_schema', 'json_schema': {
                'name': axis, 'schema': citation_schema(axis, w['evidence_chunks'])}})
            self.assertEqual((body['logprobs'], body['top_logprobs']), (True, 10))
        # The runner receives the profile only: no prompts, schemas or judge keys.
        self.assertEqual(json.loads((self.run/'engine.json').read_text()), ENGINE)

    def test_label_logprobs_option_and_old_profile_key(self):
        write_jsonl(self.data/'resources/sample/records.jsonl', [self.record()])
        count = lambda chat: sum(len(m['content']) for m in chat)
        prepare(self.data, self.root/'off', count, 15000, config(), with_controls=False, label_logprobs=0)
        self.assertFalse(any('logprobs' in r['body'] for r in iter_jsonl(self.root/'off/requests.jsonl')))
        with self.assertRaisesRegex(ValueError, 'label-logprobs'):
            prepare(self.data, self.root/'old', count, 15000, {**config(), 'label_logprobs': 10}, with_controls=False)

    def test_citation_constraints_narrow_each_request_schema(self):
        from jsonschema import Draft202012Validator
        write_jsonl(self.data/'resources/sample/records.jsonl', [self.record(body='x'*600)])
        self.run = self.root/'run'
        count = lambda chat: sum(len(m['content']) for m in chat)
        prepare(self.data, self.run, count, 15000, config(), with_controls=False)  # narrowing is the default
        with self.assertRaisesRegex(ValueError, 'judge option'):
            prepare(self.data, self.root/'old', count, 15000, {**config(), 'citation_constraints': True},
                    with_controls=False)
        prepare(self.data, self.root/'plain', count, 15000, config(), with_controls=False, citation_constraints=False)
        plain = next(r for r in iter_jsonl(self.root/'plain/requests.jsonl') if r['custom_id'].endswith(':hacking'))
        self.assertEqual(plain['body']['response_format']['json_schema']['schema'], schemas()['hacking'])
        w = next(iter_jsonl(self.run/'windows.jsonl'))
        sid = next(k for k, v in w['segment_kinds'].items() if v == 'BODY')
        n = len(w['evidence_chunks'][sid])
        row = next(r for r in iter_jsonl(self.run/'requests.jsonl') if r['custom_id'].endswith(':hacking'))
        narrowed = Draft202012Validator(row['body']['response_format']['json_schema']['schema'])
        base = Draft202012Validator(schemas()['hacking'])
        res = result('hacking', 'yes')
        action = {'target': 't', 'action': 'a', 'observed_stage': 'recorded_request',
                  'responding_system': 'unknown', 'outcome': 'unknown', 'evidence_segment_ids': [sid]}
        for evidence, actions, ok in (([{'segment_id': sid, 'chunk_id': n, 'location': 'l'}], [action], True),
                                      ([{'segment_id': sid, 'chunk_id': n + 1, 'location': 'l'}], [], False),
                                      ([{'segment_id': 'S99', 'chunk_id': 1, 'location': 'l'}], [], False),
                                      ([], [{**action, 'evidence_segment_ids': []}], False),
                                      ([], [{**action, 'evidence_segment_ids': ['S99']}], False)):
            res.update(evidence=evidence, actions=actions)
            with self.subTest(evidence=evidence, actions=actions):
                self.assertEqual(narrowed.is_valid(res), ok)
                if ok:
                    self.assertTrue(base.is_valid(res))  # narrowing never widens the saved schema

    def test_report_rejects_output_for_a_different_request(self):
        self.prepare([self.record()])
        w = next(iter_jsonl(self.run/'windows.jsonl'))
        uid = w['window_id'] + ':hacking'
        write_worker(self.run, [{**answered(self.run, uid), 'request_sha256': 'f'*64,
                                 'status': 'success', 'result': result('hacking')}])
        parts = reconcile(self.run, [self.run/'outputs'])
        self.assertIn('prepared request', parts[3][0]['validation_error'])

    def test_worker_config_mismatch_rejected(self):
        self.prepare([self.record()])
        cfg = json.loads((self.run / 'engine.json').read_text())
        cfg['sampling']['temperature'] = 0.5
        write_worker(self.run, [], cfg)
        with self.assertRaisesRegex(ValueError, 'config|identity'):
            reconcile(self.run, [self.run/'outputs'])

    def test_capture_same_body_different_request_not_merged(self):
        a = self.record('a', '404', kind='wayback_capture', metadata={'original_url':'https://a.example/x', 'capture_timestamp':'1'})
        b = self.record('b', '404', kind='wayback_capture', metadata={'original_url':'https://a.example/y', 'capture_timestamp':'1'})
        self.assertEqual(self.prepare([a,b])['items'], 2)

    def test_same_text_different_shape_not_merged(self):
        a, b = self.record('a', 'same'), self.record('b', 'same', kind='retained_message')
        self.assertEqual(self.prepare([a,b])['items'], 2)

    def test_discovery_recursive_direct_without_raw_duplicate_records(self):
        deep = self.data/'resources/sample/a/b/c/DIRECT_RECORDS.jsonl'
        deep.parent.mkdir(parents=True); deep.write_text('')
        raw = self.data/'resources/sample/archive-scanner-sources-1/records/raw.json'
        raw.parent.mkdir(parents=True); raw.write_text('{}')
        self.assertEqual([p for _,p in records.sources(self.data)], [deep])

    def test_direct_records_use_id_when_observation_id_is_absent(self):
        rec = self.record('page-1', 'wiki page text')
        direct = {**{k: v for k, v in rec.items() if k != 'observation_id'}, 'id': 'page-1', 'object_id': 'obj-9'}
        deep = self.data/'resources/wiki.example/export/DIRECT_RECORDS.jsonl'
        deep.parent.mkdir(parents=True)
        write_jsonl(deep, [direct, {k: v for k, v in direct.items() if k != 'id'}, direct])
        summary = self.prepare([self.record('other', 'unrelated')])
        observations = [o['observation_id'] for i in iter_jsonl(self.run/'items.jsonl') for o in i['observations']]
        self.assertIn('page-1', observations)
        gaps = [g['gap'] for g in iter_jsonl(self.run/'gaps.jsonl')]
        self.assertEqual(sorted(gaps), ['duplicate_observation', 'unreadable_record'])  # no id; repeated id
        self.assertEqual(summary['items'], 2)

    def test_archive_ledger_kinds_from_other_sources_are_ordinary_items(self):
        rec = self.record('redirect-1', 'Moved', kind='provider_redirect', source_url='https://short.example/x')
        write_jsonl(self.data/'resources/urlquery.net/records.jsonl', [rec])
        self.prepare([self.record('other', 'unrelated')])
        items = {i['source']: i for i in iter_jsonl(self.run/'items.jsonl')}
        self.assertEqual([o['observation_id'] for o in items['urlquery.net']['observations']], ['redirect-1'])
        self.assertFalse(any(g.get('source') == 'urlquery.net' for g in iter_jsonl(self.run/'gaps.jsonl')))

    def test_source_adapters_receive_only_their_own_kinds(self):
        from unittest.mock import patch
        from screening_judge import prepare as prep
        received = []

        def join(source, recs, corpus, items, gaps):
            received.extend((source, r['observation_id']) for r in recs)
            item = prep._new_item('joined-' + source, source, 'text')
            for r in recs:
                prep._observe(item, r)
            items[item['item_id']] = item

        write_jsonl(self.data/'resources/other/records.jsonl', [self.record('elsewhere', 'x', kind='event_part')])
        adapters = {**prep.ADAPTERS, 'sample': prep.Adapter(frozenset({'event_part'}), join)}
        with patch.object(prep, 'ADAPTERS', adapters):
            self.prepare([self.record('a', 'part one', kind='event_part'), self.record('b', 'part two', kind='event_part'),
                          self.record('c', 'standalone')])
        self.assertEqual(received, [('sample', 'a'), ('sample', 'b')])  # same kind in 'other' stays generic
        items = list(iter_jsonl(self.run/'items.jsonl'))
        joined = next(i for i in items if i['item_id'] == 'joined-sample')
        self.assertEqual([o['observation_id'] for o in joined['observations']], ['a', 'b'])
        self.assertEqual({i['source'] for i in items if i['item_id'] != 'joined-sample'}, {'sample', 'other'})

    def test_empty_corpus_rejected(self):
        with self.assertRaisesRegex(ValueError, 'No .*records|empty'):
            self.prepare([])


class EvidenceTests(unittest.TestCase):
    def unit(self, body='café 日本語 '*600):
        s = [{'segment_id':'S0','kind':'BODY','source':'document:test','text':body}]
        text, spans, chunks = render_citable(s)
        return {'window_id':'w', 'chunk_idx':0, 'text':text,'text_sha256':sha256_text(text),
                'segment_spans':spans,'evidence_chunks':chunks,'source_offsets':{'S0':[0,len(s[0]['text'])]},
                'segment_sources':{'S0':'document:test'},'segment_kinds':{'S0':'BODY'}}

    def test_chunk_coverage_exact_across_windows(self):
        body = 'café 日本語 '*600
        s = [{'segment_id':'S0','kind':'BODY','source':'document:test','text':body}]
        covered = set()
        for parts, offsets in windows(s, len, 1100):
            text, spans, chunks = render_citable(parts, offsets)
            unit={'text':text,'segment_spans':spans,'evidence_chunks':chunks,'source_offsets':offsets}
            validate_chunk_map(unit)
            for c in chunks['S0']:
                self.assertEqual(text[c['text_start']:c['text_end']],body[c['source_start']:c['source_end']])
                covered.update(range(c['source_start'],c['source_end']))
        self.assertEqual(covered,set(range(len(body))))

    def test_windows_fill_budget_and_preserve_unicode_source(self):
        body = 'ordinary browser response café 日本語 😀 ' * 2000
        segments = [{'segment_id': 'S0', 'kind': 'PROVENANCE', 'source': 'records', 'text': 'observed request'},
                    {'segment_id': 'S1', 'kind': 'BODY_VIEW', 'source': 'document:x', 'text': body}]
        for count in (lambda text: 800 + len(text)//4, lambda text: 800 + len(text.encode('utf-8'))):
            with self.subTest(count=count):
                end = 0
                for parts, offsets in windows(segments, count, 6000):
                    text, spans, chunks = render_citable(parts, offsets)
                    validate_chunk_map({'text': text, 'segment_spans': spans,
                                        'evidence_chunks': chunks, 'source_offsets': offsets})
                    start, stop = offsets['S1']
                    self.assertLessEqual(start, end)
                    self.assertGreater(stop, end)
                    self.assertEqual(next(p['text'] for p in parts if p['segment_id'] == 'S1'), body[start:stop])
                    self.assertLessEqual(count(text), 6000)
                    if stop < len(body):
                        self.assertGreaterEqual(count(text), 5700)
                    end = stop
                self.assertEqual(end, len(body))

    def test_every_window_carries_provenance(self):
        def segments(provenance):
            return [{'segment_id': 'S0', 'kind': 'ITEM', 'source': 'item', 'text': 'source: paste.example'},
                    {'segment_id': 'S1', 'kind': 'PROVENANCE', 'source': 'records', 'text': provenance},
                    {'segment_id': 'S2', 'kind': 'BODY', 'source': 'document:x', 'text': 'body text ' * 4000}]
        count = lambda text: len(text)
        short = 'observed by archive crawler at 2026-06-01'
        parts = list(windows(segments(short), count, 6000))
        self.assertGreater(len(parts), 5)
        for window, _ in parts:
            self.assertIn(short, {s['text'] for s in window if s['kind'] == 'PROVENANCE'})
        long = '\n'.join(f'observation {i}: collected via export {i}' for i in range(400))
        parts = list(windows(segments(long), count, 6000))
        body = [w for w, _ in parts if any(s['kind'] == 'BODY' for s in w)]
        self.assertGreater(len(body), 5)
        for window in body:
            header, = [s for s in window if s['kind'] == 'PROVENANCE_HEADER']
            self.assertTrue(header['text'].startswith('observation 0: collected via export 0'))
            self.assertIn('further provenance lines are in the windows of segment S1', header['text'])
        # The full provenance is still windowed and covered: nothing is omitted.
        covered = set()
        for window, offsets in parts:
            if 'S1' in offsets and any(s['segment_id'] == 'S1' for s in window):
                covered.update(range(*offsets['S1']))
        self.assertEqual(covered, set(range(len(long))))

    def test_context_segments_are_counted_once_each(self):
        segs = [{'segment_id': 'S0', 'kind': 'ITEM', 'source': 'item', 'text': 'source: paste.example'}]
        segs += [{'segment_id': f'S{i}', 'kind': 'URL', 'source': 'request_url', 'text': f'https://a.example/{i}/' + 'q' * 90}
                 for i in range(1, 41)]
        segs.append({'segment_id': 'S41', 'kind': 'BODY', 'source': 'document:x', 'text': 'body text ' * 2000})
        counted = []
        def count(text):
            counted.append(text)
            return len(text)
        list(windows(segs, count, 24000))  # every context segment fits beside the body
        context_only = sum(len(t) for t in counted if '[SEGMENT' in t and 'BODY' not in t)
        self.assertLess(context_only, 30000)  # linear in the context size, not quadratic

    def test_provenance_header_is_found_by_bisection(self):
        long = '\n'.join(f'observation {i}: collected via export {i}' for i in range(400))
        segs = [{'segment_id': 'S0', 'kind': 'ITEM', 'source': 'item', 'text': 'source: paste.example'},
                {'segment_id': 'S1', 'kind': 'PROVENANCE', 'source': 'records', 'text': long},
                {'segment_id': 'S2', 'kind': 'BODY', 'source': 'document:x', 'text': 'body text ' * 4000}]
        counted = []
        def count(text):
            counted.append(text)
            return len(text)
        parts = list(windows(segs, count, 6000))
        self.assertGreater(len(parts), 5)
        probes = [t for t in counted if 'PROVENANCE_HEADER' in t and 'BODY' not in t and '[SEGMENT S1 PROVENANCE]' not in t]
        self.assertLess(len(probes), 20)  # bisection over 400 lines, not a linear scan

    def test_single_long_provenance_line_still_heads_every_window(self):
        segs = [{'segment_id': 'S0', 'kind': 'ITEM', 'source': 'item', 'text': 'source: paste.example'},
                {'segment_id': 'S1', 'kind': 'PROVENANCE', 'source': 'records', 'text': 'line-0 ' + 'p' * 9000},
                {'segment_id': 'S2', 'kind': 'BODY', 'source': 'document:x', 'text': 'body text ' * 4000}]
        body = [w for w, _ in windows(segs, len, 6000) if any(s['kind'] == 'BODY' for s in w)]
        self.assertGreater(len(body), 5)
        for window in body:
            header, = [s for s in window if s['kind'] == 'PROVENANCE_HEADER']
            self.assertTrue(header['text'].startswith('line-0 ppp'))
            self.assertIn('provenance continues in the windows of segment S1', header['text'])
        # When not even a short prefix fits, preparation fails instead of dropping provenance.
        with self.assertRaisesRegex(ValueError, 'provenance'):
            list(windows(segs, lambda text: len(text) + 5500, 6000))

    def test_corrupt_chunk_map_becomes_validation_failure(self):
        u=self.unit();u['source_offsets']['S0'][1] += 1
        row={'status':'success','text_sha256':u['text_sha256'],'result':result('hacking')}
        with self.assertRaises(ValidationError):
            validate(row,u,schemas()['hacking'])

    def test_positive_requires_exact_citation(self):
        body = 'A short citation: café 日本語'
        u=self.unit(body);r=result('hacking','yes')
        row={'status':'success','text_sha256':u['text_sha256'],'result':r}
        with self.assertRaises(ValidationError): validate(row,u,schemas()['hacking'])
        r['evidence']=[{'segment_id':'S0','chunk_id':1,'location':'body'}]
        resolved=validate(row,u,schemas()['hacking'])
        self.assertEqual(resolved[0]['quote'], body)

    def test_integer_valued_json_number_resolves_citation(self):
        u = self.unit()
        res = result('hacking', 'yes')
        res['evidence'] = [{'segment_id': 'S0', 'chunk_id': 1.0, 'location': 'body'}]
        row = {'status': 'success', 'text_sha256': u['text_sha256'], 'result': res}
        resolved = validate(row, u, schemas()['hacking'])
        res['evidence'][0]['chunk_id'] = 1
        self.assertEqual(resolved, validate(row, u, schemas()['hacking']))
        self.assertIs(type(resolved[0]['chunk_id']), int)
        for bad in (True, 1.5, '1', 0):
            with self.subTest(chunk_id=bad):
                res['evidence'][0]['chunk_id'] = bad
                with self.assertRaises(ValidationError):
                    validate(row, u, schemas()['hacking'])

    def test_abstention_does_not_pass_negative_control(self):
        name=next(c[0] for c in controls.CONTROLS if c[4].get('hacking') is False)
        rows=[{'item_id':'control-'+name,'axes':{a:{'assessment':'insufficient_evidence',
               'injection_detected':False} for a in schemas()}}]
        scored=controls.score(rows)
        check=next(c for c in scored['checks'] if c['control']==name and c['axis']=='hacking')
        self.assertFalse(check['passed'])

    def test_unrequested_controls_are_untested_not_passed(self):
        # A pass-2 report requests only selected axes; the rest are not_requested.
        rows=[{'item_id':'control-'+c[0],'axes':{a:{'assessment':'not_requested','injection_detected':False}
               for a in schemas()}} for c in controls.CONTROLS]
        scored=controls.score(rows)
        self.assertEqual((scored['passed'], scored['total']), (0, 0))
        self.assertEqual(scored['not_requested'], len(scored['checks']))
        self.assertTrue(all(c['passed'] is None for c in scored['checks']))

    def test_scanner_preserves_payload_fields_named_hash(self):
        doc={'report_id':'a','http':[{'request':{'post_data':{'hash':'payload signal','_id':'important'}}}]}
        text, lines=scanner.view(doc)
        self.assertIn('payload signal',text)
        for p,v in lines: self.assertEqual(resolve_pointer(doc,p),v)


class ReportSuccessTests(unittest.TestCase):
    setUp = PipelineTests.setUp
    prepare = PipelineTests.prepare
    record = PipelineTests.record

    def test_report_without_controls_has_no_control_checks(self):
        self.prepare([self.record()])
        summary = write_report(self.run/'report', *reconcile(self.run, []))
        self.assertEqual(summary['controls'],
                         {'checks': [], 'passed': 0, 'total': 0, 'not_requested': 0,
                          'failed': 0, 'pending': 0, 'unverified_candidates': 0})
        self.assertEqual(list(iter_jsonl(self.run/'report/control-results.jsonl')), [])
        self.assertEqual(summary['pending'], 3)
        for prose in ('scope', 'score_semantics', 'release_ready', 'release_note'):
            self.assertNotIn(prose, summary)

    def test_requested_controls_without_outputs_still_count(self):
        write_jsonl(self.data / 'resources/sample/records.jsonl', [self.record()])
        self.run = self.root / 'run'
        prepare(self.data, self.run, lambda chat: sum(len(m['content']) for m in chat),
                15000, config(), with_controls=True)
        summary = write_report(self.run/'report', *reconcile(self.run, []))
        scored = summary['controls']
        self.assertEqual((scored['passed'], scored['total'], scored['not_requested']), (0, 25, 0))
        self.assertTrue(all(c['passed'] is False for c in scored['checks']))
        self.assertEqual(len(list(iter_jsonl(self.run/'report/control-results.jsonl'))),
                         len(controls.CONTROLS))

    def test_controls_are_excluded_from_all_reviewer_files(self):
        write_jsonl(self.data/'resources/sample/records.jsonl', [self.record()])
        self.run = self.root/'run'
        prepared = prepare(self.data, self.run, lambda chat: sum(len(m['content']) for m in chat),
                           15000, config(), with_controls=True)
        requests = [r for r in iter_jsonl(self.run/'requests.jsonl') if r['custom_id'].startswith('control-')]
        rows = []
        for request, label in zip(requests, ('yes', 'no')):
            uid = request['custom_id']
            res = result(uid.rsplit(':', 1)[1], label)  # yes deliberately lacks citations
            rows.append({**answered(self.run, uid), 'status': 'success', 'result': res,
                         'logprobs': scored(res, .6)})
        write_worker(self.run, rows)
        summary = write_report(self.run/'report', *reconcile(self.run, [self.run/'outputs']))
        for name in ('item-results', 'failures', 'pending', 'unverified-candidates',
                     'near-misses', 'review-queue', 'injection-review'):
            self.assertFalse(any(r['source'] == controls.SOURCE
                                 for r in iter_jsonl(self.run/f'report/{name}.jsonl')), name)
        self.assertEqual(len(list(iter_jsonl(self.run/'report/judgments.jsonl'))), 2)
        self.assertEqual(summary['requests_expected'], prepared['requests'])
        self.assertEqual((summary['failed'], summary['pending'], summary['unverified_candidates']), (0, 3, 0))
        self.assertEqual(summary['controls']['failed'], 1)
        self.assertEqual(summary['controls']['unverified_candidates'], 1)
        self.assertEqual(summary['controls']['pending'], len(requests) - 2)

    def test_verified_positive_and_failed_citation_are_separate(self):
        self.prepare([self.record()])
        w=json.loads((self.run/'windows.jsonl').read_text().splitlines()[0])
        sid=next(k for k,v in w['segment_kinds'].items() if v=='BODY')
        rows=[]
        for axis in schemas():
            res=result(axis,'yes' if axis!='coordination' else 'no')
            if axis!='coordination':
                res['evidence']=[{'segment_id':sid,'chunk_id':1 if axis=='hacking' else 9999,'location':'test'}]
            res['injection_detected']=axis=='coordination'
            rows.append({**answered(self.run, w['window_id']+':'+axis),'axis':axis,
                         'status':'success','result':res})
        # Raw final-answer logprobs from the runner; the report derives the label score.
        text = json.dumps(rows[0]['result'])
        cut = text.index('"label": "') + len('"label": "')
        branch = [{'token': t, 'logprob': math.log(p)} for t, p in (('yes', .6), ('no', .3), ('ins', .1))]
        rows[0]['logprobs'] = {'content': [{'token': text[:cut], 'logprob': 0.0, 'top_logprobs': []},
                                           {'token': 'yes', 'logprob': math.log(.6), 'top_logprobs': branch},
                                           {'token': text[cut + 3:], 'logprob': 0.0, 'top_logprobs': []}]}
        write_worker(self.run,rows)
        parts=reconcile(self.run,[self.run/'outputs'])
        hacking = next(j for j in parts[2] if j['axis'] == 'hacking')
        self.assertAlmostEqual(hacking['p_yes'], .6)
        summary=write_report(self.run/'report',*parts)
        self.assertEqual(summary['unverified_candidates'],1)
        self.assertEqual(summary['assessments']['hacking'],{'positive':1})
        self.assertEqual(summary['assessments']['agentic_activity'],{'incomplete':1})
        self.assertEqual(len((self.run/'report/injection-review.jsonl').read_text().splitlines()),1)

        # A runner failure is incomplete, never a negative or an unverified positive.
        next(row for row in rows if row['axis'] == 'agentic_activity').update(
            status='failed', result=None, error='generation failed')
        rows = [row for row in rows if row['axis'] != 'coordination']
        write_worker(self.run, rows)
        parts = reconcile(self.run, [self.run/'outputs'])
        summary = write_report(self.run/'report', *parts)
        self.assertEqual(summary['failed'], 1)
        self.assertEqual(summary['pending'], 1)
        self.assertEqual(summary['unverified_candidates'], 0)
        self.assertEqual(summary['assessments']['agentic_activity'], {'incomplete': 1})
        self.assertEqual(summary['assessments']['coordination'], {'incomplete': 1})


def resolve_pointer(doc, pointer):
    """Follow an RFC 6901 pointer from scanner.view back into the document."""
    cur = doc
    for raw in pointer.split('/')[1:]:
        token = raw.replace('~1', '/').replace('~0', '~')
        cur = cur[int(token)] if isinstance(cur, list) else cur[token]
    return cur if isinstance(cur, str) else json.dumps(cur)


def scored(res, p_yes):
    """Runner-style final-answer logprobs that score the label at p_yes."""
    text = json.dumps(res)
    cut = text.index('"label": "') + len('"label": "')
    label = res['label']
    rest = (1 - p_yes) / 2
    branch = [{'token': t, 'logprob': math.log(p)} for t, p in (('yes', p_yes), ('no', rest), ('ins', rest))]
    first = {'yes': 'yes', 'no': 'no', 'insufficient_evidence': 'ins'}[label]
    return {'content': [{'token': text[:cut], 'logprob': 0.0, 'top_logprobs': []},
                        {'token': first, 'logprob': -0.1, 'top_logprobs': branch},
                        {'token': text[cut + len(first):], 'logprob': 0.0, 'top_logprobs': []}]}


class ReviewQueueTests(unittest.TestCase):
    setUp = PipelineTests.setUp
    prepare = PipelineTests.prepare
    record = PipelineTests.record

    def test_queue_holds_positives_abstentions_and_failures_by_confidence(self):
        self.prepare([self.record()])
        w = next(iter_jsonl(self.run/'windows.jsonl'))
        sid = next(k for k, v in w['segment_kinds'].items() if v == 'BODY')
        uid = lambda axis: w['window_id'] + ':' + axis
        positive = result('hacking', 'yes')
        positive.update(evidence=[{'segment_id': sid, 'chunk_id': 1, 'location': 'body'}],
                        categories=['sql_injection'])
        abstain = result('agentic_activity', 'insufficient_evidence')
        rows = [{**answered(self.run, uid('hacking')), 'status': 'success', 'result': positive,
                 'logprobs': scored(positive, .3)},
                {**answered(self.run, uid('agentic_activity')), 'status': 'success', 'result': abstain,
                 'logprobs': scored(abstain, .6)},
                {**answered(self.run, uid('coordination')), 'status': 'failed', 'result': None,
                 'error': 'ValueError: Nonterminal/length finish: length'}]
        write_worker(self.run, rows)
        parts = reconcile(self.run, [self.run/'outputs'])
        queue = parts[7]
        self.assertEqual([(q['axis'], q['queue_reason']) for q in queue],
                         [('hacking', 'positive'), ('agentic_activity', 'abstention'), ('coordination', 'failure')])
        self.assertEqual([q['confidence'] for q in queue], [.9, .1, None])  # model score first, missing last
        self.assertTrue(all('assessment' not in q for q in queue))
        self.assertEqual(queue[0]['findings_cyberattack'], ['SQL injection'])
        self.assertEqual(parts[1][0]['axes']['hacking']['confidence_positive_max'], .9)
        self.assertIsNone(parts[1][0]['axes']['agentic_activity']['confidence_positive_max'])
        # A confident abstention (certainty in its label, not support) never outranks a positive.
        abstain['confidence'] = .95
        rows[1]['logprobs'] = scored(abstain, .6)
        write_worker(self.run, rows)
        ordered = reconcile(self.run, [self.run/'outputs'])[7]
        self.assertEqual([q['axis'] for q in ordered], ['hacking', 'agentic_activity', 'coordination'])
        # Within a reason, equal confidence falls back to P(yes) from the label logprobs.
        items = {w['item_id']: {'shape': 'capture', 'request_urls': []}}
        tie = [{'unit_id': u, 'item_id': w['item_id'], 'axis': 'agentic_activity', 'validation_status': 'validated',
                'result': {'label': 'yes', 'confidence': .8}, 'p_yes': p} for u, p in (('a', .3), ('b', .6))]
        self.assertEqual([q['unit_id'] for q in review_queue(tie, items)], ['b', 'a'])
        parts = reconcile(self.run, [self.run/'outputs'])
        summary = write_report(self.run/'report', *parts)
        self.assertEqual(summary['review_queue'], {'positive': 1, 'abstention': 1, 'failure': 1})
        # A negative answer is not queued.
        rows[2].update(status='success', result=result('coordination'), error=None)
        write_worker(self.run, rows)
        self.assertNotIn('coordination', {q['axis'] for q in reconcile(self.run, [self.run/'outputs'])[7]})
        # A negative with high confidence is counted as an inconsistency, not ranked.
        rows[2]['result'] = {**result('coordination'), 'confidence': .9}
        write_worker(self.run, rows)
        summary = write_report(self.run/'report-neg', *reconcile(self.run, [self.run/'outputs']))
        self.assertEqual(summary['confidence_on_negatives'], {'validated_no': 1, 'above_half': 1})


class WorkerMembershipTests(unittest.TestCase):
    setUp = PipelineTests.setUp
    record = PipelineTests.record

    def prepared_run(self):
        from screening_judge.common import iter_jsonl
        rows = [self.record('a'), self.record('b', body='second body')]
        write_jsonl(self.data/'resources/sample/records.jsonl', rows)
        self.run = self.root/'run'
        prepare(self.data, self.run, lambda chat: sum(len(m['content']) for m in chat),
                15000, config(), with_controls=False, shards=2)
        return list(iter_jsonl(self.run/'requests.jsonl'))

    def worker(self, request, manifest, limit=None):
        from screening_judge.common import sha256_file
        axis = request['custom_id'].rsplit(':', 1)[1]
        row = {**answered(self.run, request['custom_id']), 'status': 'success', 'result': result(axis)}
        worker = write_worker(self.run, [])
        identity = json.loads((worker/'identity.json').read_text())
        identity.pop('identity')
        identity.update(manifest_hash=sha256_file(self.run/manifest), max_records=limit)
        digest = sha256_text(json.dumps(identity, sort_keys=True))
        (worker/'identity.json').write_text(json.dumps({'identity': digest, **identity}))
        write_jsonl(worker/'part-000000.jsonl', [{**row, 'run_identity': digest}])

    def test_shards_keep_each_windows_axes_together(self):
        requests = self.prepared_run()
        window = lambda row: row['custom_id'].rsplit(':', 1)[0]
        seen = []
        for i in range(2):
            ids = [window(r) for r in iter_jsonl(self.run/f'shards/part-{i}.jsonl')]
            self.assertEqual(ids, [w for w in dict.fromkeys(ids) for _ in schemas()])
            seen += ids
        self.assertEqual(sorted(seen), sorted(window(r) for r in requests))

    def test_output_must_belong_to_workers_own_shard(self):
        units = self.prepared_run()
        manifest = 'shards/part-0.jsonl'
        assigned = {r['custom_id'] for r in iter_jsonl(self.run/manifest)}
        foreign = next(unit for unit in units if unit['custom_id'] not in assigned)
        self.worker(foreign, manifest)
        with self.assertRaisesRegex(ValueError, 'outside.*manifest'):
            reconcile(self.run, [self.run/'outputs'])

    def test_output_must_respect_workers_selected_prefix(self):
        units = self.prepared_run()
        self.worker(units[1], 'requests.jsonl', limit=1)
        with self.assertRaisesRegex(ValueError, 'outside.*manifest'):
            reconcile(self.run, [self.run/'outputs'])

    def test_assigned_shard_output_is_accepted(self):
        units = self.prepared_run()
        manifest = 'shards/part-0.jsonl'
        assigned = next(iter_jsonl(self.run/manifest))
        self.worker(assigned, manifest, limit=1)
        parts = reconcile(self.run, [self.run/'outputs'])
        self.assertEqual(len(parts[2]), 1)
        self.assertEqual(len(parts[3]), 0)
        expected = {u['custom_id'] for u in units}
        accepted = assigned['custom_id']
        self.assertEqual({row['unit_id'] for row in parts[2]}, {accepted})
        self.assertEqual({row['unit_id'] for row in parts[4]}, expected - {accepted})

if __name__ == '__main__':
    unittest.main()
