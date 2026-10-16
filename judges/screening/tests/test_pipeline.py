import json
import tempfile
import unittest
from pathlib import Path

from screening_judge import controls, records, scanner
from screening_judge.common import sha256_text, write_jsonl, iter_jsonl
from screening_judge.prepare import prepare
from screening_judge.prompts import load_prompts, schemas
from screening_judge.render import render_citable, windows, validate_chunk_map
from screening_judge.report import reconcile, write_report
from screening_judge.validate import validate, ValidationError

ROOT = Path(__file__).resolve().parents[1]


def config():
    return {**json.loads((ROOT / 'config/engine.example.json').read_text()),
            'prompts': load_prompts(), 'schemas': schemas(), 'boundary_salt': 'test'}


def result(axis, label='no'):
    r = {'evidence': [], 'policy_basis': 'test', 'rationale': 'Test fixture.',
         'label': label, 'injection_detected': False, 'missing_evidence': []}
    if axis == 'hacking':
        r.update(categories=[], boundary='', authorization='', actions=[])
    elif axis == 'agentic_activity':
        r.update(claimed_actor='', provider_attribution='not_established')
    else:
        r.update(mechanism='none')
    return r


def write_worker(run, rows, cfg=None):
    from screening_judge.common import sha256_file
    cfg = cfg or json.loads((run / 'config.json').read_text())
    ident = {'config': cfg, 'manifest_hash': sha256_file(run / 'manifest.jsonl'),
             'runner_hash': 'test-runner', 'prompts_hash': 'test-prompts', 'max_records': None}
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

    def test_all_axes_counted_for_window_budget(self):
        rec = self.record(body='x'*3000)
        write_jsonl(self.data / 'resources/sample/records.jsonl', [rec])
        cfg = config()
        cfg['prompts']['axes'] = {axis: axis for axis in schemas()}
        def count(chat):
            return len(chat[-1]['content']) + (3500 if chat[-1]['content'].endswith('coordination') else 0)
        run = self.root / 'run'
        prepare(self.data, run, count, 7000, cfg, with_controls=False)
        from screening_judge.prompts import messages
        cfg = json.loads((run/'config.json').read_text())
        for line in (run/'manifest.jsonl').read_text().splitlines():
            row = json.loads(line)
            for axis in row['axes']:
                self.assertLessEqual(count(messages(cfg, row['shape'], axis, row['text'])), 7000)

    def test_worker_config_mismatch_rejected(self):
        self.prepare([self.record()])
        cfg = json.loads((self.run / 'config.json').read_text())
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

    def test_scanner_preserves_payload_fields_named_hash(self):
        doc={'report_id':'a','http':[{'request':{'post_data':{'hash':'payload signal','_id':'important'}}}]}
        text, lines=scanner.view(doc)
        self.assertIn('payload signal',text)
        for p,v in lines: self.assertEqual(scanner.resolve(doc,p),v)


class ReportSuccessTests(unittest.TestCase):
    setUp = PipelineTests.setUp
    prepare = PipelineTests.prepare
    record = PipelineTests.record
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
            rows.append({'unit_id':w['window_id']+':'+axis,'window_id':w['window_id'],'axis':axis,
                         'chunk_idx':w['chunk_idx'],'text_sha256':w['text_sha256'],'status':'success','result':res})
        write_worker(self.run,rows)
        parts=reconcile(self.run,[self.run/'outputs'])
        summary=write_report(self.run/'report',*parts,None)
        self.assertEqual(summary['unverified_candidates'],1)
        self.assertEqual(summary['assessments']['hacking'],{'positive_uncalibrated':1})
        self.assertEqual(summary['assessments']['agentic_activity'],{'incomplete':1})
        self.assertEqual(len((self.run/'report/injection-review.jsonl').read_text().splitlines()),1)

        # A runner failure is incomplete, never a negative or an unverified positive.
        next(row for row in rows if row['axis'] == 'agentic_activity').update(
            status='failed', result=None, error='generation failed')
        rows = [row for row in rows if row['axis'] != 'coordination']
        write_worker(self.run, rows)
        parts = reconcile(self.run, [self.run/'outputs'])
        summary = write_report(self.run/'report', *parts, None)
        self.assertEqual(summary['failed'], 1)
        self.assertEqual(summary['pending'], 1)
        self.assertEqual(summary['unverified_candidates'], 0)
        self.assertEqual(summary['assessments']['agentic_activity'], {'incomplete': 1})
        self.assertEqual(summary['assessments']['coordination'], {'incomplete': 1})


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
        return list(iter_jsonl(self.run/'manifest.jsonl'))

    def worker(self, unit, manifest, limit=None):
        from screening_judge.common import sha256_file
        axis = 'hacking'
        row = {'unit_id': unit['unit_id']+':'+axis, 'window_id': unit['unit_id'],
               'axis': axis, 'chunk_idx': unit['chunk_idx'], 'text_sha256': unit['text_sha256'],
               'status': 'success', 'result': result(axis)}
        worker = write_worker(self.run, [])
        identity = json.loads((worker/'identity.json').read_text())
        identity.pop('identity')
        identity.update(manifest_hash=sha256_file(self.run/manifest), max_records=limit)
        digest = sha256_text(json.dumps(identity, sort_keys=True))
        (worker/'identity.json').write_text(json.dumps({'identity': digest, **identity}))
        write_jsonl(worker/'part-000000.jsonl', [{**row, 'run_identity': digest}])

    def test_output_must_belong_to_workers_own_shard(self):
        units = self.prepared_run()
        manifest = 'shards/manifest-0000.jsonl'
        assigned = {r['unit_id'] for r in iter_jsonl(self.run/manifest)}
        foreign = next(unit for unit in units if unit['unit_id'] not in assigned)
        self.worker(foreign, manifest)
        with self.assertRaisesRegex(ValueError, 'outside.*manifest'):
            reconcile(self.run, [self.run/'outputs'])

    def test_output_must_respect_workers_selected_prefix(self):
        units = self.prepared_run()
        self.worker(units[1], 'manifest.jsonl', limit=1)
        with self.assertRaisesRegex(ValueError, 'outside.*manifest'):
            reconcile(self.run, [self.run/'outputs'])

    def test_assigned_shard_output_is_accepted(self):
        units = self.prepared_run()
        manifest = 'shards/manifest-0000.jsonl'
        assigned = next(iter_jsonl(self.run/manifest))
        self.worker(assigned, manifest, limit=1)
        parts = reconcile(self.run, [self.run/'outputs'])
        self.assertEqual(len(parts[2]), 1)
        self.assertEqual(len(parts[3]), 0)
        expected = {f"{u['unit_id']}:{axis}" for u in units for axis in u['axes']}
        accepted = assigned['unit_id'] + ':hacking'
        self.assertEqual({row['unit_id'] for row in parts[2]}, {accepted})
        self.assertEqual({row['unit_id'] for row in parts[4]}, expected - {accepted})

if __name__ == '__main__':
    unittest.main()
