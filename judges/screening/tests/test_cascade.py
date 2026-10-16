import json
import unittest

from screening_judge import cascade, rules
from screening_judge.common import iter_jsonl
from screening_judge.report import reconcile, write_report
import test_pipeline as tp
from test_pipeline import answered, config, result, scored, write_worker

THINKING = {**config(), 'sampling': {'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'max_tokens': 16384},
            'chat_template_kwargs': {'enable_thinking': True}}


class RuleTests(unittest.TestCase):
    def test_rules_fire_on_their_patterns_only(self):
        cases = {
            'backend_port': ('GET http://shop.example:8089/admin', 'https://shop.example:443/ and http://a.example/'),
            'direct_ip_host_override': ('curl https://203.0.113.7/login -H "Host: bank.example"',
                                        'https://203.0.113.7/status with no header override'),
            'file_version_path': ('PROPFIND /remote.php/dav/versions/alice/', 'release notes for version 2'),
            'captcha_ocr_chain': ('fetched captcha.png then ran tesseract on it', 'a captcha on the login page'),
        }
        for name, (hit, miss) in cases.items():
            with self.subTest(rule=name):
                self.assertIn(name, rules.hits(hit))
                self.assertNotIn(name, rules.hits(miss))


class CascadeTests(unittest.TestCase):
    setUp = tp.PipelineTests.setUp
    record = tp.PipelineTests.record

    def pass1(self, outcomes):
        """Prepare three items and commit pass-1 answers: {item: {axis: (label, p_yes)}}."""
        from screening_judge.common import write_jsonl
        from screening_judge.prepare import prepare
        bodies = {'a': 'ordinary text', 'b': 'PROPFIND /remote.php/dav/versions/alice/', 'c': 'weather report'}
        write_jsonl(self.data/'resources/sample/records.jsonl', [self.record(k, v) for k, v in bodies.items()])
        self.run = self.root/'pass1'
        prepare(self.data, self.run, lambda chat: sum(len(m['content']) for m in chat), 15000, config(),
                with_controls=False)
        windows = {w['item_id']: w for w in iter_jsonl(self.run/'windows.jsonl')}
        items = {i['observations'][0]['observation_id']: i['item_id'] for i in iter_jsonl(self.run/'items.jsonl')}
        rows, self.uid = [], {}
        for name, axes in outcomes.items():
            w = windows[items[name]]
            sid = next(k for k, v in w['segment_kinds'].items() if v == 'BODY')
            for axis, (label, p_yes) in axes.items():
                uid = self.uid[(name, axis)] = f"{w['window_id']}:{axis}"
                res = result(axis, label)
                if label == 'yes':
                    res['evidence'] = [{'segment_id': sid, 'chunk_id': 1, 'location': 'body'}]
                rows.append({**answered(self.run, uid), 'status': 'success', 'result': res,
                             'logprobs': scored(res, p_yes)})
        write_worker(self.run, rows)
        write_report(self.run/'report', *reconcile(self.run, [self.run/'outputs']), None)

    def outcomes(self, overrides=None):
        base = {name: {axis: ('no', .05) for axis in ('hacking', 'agentic_activity', 'coordination')}
                for name in 'abc'}
        for (name, axis), value in (overrides or {}).items():
            base[name][axis] = value
        return base

    def test_escalation_selects_by_reason_and_writes_an_ordinary_run(self):
        self.pass1(self.outcomes({('a', 'hacking'): ('yes', .9), ('a', 'agentic_activity'): ('no', .5),
                                  ('c', 'coordination'): ('insufficient_evidence', .1)}))
        out = self.root/'pass2'
        summary = cascade.escalate(self.run, self.run/'report', out, THINKING, audit_rate=0)
        selection = {r['custom_id']: r['reasons'] for r in iter_jsonl(out/'selection.jsonl')}
        self.assertEqual(selection, {
            self.uid[('a', 'hacking')]: ['positive'],
            self.uid[('a', 'agentic_activity')]: ['borderline'],
            self.uid[('c', 'coordination')]: ['abstention'],
            **{self.uid[('b', axis)]: ['rule:file_version_path']
               for axis in ('hacking', 'agentic_activity', 'coordination')}})
        self.assertEqual(summary['selection'], {'positive': 1, 'borderline': 1, 'abstention': 1, 'rule': 3})
        requests = list(iter_jsonl(out/'requests.jsonl'))
        original = {r['custom_id']: r for r in iter_jsonl(self.run/'requests.jsonl')}
        self.assertEqual([r['custom_id'] for r in requests], [u for u in original if u in selection])
        self.assertTrue(all(r == original[r['custom_id']] for r in requests))  # same requests, new profile
        self.assertEqual(json.loads((out/'engine.json').read_text()), THINKING)
        # The pass-2 run reports only its own requests: nothing else is pending.
        parts = reconcile(out, [])
        self.assertEqual(len(parts[4]), 6)
        a = next(i for i in parts[1] if any(u.startswith(i['item_id']) for u in (self.uid[('a', 'hacking')],)))
        self.assertEqual(a['axes']['coordination']['assessment'], 'not_requested')
        self.assertEqual(a['axes']['hacking']['assessment'], 'incomplete')  # selected, not yet run

    def test_audit_is_seeded_and_pending_pass1_is_refused(self):
        self.pass1(self.outcomes())
        all_no = cascade.escalate(self.run, self.run/'report', self.root/'audit', THINKING, audit_rate=1.0)
        self.assertEqual(all_no['selection'], {'audit': 6, 'rule': 3})
        again = cascade.escalate(self.run, self.run/'report', self.root/'audit2', THINKING, audit_rate=.5, seed=4)
        same = cascade.escalate(self.run, self.run/'report', self.root/'audit3', THINKING, audit_rate=.5, seed=4)
        self.assertEqual((self.root/'audit2/selection.jsonl').read_text(), (self.root/'audit3/selection.jsonl').read_text())
        self.assertEqual(again['selection'], same['selection'])
        with (self.run/'report/pending.jsonl').open('a') as f:
            f.write(json.dumps({'unit_id': 'x'}) + '\n')
        with self.assertRaisesRegex(ValueError, 'pending'):
            cascade.escalate(self.run, self.run/'report', self.root/'refused', THINKING)
        with self.assertRaisesRegex(ValueError, 'runner profile'):
            cascade.escalate(self.run, self.run/'report', self.root/'bad', {**THINKING, 'citation_constraints': True},
                             allow_pending=True)

    def test_combine_keeps_demotions_visible_and_counts_audit_flips(self):
        self.pass1(self.outcomes({('a', 'hacking'): ('yes', .9), ('a', 'agentic_activity'): ('no', .5)}))
        out = self.root/'pass2'
        cascade.escalate(self.run, self.run/'report', out, THINKING, audit_rate=1.0)
        selected = [r['custom_id'] for r in iter_jsonl(out/'requests.jsonl')]
        flips = {self.uid[('a', 'hacking')]: 'no', self.uid[('a', 'agentic_activity')]: 'yes',
                 self.uid[('c', 'hacking')]: 'yes'}
        windows = {w['window_id']: w for w in iter_jsonl(out/'windows.jsonl')}
        rows = []
        for uid in selected:
            axis = uid.rsplit(':', 1)[1]
            res = result(axis, flips.get(uid, 'no'))
            if res['label'] == 'yes':
                w = windows[uid.rsplit(':', 1)[0]]
                sid = next(k for k, v in w['segment_kinds'].items() if v == 'BODY')
                res['evidence'] = [{'segment_id': sid, 'chunk_id': 1, 'location': 'body'}]
            rows.append({**answered(out, uid), 'status': 'success', 'result': res})
        write_worker(out, rows)
        write_report(out/'report', *reconcile(out, [out/'outputs']), None)
        summary = cascade.combine(self.run/'report', out, out/'report', self.root/'final')
        self.assertEqual((summary['demoted'], summary['promoted']), (1, 2))
        self.assertEqual(summary['audit']['audited'], 4)  # a:coordination and c's axes; b's came from a rule
        self.assertEqual(summary['audit']['pass2_positive'], 1)
        demoted = list(iter_jsonl(self.root/'final/demoted.jsonl'))
        self.assertEqual([d['unit_id'] for d in demoted], [self.uid[('a', 'hacking')]])
        self.assertEqual(demoted[0]['pass1']['result']['label'], 'yes')
        queue = list(iter_jsonl(self.root/'final/review-queue.jsonl'))
        self.assertEqual({q['unit_id'] for q in queue if q['queue_reason'] == 'positive'},
                         {self.uid[('a', 'agentic_activity')], self.uid[('c', 'hacking')]})
        self.assertNotIn(self.uid[('a', 'hacking')], {q['unit_id'] for q in queue})
        final = {j['unit_id']: j for j in iter_jsonl(self.root/'final/final-judgments.jsonl')}
        self.assertEqual(len(final), 9)
        self.assertEqual({j['pass'] for j in final.values()}, {2})  # every request was escalated here
        with self.assertRaisesRegex(ValueError, 'differs'):
            (self.run/'report/judgments.jsonl').write_text('')
            cascade.combine(self.run/'report', out, out/'report', self.root/'final2')

    def pass2_answers(self, out, labels=None, bad_citation=()):
        """Commit pass-2 answers for every selected request (default no)."""
        windows = {w['window_id']: w for w in iter_jsonl(out/'windows.jsonl')}
        rows = []
        for r in iter_jsonl(out/'requests.jsonl'):
            uid = r['custom_id']
            res = result(uid.rsplit(':', 1)[1], (labels or {}).get(uid, 'no'))
            if res['label'] == 'yes':
                w = windows[uid.rsplit(':', 1)[0]]
                sid = next(k for k, v in w['segment_kinds'].items() if v == 'BODY')
                res['evidence'] = [{'segment_id': sid, 'chunk_id': 9999 if uid in bad_citation else 1,
                                    'location': 'body'}]
            rows.append({**answered(out, uid), 'status': 'success', 'result': res})
        write_worker(out, rows)
        write_report(out/'report', *reconcile(out, [out/'outputs']), None)

    def test_reports_are_bound_to_their_own_prepared_run(self):
        self.pass1(self.outcomes())
        a, b = self.root/'pass2a', self.root/'pass2b'
        cascade.escalate(self.run, self.run/'report', a, THINKING, audit_rate=1.0)
        cascade.escalate(self.run, self.run/'report', b, {**THINKING, 'sampling': {**THINKING['sampling'], 'seed': 3}},
                         audit_rate=1.0)
        self.pass2_answers(a)
        with self.assertRaisesRegex(ValueError, 'does not belong'):
            cascade.combine(self.run/'report', b, a/'report', self.root/'swapped')
        with self.assertRaisesRegex(ValueError, 'does not belong'):
            cascade.combine(self.run/'report', a, self.run/'report', self.root/'pass1-as-pass2')
        with self.assertRaisesRegex(ValueError, 'pass-1'):
            cascade.escalate(a, a/'report', self.root/'again', THINKING)
        cascade.combine(self.run/'report', a, a/'report', self.root/'ok')

    def test_unresolved_pass2_answers_stay_in_the_queue(self):
        self.pass1(self.outcomes())
        out = self.root/'pass2'
        cascade.escalate(self.run, self.run/'report', out, THINKING, audit_rate=1.0)
        uid = self.uid[('c', 'hacking')]
        self.pass2_answers(out, {uid: 'yes'}, bad_citation={uid})
        summary = cascade.combine(self.run/'report', out, out/'report', self.root/'final')
        final = {j['unit_id']: j for j in iter_jsonl(self.root/'final/final-judgments.jsonl')}
        self.assertEqual((final[uid]['pass'], final[uid]['result']['label']), (1, 'no'))  # pass 1 kept
        queued = [q for q in iter_jsonl(self.root/'final/review-queue.jsonl') if q['unit_id'] == uid]
        self.assertEqual([(q['pass'], q['queue_reason']) for q in queued], [(2, 'unverified_positive')])
        self.assertEqual((summary['pass2_failed'], summary['pass2_unresolved_queued']), (1, 1))
        # Six audited requests; the failed one is not counted among validated audit answers.
        self.assertEqual(summary['audit'], {'audited': 5, 'pass2_positive': 0, 'note': summary['audit']['note']})

    def test_controls_stay_out_of_cascade_queues_and_counts(self):
        from screening_judge.common import write_jsonl
        from screening_judge.controls import SOURCE
        from screening_judge.prepare import prepare
        write_jsonl(self.data/'resources/sample/records.jsonl', [self.record('a', 'ordinary text')])
        self.run = self.root/'pass1'
        prepare(self.data, self.run, lambda chat: sum(len(m['content']) for m in chat), 15000, config())
        rows = [{**answered(self.run, r['custom_id']), 'status': 'success',
                 'result': result(r['custom_id'].rsplit(':', 1)[1], 'insufficient_evidence')}
                for r in iter_jsonl(self.run/'requests.jsonl')]
        write_worker(self.run, rows)
        write_report(self.run/'report', *reconcile(self.run, [self.run/'outputs']), None)
        out = self.root/'pass2'
        cascade.escalate(self.run, self.run/'report', out, THINKING, audit_rate=0)
        self.pass2_answers(out)
        summary = cascade.combine(self.run/'report', out, out/'report', self.root/'final')
        controls = list(iter_jsonl(self.root/'final/control-judgments.jsonl'))
        self.assertTrue(controls and all(c['source'] == SOURCE for c in controls))
        for name in ('final-judgments', 'review-queue', 'demoted'):
            self.assertFalse(any(r['source'] == SOURCE for r in iter_jsonl(self.root/f'final/{name}.jsonl')))
        self.assertEqual(summary['requests'], 3)  # the corpus item's three axes only

    def test_shards_never_outnumber_selected_windows(self):
        self.pass1(self.outcomes({('a', 'hacking'): ('yes', .9)}))
        out = cascade.escalate(self.run, self.run/'report', self.root/'one', THINKING, audit_rate=0, shards=8)
        # Window a (positive) and window b (rule hit) are selected: two shards, none empty.
        names = sorted(p.name for p in (self.root/'one/shards').iterdir())
        self.assertEqual(names, ['part-0.jsonl', 'part-1.jsonl'])
        self.assertTrue(all((self.root/'one/shards'/n).stat().st_size for n in names))
        self.assertEqual(len(out['shard_characters']), 2)

    def test_escalation_that_selects_nothing_writes_no_shards(self):
        from screening_judge.common import write_jsonl
        from screening_judge.prepare import prepare
        write_jsonl(self.data/'resources/sample/records.jsonl', [self.record('a', 'ordinary text')])
        self.run = self.root/'pass1'
        prepare(self.data, self.run, lambda chat: sum(len(m['content']) for m in chat), 15000, config(),
                with_controls=False)
        write_worker(self.run, [{**answered(self.run, r['custom_id']), 'status': 'success',
                                 'result': result(r['custom_id'].rsplit(':', 1)[1])}
                                for r in iter_jsonl(self.run/'requests.jsonl')])
        write_report(self.run/'report', *reconcile(self.run, [self.run/'outputs']), None)
        summary = cascade.escalate(self.run, self.run/'report', self.root/'none', THINKING, audit_rate=0, shards=8)
        self.assertEqual(summary['requests'], 0)
        self.assertIn('nothing to run', summary['note'])
        self.assertFalse((self.root/'none/shards').exists())


if __name__ == '__main__':
    unittest.main()
