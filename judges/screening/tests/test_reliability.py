"""Observed failure regressions and legacy compatibility; no model-accuracy claims."""
import copy
import unittest
from jsonschema import Draft202012Validator
from screening_judge.schemas import schema, citation_schema, _base_schema
from screening_judge.validate import validate, ValidationError
from screening_judge.warnings import review_warnings
from screening_judge.report import review_queue, content_families
from screening_judge.cascade import _pass1_rank
import test_pipeline as tp
from test_pipeline import result
from screening_judge.common import iter_jsonl
from screening_judge.records import import_location


class ReliabilityTests(unittest.TestCase):
    setUp = tp.PipelineTests.setUp
    record = tp.PipelineTests.record
    prepare = tp.PipelineTests.prepare

    def setup_action(self):
        self.prepare([self.record(body='GET /secret HTTP/1.1')])
        w = next(iter_jsonl(self.run/'windows.jsonl'))
        sid = next(s for s,k in w['segment_kinds'].items() if k == 'BODY')
        citation = {'segment_id': sid, 'chunk_id': 1, 'location': 'request'}
        action = {'target': 'target', 'action': 'probe secret', 'observed_stage': 'recorded_request',
                  'responding_system': 'unknown', 'outcome': 'unknown', 'evidence': [citation]}
        return w, citation, action

    def test_gate_and_action_local_citations_resolve_without_top_level_copy(self):
        w, citation, action = self.setup_action()
        res = {**result('hacking','yes'), 'attempt_present': True, 'actions':[action]}
        constrained = citation_schema('hacking', w['evidence_chunks'])
        validator = Draft202012Validator(constrained)
        self.assertTrue(validator.is_valid(res))
        evidence = validate({'status':'success','result':res},w,schema('hacking'))
        self.assertEqual([e['quote'] for e in evidence], ['GET /secret HTTP/1.1'])
        # The old two-list mismatch is impossible: action evidence is resolved directly.
        res['evidence'] = [citation]
        self.assertEqual(len(validate({'status':'success','result':res},w,schema('hacking'))),1)
        for changes in ({'attempt_present':False}, {'actions':[]},
                        {'actions':[{**action,'evidence':[]}]},
                        {'actions':[{**action,'evidence':[{**citation,'chunk_id':99}]}]}):
            with self.subTest(changes=changes):
                self.assertFalse(validator.is_valid({**res,**changes}))
        res.update(attempt_present=False,actions=[],label='no')
        self.assertTrue(validator.is_valid(res))

    def test_no_citable_chunks_has_only_the_empty_action_branch(self):
        validator=Draft202012Validator(citation_schema('hacking',{}))
        res=result('hacking','insufficient_evidence')
        self.assertTrue(validator.is_valid(res))
        self.assertFalse(validator.is_valid({**res,'attempt_present':True}))
        self.assertFalse(validator.is_valid({**res,'evidence':[
            {'segment_id':'S0','chunk_id':1,'location':'unavailable'}]}))

    def test_positive_requires_citations_on_every_axis(self):
        w, citation, _ = self.setup_action()
        for axis in ('hacking', 'agentic_activity', 'coordination'):
            for spec in (schema(axis), citation_schema(axis, w['evidence_chunks'])):
                validator = Draft202012Validator(spec)
                with self.subTest(axis=axis, spec=spec):
                    self.assertFalse(validator.is_valid(result(axis, 'yes')))
                    for label in ('no', 'insufficient_evidence'):
                        self.assertTrue(validator.is_valid(result(axis, label)))
                    cited = {**result(axis, 'yes'), 'evidence': [citation]}
                    self.assertTrue(validator.is_valid(cited))
                    self.assertEqual(len(validate({'status':'success', 'result':cited}, w, spec)), 1)
            empty = Draft202012Validator(citation_schema(axis, {}))
            self.assertFalse(empty.is_valid(result(axis, 'yes')))
            self.assertTrue(empty.is_valid(result(axis, 'insufficient_evidence')))

    def test_legacy_actions_still_validate_against_the_saved_schema(self):
        w,citation,action = self.setup_action()
        # Recreate the old field layout: saved v1 schemas, not current v2, govern old rows.
        old = copy.deepcopy(_base_schema('hacking')); props=old['properties']
        del props['attempt_present'];old['required'].remove('attempt_present')
        action_spec=props['actions']['items'];props['actions'].pop('maxItems')
        del action_spec['properties']['evidence'];action_spec['required'].remove('evidence')
        action_spec['properties']['evidence_segment_ids']={'type':'array','items':{'type':'string'}}
        action_spec['required'].append('evidence_segment_ids')
        action.pop('evidence');action['evidence_segment_ids']=[citation['segment_id']]
        res={**result('hacking','yes'),'actions':[action],'evidence':[citation]};res.pop('attempt_present')
        self.assertEqual(len(validate({'status':'success','result':res},w,old)),1)
        res['evidence']=[]
        with self.assertRaises(ValidationError): validate({'status':'success','result':res},w,old)

    def test_scores_cannot_change_queue_or_demotion_order(self):
        items={'i':{'shape':'text','request_urls':[]}}
        rows=[{'unit_id':u,'item_id':'i','axis':'coordination','validation_status':'validated',
               'result':{'label':'yes','confidence':score},'p_yes':score} for u,score in [('b',.99),('a',.01)]]
        before=[r['unit_id'] for r in review_queue(rows,items)]
        demoted=[{'unit_id':r['unit_id'],'pass1':r} for r in rows]
        order=[r['unit_id'] for r in sorted(demoted,key=_pass1_rank)]
        for r in rows:r['result']['confidence']=1-r['result']['confidence'];r['p_yes']=1-r['p_yes']
        self.assertEqual(before,[r['unit_id'] for r in review_queue(rows,items)])
        self.assertEqual(order,[r['unit_id'] for r in sorted(demoted,key=_pass1_rank)])

    def test_warnings_do_not_rewrite_results_and_respect_confidence_version(self):
        res={**result('agentic_activity'),'confidence':.99,'provider_attribution':'claimed','claimed_actor':'Green AI Bot'}
        original=copy.deepcopy(res)
        self.assertIn('legacy_confidence_polarity',review_warnings(res,[]))
        flags=review_warnings(res,[],'chosen_label')
        self.assertNotIn('legacy_confidence_polarity',flags)
        self.assertIn('provider_attribution_needs_named_model_review',flags)
        self.assertEqual(res,original)
        res['claimed_actor']='Claude'
        self.assertEqual(review_warnings(res,[],'chosen_label'),[])
        a={'observed_stage':'recorded_request','evidence':[{'segment_id':'S2','chunk_id':1}]}
        evidence=[{'segment_id':'S2','chunk_id':1,'quote':'Configuration: connect_timeout=30; logging level DEBUG'}]
        self.assertIn('recorded_request_cites_setup_review',review_warnings({'actions':[a]},evidence))
        evidence[0]['quote']+='\nGET /x HTTP/1.1'
        self.assertEqual(review_warnings({'actions':[a]},evidence),[])

    def texts(self, kind):
        w = next(iter_jsonl(self.run/'windows.jsonl'))
        return [''.join(w['text'][c['text_start']:c['text_end']] for c in w['evidence_chunks'][sid])
                for sid, k in w['segment_kinds'].items() if k == kind]

    def test_import_location_hidden_from_prompt_metadata_but_kept_for_audit(self):
        path = 'https://raw.githubusercontent.com/someone/collection/abc/agent-logs/wiki/revisions.jsonl'
        self.prepare([self.record(body='A source link: '+path, kind='prior_mapping',
                                  collection_method='imported', source_url=path)])
        item = next(iter_jsonl(self.run/'items.jsonl'))
        self.assertIn(path, item['observations'][0]['line'])
        self.assertEqual(item['observations'][0]['import_location'], path)
        self.assertEqual(item['request_urls'], [])
        self.assertTrue(all('agent-logs' not in t for t in self.texts('PROVENANCE') + self.texts('URL')))
        self.assertTrue(any(path in t for t in self.texts('BODY')))  # document text is evidence

    def test_source_site_and_directly_collected_urls_stay_visible(self):
        urls = ['https://www.sample/view/abc', 'https://elsewhere.example/agent-logs/x']
        self.prepare([self.record('own', kind='prior_mapping', collection_method='imported', source_url=urls[0]),
                      self.record('direct', kind='prior_mapping', collection_method='wayback', source_url=urls[1])])
        items = list(iter_jsonl(self.run/'items.jsonl'))
        self.assertEqual(sorted(u for i in items for u in i['request_urls']), sorted(urls))
        self.assertFalse(any('import_location' in o for i in items for o in i['observations']))

    def test_request_url_skips_import_location_and_keeps_records_apart(self):
        export = 'https://raw.githubusercontent.com/someone/collection/main/links.csv'
        targets = ['https://target.example/?q=%27%20OR%201=1--', 'https://other.example/page']
        self.prepare([self.record(str(n), kind='prior_mapping', collection_method='imported',
                                  source_url=export, long_url=url, retrieved_at='2026-01-01')
                      for n, url in enumerate(targets)])
        items = list(iter_jsonl(self.run/'items.jsonl'))
        self.assertEqual(sorted(u for i in items for u in i['request_urls']), sorted(targets))

    def test_redaction_rewrites_only_the_source_url_field(self):
        self.prepare([self.record(collection_method='imported', source_url='data', note='metadata dump')])
        provenance = self.texts('PROVENANCE')[0]
        self.assertIn('note=metadata dump', provenance)
        self.assertIn('source_url=[import location; kept in audit record]', provenance)

    def test_source_site_match_ignores_case(self):
        rec = {'collection_method': 'imported', 'source_url': 'https://www.example.com/page'}
        self.assertIsNone(import_location(rec, 'Example.com'))

    def test_schema_failure_names_the_failing_field(self):
        w, _, _ = self.setup_action()
        bad = {**result('coordination'), 'mechanism': 'bogus'}
        with self.assertRaisesRegex(ValidationError, r"^Schema: mechanism: 'bogus' is not one of"):
            validate({'status': 'success', 'result': bad}, w, schema('coordination'))

    def test_body_groups_keep_context_disagreements_and_do_not_merge_missing_bodies(self):
        axes=lambda label:{a:{'assessment':label} for a in ('hacking','agentic_activity','coordination')}
        rows=[{'item_id':str(n),'source':'site','text_sha256s':h,'axes':axes(l)} for n,h,l in
              [(1,['hash'],'positive'),(2,['hash'],'no_evidence'),(3,[],'incomplete'),(4,[],'incomplete')]]
        groups=content_families(rows)
        self.assertEqual(len(groups),1)
        self.assertEqual(groups[0]['item_ids'],['1','2'])
        self.assertEqual(groups[0]['axes']['hacking'],{'positive':1,'no_evidence':1})
