"""Observed failure regressions and legacy compatibility; no model-accuracy claims."""
import copy
import json
import unittest
from jsonschema import Draft202012Validator
from screening_judge.schemas import schema, citation_schema
from screening_judge.validate import validate, ValidationError
from screening_judge.warnings import review_warnings
from screening_judge.report import review_queue, content_families
from screening_judge.cascade import _pass1_rank
from screening_judge.shapes import build_segments
import test_pipeline as tp
from test_pipeline import result
from screening_judge.common import iter_jsonl


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

    def test_legacy_actions_still_validate_against_the_saved_schema(self):
        w,citation,action = self.setup_action()
        # Recreate the old field layout: saved v1 schemas, not current v2, govern old rows.
        old = copy.deepcopy(schema('hacking')['anyOf'][0]); props=old['properties']
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

    def test_collection_path_neutralized_only_in_provenance_full_audit_preserved(self):
        path='https://raw.githubusercontent.com/JoshuaDavid/WikiAgentSwarmInvestigation/abc/agent-logs/milkwiki/revisions.jsonl'
        self.prepare([self.record(body='A source link: '+path,kind='prior_repository_revision',source_url=path)])
        item=next(iter_jsonl(self.run/'items.jsonl'));w=next(iter_jsonl(self.run/'windows.jsonl'))
        self.assertIn(path,item['observations'][0]['line'])
        texts={sid:''.join(w['text'][c['text_start']:c['text_end']] for c in cs) for sid,cs in w['evidence_chunks'].items()}
        self.assertTrue(all('WikiAgentSwarmInvestigation' not in texts[s] for s,k in w['segment_kinds'].items() if k=='PROVENANCE'))
        self.assertTrue(any(path in texts[s] for s,k in w['segment_kinds'].items() if k=='BODY'))

    def test_collection_source_url_fallback_cannot_leak(self):
        path='https://raw.githubusercontent.com/JoshuaDavid/WikiAgentSwarmInvestigation/abc/agent-logs/milkwiki/revisions.jsonl'
        rec=self.record(body='ordinary text',kind='wayback_capture',source_url=path)
        self.prepare([rec])
        item=next(iter_jsonl(self.run/'items.jsonl'))
        self.assertEqual(item['request_urls'],[path])  # full audit trail
        for request in iter_jsonl(self.run/'requests.jsonl'):
            self.assertNotIn('WikiAgentSwarmInvestigation',json.dumps(request))
        w=next(iter_jsonl(self.run/'windows.jsonl'))
        self.assertTrue(any(source.startswith('request_url:neutral_collection_view')
                            for source in w['segment_sources'].values()))

    def test_unhandled_collection_metadata_leak_stops_preparation(self):
        with self.assertRaisesRegex(ValueError,'repository name leaked'):
            self.prepare([self.record(metadata={'collection':'WikiAgentSwarmInvestigation'})])
        self.assertFalse((self.run/'preparation.json').exists())

    def test_body_groups_keep_context_disagreements_and_do_not_merge_missing_bodies(self):
        axes=lambda label:{a:{'assessment':label} for a in ('hacking','agentic_activity','coordination')}
        rows=[{'item_id':str(n),'source':'site','text_sha256s':h,'axes':axes(l)} for n,h,l in
              [(1,['hash'],'positive'),(2,['hash'],'no_evidence'),(3,[],'incomplete'),(4,[],'incomplete')]]
        groups=content_families(rows)
        self.assertEqual(len(groups),1)
        self.assertEqual(groups[0]['item_ids'],['1','2'])
        self.assertEqual(groups[0]['axes']['hacking'],{'positive':1,'no_evidence':1})
