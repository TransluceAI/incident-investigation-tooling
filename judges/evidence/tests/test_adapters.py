import base64
import gzip
import json
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest.mock import patch

from evidence_judge import decode, controls, shapes
from evidence_judge.common import iter_jsonl, write_jsonl, sha256_text, classifier_hash
from evidence_judge.prepare import prepare
from evidence_judge.report import reconcile
from fixtures import build
from test_pipeline import config, write_worker

class AdapterTests(unittest.TestCase):
    def test_base64_path_zlib_gzip_and_code_key(self):
        text='{"country":"PT","year":2020}'
        for compressor in (lambda b:b,zlib.compress,gzip.compress):
            encoded=base64.urlsafe_b64encode(compressor(text.encode())).decode()
            for url in ('https://a.example/'+encoded,'https://a.example/?payload='+encoded):
                views,gaps=decode.decode_url(url)
                self.assertFalse(gaps)
                self.assertTrue(any(v['text']==text for v in views))

    def test_compression_limit_and_incomplete_stream_are_gaps(self):
        payload=zlib.compress(b'ordinary text '*1000)
        with patch.object(decode,'MAX_INFLATED',100):
            views,gaps=decode.decode_url('https://a.example/'+base64.urlsafe_b64encode(payload).decode())
            self.assertTrue(gaps)
            self.assertFalse(any(v['kind']=='DECODED_B64' for v in views))
        views,gaps=decode.decode_url('https://a.example/'+base64.urlsafe_b64encode(payload[:-3]).decode())
        self.assertTrue(gaps)
        self.assertFalse(any(v['kind']=='DECODED_B64' for v in views))

    def test_all_shapes_have_controls_and_names_not_in_prompt(self):
        self.assertEqual({c[1] for c in controls.CONTROLS},set(shapes.SHAPES))
        for item,bodies,_,_ in controls.control_items():
            text='\n'.join(s['text'] for s in shapes.build_segments(item,bodies))
            self.assertNotIn('control-'+item['item_id'],text)
            self.assertNotIn('observation_id='+item['item_id'],text)

    def test_multi_source_end_to_end_prepare_shard_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=build(root/'data');run=root/'run'
            summary=prepare(data,run,lambda chat:sum(len(m['content']) for m in chat)//3,4500,config(),shards=3)
            self.assertEqual(set(summary['items_by_shape']),set(shapes.SHAPES))
            self.assertEqual(summary['excluded_integrity'],1)
            self.assertGreater(summary['windowed_items'],0)
            original=list(iter_jsonl(run/'manifest.jsonl'))
            sharded=[r for f in sorted((run/'shards').glob('*.jsonl')) for r in iter_jsonl(f)]
            self.assertEqual(sorted(r['unit_id'] for r in original),sorted(r['unit_id'] for r in sharded))
            parts=reconcile(run,[])
            self.assertEqual(len(parts[1]),summary['items'])
            self.assertTrue(all(a['assessment']=='incomplete' for row in parts[1] for a in row['axes'].values()))
            # Prepared evidence edits must be detected even if output text hashes happen to match.
            with (run/'windows.jsonl').open('a') as f:f.write('\n')
            with self.assertRaisesRegex(ValueError,'artifact hash'):reconcile(run,[])

    def test_arquivo_index_hash_checked_before_event_creation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);data=build(root/'data');run=root/'run'
            p=data/'resources/arquivo.pt/records.jsonl'
            rows=list(iter_jsonl(p));rows[0]['text_sha256']='0'*64;write_jsonl(p,rows)
            summary=prepare(data,run,lambda chat:sum(len(m['content']) for m in chat)//3,6000,config(),with_controls=False)
            gaps=list(iter_jsonl(run/'gaps.jsonl'))
            self.assertTrue(any(g['gap']=='document_hash_mismatch' and g['path']=='documents/index.jsonl' for g in gaps))
            self.assertEqual(summary['excluded_integrity'],2)

    def test_classifier_identity_ignores_nonce_but_not_policy(self):
        a=config();b=config();b['boundary_salt']='another-run'
        self.assertEqual(classifier_hash(a),classifier_hash(b))
        b['prompts']['axes']['hacking']+=' changed'
        self.assertNotEqual(classifier_hash(a),classifier_hash(b))

if __name__=='__main__':unittest.main()

class AdditionalIntegrityTests(unittest.TestCase):
    def test_malformed_record_fields_are_gaps(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);data=build(p/'data')
            path=data/'resources/new.example/records.jsonl'
            with path.open('a') as out:
                for row in ({'observation_id':['bad'],'kind':'raw_paste'},
                            {'observation_id':'bad2','kind':{'bad':1}},
                            {'observation_id':'bad3','kind':'raw_paste','document_path':42}):
                    out.write(json.dumps(row)+'\n')
            summary=prepare(data,p/'run',lambda chat:sum(len(m['content']) for m in chat)//3,6000,config(),with_controls=False)
            self.assertEqual(summary['gaps']['unreadable_record'],3)

class ProfileContractTests(unittest.TestCase):
    def test_live_public_content_is_source_specific(self):
        self.assertEqual(shapes.shape_for('live_public_content', source='ntfy.sh'), 'message')
        self.assertEqual(shapes.shape_for('live_public_content', source='ntfy.sh/export-one'), 'message')
        self.assertEqual(shapes.shape_for('live_public_content', source='url.popcat.xyz'), 'text')
        self.assertEqual(shapes.shape_for('live_public_content'), 'text')

    def test_compressed_trailing_bytes_are_not_silently_lost(self):
        for payload in (zlib.compress(b'first text')+b'extra bytes',
                        gzip.compress(b'first text')+gzip.compress(b'second text')):
            views,gaps=decode.decode_url('https://a.example/'+base64.urlsafe_b64encode(payload).decode())
            self.assertTrue(gaps)
            self.assertFalse(any(v['kind']=='DECODED_B64' for v in views))
