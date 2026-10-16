import json
import tempfile
import unittest
from pathlib import Path

from screening_judge.common import sha256_text, write_jsonl, iter_jsonl
from screening_judge.prepare import prepare
from screening_judge import evaluate
from screening_judge.report import reconcile
from test_pipeline import config
from test_evaluate import prediction


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.data = self.root/'data'
        self.data.mkdir()

    def inputs(self):
        body='Preserved public document.'
        (self.data/'positive-labelled-path.txt').write_text(body)
        rows=[{'id':f'positive-case-{i}','document_path':'positive-labelled-path.txt',
               'text_sha256':sha256_text(body),'metadata':{'source_family':'wiki','format':'text',
                 'answer':'SECRET_KEY_HINT','review_reason':'SECRET_REVIEW_HINT'},
               'overall_label':'SECRET_LABEL_HINT'} for i in range(2)]
        write_jsonl(self.data/'inputs.jsonl',rows)
        # Deliberately unreadable: preparation must never open the answer key.
        (self.data/'answer_key.jsonl').write_bytes(b'\xffINVALID_SECRET_ANSWER_KEY')
        return rows

    def run_prepare(self):
        return prepare(self.data,self.root/'run',lambda chat:sum(len(m['content']) for m in chat)//4,
                       6000,config(),with_controls=False,benchmark_manifest='inputs.jsonl')

    def test_neutral_inputs_and_full_document_without_label_leakage(self):
        rows=self.inputs();summary=self.run_prepare()
        self.assertEqual(summary['items'],2)
        requests=list(iter_jsonl(self.root/'run/requests.jsonl'))
        for request in requests:
            prompt='\n'.join(m['content'] for m in request['body']['messages'])
            self.assertIn('Preserved public document.',prompt)
            self.assertIn('wiki',prompt)
            for secret in ('SECRET_', 'positive-case', 'positive-labelled-path', 'answer_key'):
                self.assertNotIn(secret,prompt)
                self.assertNotIn(secret,request['custom_id'])
        results=reconcile(self.root/'run',[])[1]
        self.assertEqual({r['benchmark_case_id'] for r in results},{r['id'] for r in rows})

    def test_duplicate_ids_rejected(self):
        rows=self.inputs();write_jsonl(self.data/'inputs.jsonl',[rows[0],rows[0]])
        with self.assertRaisesRegex(ValueError,'Duplicate benchmark'):self.run_prepare()

    def test_hash_and_path_escape_fail_closed(self):
        rows=self.inputs();rows[0]['text_sha256']='0'*64
        rows[1]['document_path']='../outside.txt';(self.root/'outside.txt').write_text('outside')
        write_jsonl(self.data/'inputs.jsonl',rows)
        self.assertEqual(self.run_prepare()['excluded_integrity'],2)

    def test_shared_body_is_disambiguated_by_case_id(self):
        a=prediction('same','positive',iid='a');a['benchmark_case_id']='positive'
        b=prediction('same','no_evidence',iid='b');b['benchmark_case_id']='negative'
        by_hash=evaluate.predictions([a,b])
        self.assertIs(evaluate.case_prediction(by_hash,{'id':'positive','document_sha256':'same'},'overall'),True)
        self.assertIs(evaluate.case_prediction(by_hash,{'id':'negative','document_sha256':'same'},'overall'),False)
        self.assertIsNone(evaluate.case_prediction(by_hash,{'id':'unknown','document_sha256':'same'},'overall'))


if __name__ == '__main__':
    unittest.main()
