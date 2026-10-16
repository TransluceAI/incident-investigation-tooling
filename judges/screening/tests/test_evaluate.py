import tempfile
import unittest
from pathlib import Path
from screening_judge import evaluate as e
from screening_judge.common import write_jsonl
from screening_judge.schemas import AXES


def prediction(h, label='no_evidence', score=None, iid=None):
    return {'item_id':iid or h,'text_sha256s':[h], 'config_sha256':'config-one',
            'axes':{a:{'assessment':label,'p_yes_max':score} for a in AXES}}


class EvaluationTests(unittest.TestCase):
    def test_verified_positive_is_predicted_positive(self):
        self.assertIs(e.predict([prediction('x','positive')],'hacking'),True)

    def test_unknown_states_not_negatives(self):
        p=prediction('x','new_invalid_state')
        self.assertEqual(e.predict([p],'hacking'),'undecided')

    def test_hash_with_multiple_event_contexts_is_ambiguous(self):
        rows=[prediction('x','no_evidence',iid='a'),prediction('x','positive',iid='b')]
        self.assertEqual(e.predict(rows,'hacking'),'ambiguous')

    def evaluate(self, rows, key, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_jsonl(root/'results',rows);write_jsonl(root/'key',key)
            return e.evaluate(root/'results',root/'key',**kwargs)

    def test_related_cases_cannot_cross_partitions(self):
        cases = {
            'shared group': [('a', 'g'), ('b', 'g')],
            'shared document': [('same', 'g1'), ('same', 'g2')],
        }
        for reason, members in cases.items():
            with self.subTest(reason=reason):
                key = [{'id': str(i), 'document_sha256': doc, 'leakage_group': group,
                        'evaluation_partition': split}
                       for i, ((doc, group), split) in enumerate(zip(members, ('dev', 'test')))]
                with self.assertRaises(ValueError):
                    self.evaluate([], key)

    def test_test_split_sealed_by_default(self):
        key=[{'id':'1','document_sha256':'a','leakage_group':'g','evaluation_partition':'test','overall_label':'positive'}]
        report=self.evaluate([prediction('a','positive')],key)
        self.assertNotIn('test',report['axes']['overall'])

    def test_coverage_and_missing_axis_labels(self):
        key=[{'id':str(i),'document_sha256':str(i),'leakage_group':str(i),
              'evaluation_partition':'dev','overall_label':'positive'} for i in range(3)]
        report=self.evaluate([prediction('0','positive'),prediction('1','insufficient_evidence')],key)
        dev=report['axes']['overall']['dev']
        self.assertEqual((dev['tp'],dev['missing'],dev['undecided']),(1,1,1))
        self.assertAlmostEqual(dev['coverage'],1/3)
        self.assertAlmostEqual(dev['end_to_end_positive_recall'],1/3)
        self.assertEqual(report['axes']['hacking']['dev']['unlabeled'],3)
        for prose in ('notes','split_protocol','historical_holdout_independence'):
            self.assertNotIn(prose,report)

    def test_unrecognized_partition_rejected(self):
        with self.assertRaisesRegex(ValueError,'partition'):
            self.evaluate([],[{'id':'a','document_sha256':'a','evaluation_partition':'contest'}])


if __name__=='__main__':unittest.main()
