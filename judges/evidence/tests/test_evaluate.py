import json
import tempfile
import unittest
from pathlib import Path
from evidence_judge import evaluate as e
from evidence_judge.common import write_jsonl
from evidence_judge.schemas import AXES


def prediction(h, label='no_evidence', score=None, iid=None):
    return {'item_id':iid or h,'text_sha256s':[h], 'config_sha256':'config-one',
            'axes':{a:{'assessment':label,'p_yes_max':score} for a in AXES}}


class EvaluationTests(unittest.TestCase):
    def test_positive_band_uses_positive_window_score(self):
        row=prediction('x','positive_uncalibrated',.99)
        row['axes']['hacking']['p_yes_positive_max']=.4
        self.assertEqual(e.predict([row],'hacking'),(True,.4))

    def test_tied_scores_never_split(self):
        self.assertIsNone(e.choose_threshold([(False,True,0.8),(True,True,0.8)],0.9))

    def test_unknown_states_not_negatives(self):
        p=prediction('x','new_invalid_state')
        self.assertEqual(e.predict([p],'hacking')[0],'undecided')

    def test_hash_with_multiple_event_contexts_is_ambiguous(self):
        rows=[prediction('x','no_evidence',iid='a'),prediction('x','positive_uncalibrated',iid='b')]
        self.assertEqual(e.predict(rows,'hacking')[0],'ambiguous')

    def evaluate(self, rows, key, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);write_jsonl(root/'results',rows);write_jsonl(root/'key',key)
            return e.evaluate(root/'results',root/'key',**kwargs)

    def test_cross_partition_group_rejected(self):
        key=[{'id':'1','document_sha256':'a','leakage_group':'g','evaluation_partition':'dev'},
             {'id':'2','document_sha256':'b','leakage_group':'g','evaluation_partition':'test'}]
        with self.assertRaisesRegex(ValueError,'leakage|partition'):
            self.evaluate([],key)

    def test_same_document_across_groups_rejected(self):
        key=[{'id':'1','document_sha256':'same','leakage_group':'g1','evaluation_partition':'dev'},
             {'id':'2','document_sha256':'same','leakage_group':'g2','evaluation_partition':'test'}]
        with self.assertRaisesRegex(ValueError,'leakage|partition'):
            self.evaluate([],key)

    def test_test_split_sealed_by_default(self):
        key=[{'id':'1','document_sha256':'a','leakage_group':'g','evaluation_partition':'test','overall_label':'positive'}]
        report,_=self.evaluate([prediction('a','positive_uncalibrated')],key)
        self.assertNotIn('test',report['axes']['overall'])

    def test_coverage_and_missing_axis_labels(self):
        key=[{'id':str(i),'document_sha256':str(i),'leakage_group':str(i),
              'evaluation_partition':'dev','overall_label':'positive'} for i in range(3)]
        report,_=self.evaluate([prediction('0','positive_uncalibrated'),prediction('1','insufficient_evidence')],key)
        dev=report['axes']['overall']['dev']
        self.assertEqual((dev['tp'],dev['missing'],dev['undecided']),(1,1,1))
        self.assertAlmostEqual(dev['coverage'],1/3)
        self.assertAlmostEqual(dev['end_to_end_positive_recall'],1/3)
        self.assertEqual(report['axes']['hacking']['dev']['unlabeled'],3)

    def test_unrecognized_partition_rejected(self):
        with self.assertRaisesRegex(ValueError,'partition'):
            self.evaluate([],[{'id':'a','document_sha256':'a','evaluation_partition':'contest'}])

    def test_confusion_and_projection(self):
        rates=e.rates(e.confusion([(True,True),(True,False),(False,False),(False,True)]))
        self.assertEqual(rates['tpr'],0.5);self.assertEqual(rates['fpr'],0.5)
        self.assertAlmostEqual(e.project(0.9,0.01,[0.001],100000)[0]['precision'],0.0009/0.01089)

    def test_perfect_bootstrap_is_not_claimed_certainty(self):
        report,_=self.evaluate([prediction('a','positive_uncalibrated')],[{
            'id':'a','document_sha256':'a','leakage_group':'a','evaluation_partition':'dev','overall_label':'positive'}])
        self.assertIn('interval_warnings',report['axes']['overall']['dev'])

if __name__=='__main__':unittest.main()
