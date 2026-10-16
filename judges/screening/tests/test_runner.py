import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

from test_pipeline import config, result
from screening_judge.common import sha256_text

spec=importlib.util.spec_from_file_location('screening_runner',Path(__file__).resolve().parents[1]/'scripts/inference_runner.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)

class Tokenizer:
    def encode(self,text): return [ord(c)+100 for c in text]
    def decode(self,ids,**kw):return ''.join(chr(i-100) for i in ids if i>=100)
    def get_vocab(self):return {'<think>':1,'</think>':2}
    def apply_chat_template(self,chat,**kw):
        return self.encode(''.join(m['content'] for m in chat))+([1] if kw.get('enable_thinking') else [])


class RunnerTests(unittest.TestCase):
    def test_partial_logprob_candidates_never_imply_certainty(self):
        prefix='{"label":"'
        tokens=[prefix,'yes','"}']
        tables=[{i:NS(decoded_token=t,logprob=0)} for i,t in enumerate(tokens)]
        answer=NS(token_ids=[0,1,2],logprobs=tables)
        for logprobs in (None, [None] * len(tokens), tables):
            with self.subTest(logprobs=logprobs):
                answer.logprobs = logprobs
                self.assertIsNone(r.label_probabilities(answer))
        answer.logprobs = tables
        tables[1]={1:NS(decoded_token='yes',logprob=math.log(.6)),
                   3:NS(decoded_token='no',logprob=math.log(.3)),
                   4:NS(decoded_token='ins',logprob=math.log(.1))}
        self.assertAlmostEqual(r.label_probabilities(answer)['yes'],.6)

    def test_score_uses_top_level_label_after_quoted_rationale(self):
        text = json.dumps({'rationale': 'say "label":"yes"', 'label': 'no'})
        quoted = text.index('yes')
        actual = text.rindex('no')
        pieces = [text[:quoted], 'yes', text[quoted + 3:actual], 'no', text[actual + 2:]]
        tables = [{i: NS(decoded_token=piece, logprob=0)} for i, piece in enumerate(pieces)]
        for index, chosen, other, p_yes in ((1, 'yes', 'no', .9), (3, 'no', 'yes', .2)):
            probabilities = {'yes': p_yes, 'no': 1 - p_yes - .05, 'ins': .05}
            tables[index] = {token: NS(decoded_token=label, logprob=math.log(probabilities[label]))
                             for token, label in ((index, chosen), (10, other), (11, 'ins'))}
        answer = NS(token_ids=list(range(len(pieces))), logprobs=tables)
        self.assertAlmostEqual(r.label_probabilities(answer)['yes'], .2)

    def test_combined_label_boundary_is_unavailable(self):
        answer=NS(token_ids=[0],logprobs=[{0:NS(decoded_token='{"label":"yes"}',logprob=0)}])
        self.assertIsNone(r.label_probabilities(answer))

    def test_thinking_final_json_and_truncation(self):
        from jsonschema import Draft202012Validator
        tok=Tokenizer();validator=Draft202012Validator({'type':'object'})
        ids=tok.encode('candidate {"wrong":1}')+[2]+tok.encode('{"right":1}')
        answer=NS(token_ids=ids,text=tok.decode(ids),finish_reason='stop')
        parsed,error,fields=r.validate_answer(answer,[10],1,tok,validator,True)
        self.assertIsNone(error);self.assertEqual(parsed,{'right':1})
        self.assertEqual(fields['reasoning_tokens']+fields['final_tokens'],len(ids))
        answer.token_ids=tok.encode('{"wrong":1}')
        parsed,error,_=r.validate_answer(answer,[10],1,tok,validator,True)
        self.assertIsNone(parsed);self.assertIsNotNone(error)
        answer.token_ids=ids;answer.finish_reason='length'
        self.assertIsNotNone(r.validate_answer(answer,[10],1,tok,validator,True)[1])

    def test_unit_expansion_duplicates_and_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'input'
            row={'unit_id':'w','chunk_idx':0,'text':'hello','text_sha256':sha256_text('hello'),
                 'shape':'text','axes':['hacking','coordination']}
            path.write_text(json.dumps(row)+'\n')
            self.assertEqual([x['unit_id'] for x in r.units(path)],['w:hacking','w:coordination'])
            path.write_text((json.dumps(row)+'\n')*2)
            with self.assertRaisesRegex(ValueError,'Duplicate'):list(r.units(path))
            row['text']='tampered';path.write_text(json.dumps(row)+'\n')
            with self.assertRaisesRegex(ValueError,'hash'):list(r.units(path))

    def test_context_cap_rejects_instead_of_truncating(self):
        cfg=config();cfg['engine']['max_model_len']=1
        row={'unit_id':'w:hacking','axis':'hacking','shape':'text','text':'hello'}
        with self.assertRaisesRegex(ValueError,'Oversize'):
            list(r.batches([row],Tokenizer(),cfg,{}))

    def test_interrupted_part_ignored_and_identity_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);row={'unit_id':'w','run_identity':'a'}
            (p/'part-000000.jsonl').write_text(json.dumps(row)+'\n')
            (p/'part-000001.jsonl.tmp').write_text('{')
            self.assertEqual(list(r.committed(p,'a')),['w'])
            with self.assertRaisesRegex(ValueError,'Run mismatch'):r.committed(p,'b')

    def test_full_runner_resume_and_no_gpu_reload(self):
        tok=Tokenizer();cfg=config();cfg['engine']['max_model_len']=100000
        cfg['host_batch_tokens']=100000;cfg['host_batch_records']=10
        loads=[]
        class LLM:
            def __init__(self,**kw):loads.append(kw)
            def chat(self,chats,**kw):
                output=[]
                for i,(chat,sampling) in enumerate(zip(chats,kw['sampling_params'])):
                    props=sampling['structured_outputs']['json']['properties']
                    axis='hacking' if 'actions' in props else 'coordination' if 'mechanism' in props else 'agentic_activity'
                    text=json.dumps(result(axis))
                    answer=NS(token_ids=tok.encode(text),text=text,finish_reason='stop' if i!=1 else 'length')
                    output.append(NS(prompt_token_ids=tok.apply_chat_template(chat),outputs=[answer]))
                return output
        modules={'transformers':NS(AutoTokenizer=NS(from_pretrained=lambda *a,**kw:tok)),
                 'vllm':NS(LLM=LLM,SamplingParams=lambda **kw:kw),
                 'vllm.sampling_params':NS(StructuredOutputsParams=lambda **kw:kw),
                 'torch':NS(cuda=NS(device_count=lambda:1,get_device_name=lambda i:'fake'),__version__='fake',version=NS(cuda='fake'))}
        with tempfile.TemporaryDirectory() as tmp,patch.dict(sys.modules,modules),patch.object(r.signal,'signal'),contextlib.redirect_stdout(io.StringIO()):
            p=Path(tmp);(p/'config').write_text(json.dumps(cfg))
            row={'unit_id':'w','chunk_idx':0,'text':'hello','text_sha256':sha256_text('hello'),
                 'shape':'text','axes':list(r.prompts.AXES)}
            (p/'input').write_text(json.dumps(row)+'\n')
            argv=['runner','--config',str(p/'config'),'--input-manifest',str(p/'input'),
                  '--output-dir',str(p/'outputs'),'--worker-id','0']
            with patch.object(sys,'argv',argv):
                with self.assertRaises(SystemExit) as exc:r.main()
                self.assertEqual(exc.exception.code,2)
            part=p/'outputs/worker-0/part-000000.jsonl';before=part.read_bytes()
            rows=[json.loads(x) for x in before.splitlines()]
            self.assertEqual([x['status'] for x in rows],['success','failed','success'])
            with patch.object(sys,'argv',argv+['--resume']):
                with self.assertRaises(SystemExit):r.main()
                self.assertEqual(len(loads),1);self.assertEqual(part.read_bytes(),before)
                cfg['sampling']['temperature']=.4;(p/'config').write_text(json.dumps(cfg))
                with self.assertRaisesRegex(ValueError,'matching identity'):r.main()

if __name__=='__main__':unittest.main()
