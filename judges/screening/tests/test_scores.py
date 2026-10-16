import json
import math
import unittest

from screening_judge.scores import label_probabilities


def step(token, candidates=None):
    """One runner logprobs entry; candidates default to the sampled token alone."""
    candidates = candidates or {token: 1.0}
    return {'token': token, 'logprob': math.log(candidates.get(token, 1.0)),
            'top_logprobs': [{'token': t, 'logprob': math.log(p)} for t, p in candidates.items()]}


class ScoreTests(unittest.TestCase):
    def test_partial_candidates_never_imply_certainty(self):
        content = [step('{"label":"'), step('yes'), step('"}')]
        for logprobs in (None, {}, {'content': []}, {'content': content}):
            with self.subTest(logprobs=logprobs):
                self.assertIsNone(label_probabilities(logprobs))
        content[1] = step('yes', {'yes': .6, 'no': .3, 'ins': .1})
        self.assertAlmostEqual(label_probabilities({'content': content})['yes'], .6)

    def test_missing_or_invalid_values_are_unavailable(self):
        branch = step('yes', {'yes': .6, 'no': .3, 'ins': .1})
        for broken in ({**step('{"label":"'), 'token': None},
                       {**branch, 'top_logprobs': branch['top_logprobs'][:2]}):
            content = [step('{"label":"'), branch, step('"}')]
            content[0 if broken['token'] is None else 1] = broken
            with self.subTest(broken=broken):
                self.assertIsNone(label_probabilities({'content': content}))
        null = {**branch, 'top_logprobs': [{'token': 'yes', 'logprob': None}, *branch['top_logprobs'][1:]]}
        self.assertIsNone(label_probabilities({'content': [step('{"label":"'), null, step('"}')]}))

    def test_score_uses_top_level_label_after_quoted_rationale(self):
        text = json.dumps({'rationale': 'say "label":"yes"', 'label': 'no'})
        quoted, actual = text.index('yes'), text.rindex('no')
        content = [step(text[:quoted]), step('yes', {'yes': .9, 'no': .05, 'ins': .05}),
                   step(text[quoted + 3:actual]), step('no', {'no': .75, 'yes': .2, 'ins': .05}),
                   step(text[actual + 2:])]
        self.assertAlmostEqual(label_probabilities({'content': content})['yes'], .2)

    def test_combined_label_boundary_is_unavailable(self):
        self.assertIsNone(label_probabilities({'content': [step('{"label":"yes"}')]}))


if __name__ == '__main__':
    unittest.main()
