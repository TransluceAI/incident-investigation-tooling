# Evaluation of the campaign-aware prompt

Each test below was pre-registered: the prediction and the pass rule were written down before any judge output was seen, and each test is scored strictly by that rule. The scorer for each test is in [`scripts/`](scripts/), and its docstring repeats the rules. The labeled documents are not in this repository because benchmark captures can contain credential strings and other private data.

## Setup

- Model: Qwen3.8-27B-FP8 on vLLM 0.31.0, thinking off, temperature 0, 160,000-token windows, one engine profile for every arm.
- Campaign judge: `classifier-judge prepare --prompt config/recall_prompt.txt --hints` over windows from `screening-judge prepare` at 0611e37. It sends one request per window.
- v5 screening judge: main at a060ef0. It sends one request per axis, so three per window.
- An item counts as flagged when its result is positive on any axis. Failed, undecided and missing items count as not flagged.

## Labeled sets

- **Findings benchmark**: 200 positives and 400 negatives. TR5 split it into a tune split (134 positives) and a check split (66 positives). Prompt tuning used the tune split only.
- **Fresh set** (TR3 to TR5): 21 positives from findings 152 to 169 and 76 negatives (29 scanner reports, 47 wiki pages). None of these documents is in the benchmark.
- **TR6 negatives**: 89 new negatives (25 scanner reports, 64 wiki pages). They come from the same seeded candidate stream as the fresh negatives, continued past the TR3 picks, and were reviewed the same way. Reviewers read each document in full against the screening rubrics, and an unsure document was excluded.

Labels were reviewed by Claude subagents, not by people.

## Results

| Test | Question | Result |
|---|---|---|
| TR2 | Does screening OR untuned campaign judge reach 190/200 recall with at most 8/400 false positives on the benchmark? | Failed. Recall passed; the campaign judge flagged too many ordinary scanner reports. |
| TR3 | Does screening OR campaign (coordination or hacking only) add recall on the fresh set without false alarms? | Passed. Recall went from 9/21 to 11/21 with 0/76 false positives. With all axes, the campaign judge flagged 14 of 29 scanner negatives. |
| TR4 | Does TR3 hold with v5 as the screening arm? | Passed. v5 alone: 8/21. Ensemble: 11/21. Both had 0/76 false positives. |
| TR5 | After one tuning round, is the tuned judge at most 2 false positives and at least 13/21 recall on the fresh set? | Passed: 15/21 and 0/76. Its third prediction, that tuning would halve check-split scanner false alarms, was refuted (39/46 down to 31/46). |
| TR6 | Does the precision gain hold on 89 negatives no judge had seen? | Passed. Tuned: 0/89. Untuned: 6/89, all on scanner reports. v5: 0/89. |

| Arm | Fresh recall | Fresh false positives | TR6 false positives |
|---|---|---|---|
| v5 screening | 8/21 | 0/76 | 0/89 |
| Untuned campaign judge | 16/21 | 14/76 | 6/89 |
| Tuned campaign judge | 15/21 | 0/76 | 0/89 |
| v5 OR tuned campaign judge | 16/21 | 0/76 | 0/89 |

Fresh-set recall per finding:

| Finding | Positives | Tuned | TR4 ensemble | v5 |
|---|---|---|---|---|
| 152 | 1 | 1 | 1 | 0 |
| 154 | 6 | 1 | 0 | 0 |
| 157 | 3 | 3 | 2 | 2 |
| 163 | 5 | 5 | 3 | 2 |
| 167 | 4 | 3 | 3 | 3 |
| 169 | 2 | 2 | 2 | 1 |

On the fresh set, the tuned judge was positive on agent activity for 15 items, on coordination for 4 and on hacking for 3, with no false positives on any axis.

## What tuning changed

On the tune split, the untuned prompt raised false alarms on 34 of 47 scanner negatives, all on agent activity, and on none of the 206 wiki negatives. The judge's reasons fell into two groups:

- Ordinary scans flagged because of the scanner's own browser behaviour, a feed or API submission, or the operator's `benchmark` source label.
- urlquery reports whose requested URL is itself the incident pattern, for example data-portal queries or `r.jina.ai` and `allorigins` relays.

Round 1 added one paragraph under the agent-activity question that rules out the first group. Scanner false alarms on the tune split fell to 14/47, with recall unchanged at 133/134. Tuning stopped there. The remaining 14 belong to the second group, and suppressing them would mean training the judge against confirmed positives. In TR6, five of the untuned judge's six reasons cite the operator label or API submission, the cause the paragraph addresses.

## Limits

- The positive set is small: 21 documents from 6 findings. The tuned judge misses 5 of the 6 documents from finding 154.
- The fresh set was scored three times (TR3 to TR5), and its TR3 false alarms were known when the paragraph was written. TR6 is a clean check of precision only. Recall has no clean retest, because no new positives were available.
- The TR6 negatives are the clearly clean ones: 25 scanner reports covering about 17 distinct targets.
- On the check split, 31 of 46 scanner negatives are still flagged. Many look like the incident pattern. Whether they are mislabeled needs human review.
- The campaign judge's windows came from the screening prepare at 0611e37. Main's v5 prepare shows the model fewer provenance fields, so windows built from main are not byte-identical to the tested ones. The tests in this directory pass against main.

## Scripts

- `compare.py`: classifier judge against the screening judge on a labeled key (TR1).
- `ensemble.py`: OR-ensembles and the matcher-alone baseline on the benchmark (TR2).
- `tr3.py`, `tr4.py`: restricted ensemble on the fresh set, against the 0611e37 screening judge and against v5.
- `tr5_dev.py`: one tuning round on a benchmark split, with the reason for every false alarm.
- `tr5.py`: the tuned judge on the fresh set.
- `tr6.py`: false positives of the tuned, untuned and v5 arms on negatives only.
