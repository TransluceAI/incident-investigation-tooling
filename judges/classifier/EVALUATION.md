# Evaluation of the campaign-aware prompt

We wrote the prediction and the pass rule for each test before we saw any judge output. We then scored each test only by its rule. The scorer for each test is in [`scripts/`](scripts/), and its docstring gives the rules again. This repository does not include the labeled documents, because benchmark captures can contain credential strings and other private data.

## Setup

- Model: Qwen3.8-27B-FP8 on vLLM 0.31.0, with thinking off, temperature 0 and 160,000-token windows. All arms use the same engine profile.
- Campaign judge: `classifier-judge prepare --prompt config/recall_prompt.txt --hints` over windows from `screening-judge prepare` at 0611e37. The judge sends 1 request per window.
- v5 screening judge: main at a060ef0. The judge sends 1 request per axis, so 3 requests per window.
- Scoring: an item is flagged when its result is positive on any axis. A failed, undecided or missing item counts as not flagged.

## Labeled sets

- Findings benchmark: 200 positives and 400 negatives. TR5 divided it into a tune split (134 positives) and a check split (66 positives). The prompt tuning used only the tune split.
- Fresh set, used in TR3 to TR5: 21 positives from findings 152 to 169, and 76 negatives (29 scanner reports and 47 wiki pages). The benchmark does not contain these documents.
- TR6 negatives: 89 new negatives (25 scanner reports and 64 wiki pages). They continue the seeded candidate stream of the fresh negatives, after the TR3 picks. The reviewers used the same procedure for both sets.

Claude subagents reviewed all labels. No person reviewed them. Each reviewer read the full document against the screening rubrics and excluded a document when unsure.

## Results

| Test | Question | Result |
|---|---|---|
| TR2 | Does screening OR the untuned campaign judge get 190/200 recall and at most 8/400 false positives on the benchmark? | Failed. Recall passed, but the campaign judge flagged too many ordinary scanner reports. |
| TR3 | Does screening OR the campaign judge (coordination or hacking only) add recall on the fresh set with no false alarms? | Passed. Recall went from 9/21 to 11/21, with 0/76 false positives. On all axes, the campaign judge flagged 14 of 29 scanner negatives. |
| TR4 | Does the TR3 result hold when v5 is the screening arm? | Passed. v5 alone found 8/21 and the ensemble found 11/21, both with 0/76 false positives. |
| TR5 | After 1 tuning round, does the tuned judge get at most 2 false positives and at least 13/21 recall on the fresh set? | Passed with 15/21 and 0/76. Its third prediction failed. It said that tuning would halve the scanner false alarms on the check split, but they went only from 39/46 to 31/46. |
| TR6 | Does the precision gain hold on 89 negatives that no judge saw before? | Passed. The tuned judge flagged 0/89, the untuned judge 6/89 (all scanner reports) and v5 0/89. |

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

On the fresh set, the tuned judge was positive on agent activity for 15 items, on coordination for 4 items and on hacking for 3 items. It had no false positives on any axis.

## What tuning changed

On the tune split, the untuned prompt flagged 34 of 47 scanner negatives, all on agent activity. It flagged none of the 206 wiki negatives. The judge's reasons fell into 2 groups:

- Ordinary scans. The judge flagged them because of the scanner's own browser behaviour, a feed or API submission, or our `benchmark` source label.
- urlquery reports where the requested URL itself matches the incident pattern. Examples are data-portal queries and `r.jina.ai` or `allorigins` relays.

Round 1 added 1 paragraph to the agent-activity question. The paragraph rules out the first group. Scanner false alarms on the tune split fell to 14/47, and recall stayed at 133/134. We stopped after round 1. The remaining 14 belong to the second group, so more tuning would teach the judge against confirmed positives. In TR6, 5 of the untuned judge's 6 reasons cite our source label or API submission, and the paragraph forbids both.

## Limits

- The positive set is small, with 21 documents from 6 findings. The tuned judge misses 5 of the 6 documents from finding 154.
- Recall has no clean retest.
  - TR3 to TR5 scored the fresh set 3 times, and we knew its TR3 false alarms when we wrote the paragraph.
  - No new positives were available, so TR6 tests precision only.
- The hard negatives are still open.
  - The TR6 negatives are the clearly clean ones. The 25 scanner reports cover about 17 distinct targets.
  - On the check split, the tuned judge still flags 31 of 46 scanner negatives. Many of them look like the incident pattern, so a person must review their labels.
- The tested windows came from the screening prepare at 0611e37. Main's v5 prepare shows the model fewer provenance fields, so its windows differ from the tested windows. The tests in this directory pass against main.

## Scripts

- Benchmark scorers:
  - `compare.py` scores the classifier judge against the screening judge on a labeled key (TR1).
  - `ensemble.py` scores OR-ensembles and the matcher-only baseline (TR2).
  - `tr5_dev.py` scores 1 tuning round on a benchmark split and writes the reason for each false alarm.
- Fresh-set scorers: `tr3.py` and `tr4.py` score the restricted ensemble against the 0611e37 screening judge and against v5. `tr5.py` scores the tuned judge.
- Negatives-only scorer: `tr6.py` counts the false positives of the tuned, untuned and v5 arms.
