# Local continuation — 2026-10-16

Recovered Claude session `d56e3581-40f3-465d-8908-d1edbc8d5585` and the supplied pasted transcript. The task is local iteration on the reusable three-axis Qwen 27B judge, backed by concrete correctness evidence or verified expert guidance. No PR, commit, cluster job, server upload, Findings API mutation, or external model inference was performed.

## Evidence inspected

- Existing `judges/evidence/` implementation and the interrupted `tests/fixtures.py`.
- Workspace `cluster-all-data-profile.json`, `cluster-source-semantics-profile.json`, `cluster-benchmark-coverage-profile.json`, `cluster-date-and-benchmark-summary.json`, and `cluster-profile-all-data.py`.
- Cached `evidence-judge-work/findings/findings.json`: 115 findings, latest recorded update 2026-10-15T20:14:10Z. This is context, not a new gold label set.
- `reports/Effective LLM judge practices.md`; important underlying sources were re-opened rather than accepting every claim in the prior narrative.

## Reproduced and corrected defects

| Area | Defect and correction | Regression evidence |
| --- | --- | --- |
| Discovery | Fixed-depth globs missed deep accepted exports and imported historical raw files. Match the profiler's accepted-record scope. | Recursive DIRECT export and raw-copy exclusion |
| Evidence preservation | Nested/list metadata and arbitrary scanner payload fields called `hash`/`_id` disappeared. Preserve them. | Exact retained scalar/pointer tests |
| Grouping | Equal bodies with different shapes/provenance merged; missing capture times could merge events. Preserve evidence context. | Same-body capture and shape tests |
| Integrity | Missing bodies aborted preparation, bad hashes vanished from reports, archive indexes influenced events before hash verification. Keep incomplete items and verify index/redirect content. | Missing/bad document and index tests |
| Token bounds | Largest policy by character count was assumed largest by token count. Measure every axis. | Deliberately adverse per-axis counter |
| Reporting | Outputs from another configuration were accepted; edited preparation artifacts were unchecked. Bind artifacts, manifests, configs, and worker identities. | Foreign config and tampered-window tests |
| Citations | Corrupt citation maps escaped as uncaught exceptions. Normalize validation errors; require cited action segments. | Unicode window coverage and invalid-citation routing |
| Controls | Abstention passed negative controls; no scanner controls existed; semantic control names appeared in provenance. Correct all three. | Abstention and five-shape control tests |
| Scores | Missing top-k alternatives implied zero mass/confidence. Withhold incomplete scores and label retained values diagnostic. | Complete/partial/null logprob and boundary tests |
| Calibration | Tied scores split unrealistically; related cases could cross splits; test metrics appeared automatically. Preserve ties, connect leakage groups/hash duplicates, and seal test metrics by default. | Threshold, leakage, and sealed-test tests |
| Evaluation coverage | Unknown states became negatives; shared bodies were max-pooled across distinct events. Keep undecided/ambiguous states; report full positive denominator. | Ambiguous join and coverage tests |
| Compression | Incomplete compressed streams could become apparently complete text; encoded code-key payloads were missed. Reject incomplete streams and decode code-key values inertly. | Base64 path/query, zlib/gzip, cap and truncation tests |
| Runtime | Interrupted framework lacked its tests. Cover final-JSON extraction, resume identity, failed commits, and token caps with a simulated engine. | Runner integration tests; no actual GPU claim |
| Packaging | Installed package lacked prompt assets. Bundle policies/config and verify loading outside the checkout. | Built wheel, extracted installation smoke test |

## Verified guidance and limits

- [Hamel Husain: Using LLM-as-a-Judge](https://hamel.dev/blog/posts/llm-judge/) supports expert-labeled validation, error analysis, and measuring true/false positives. It does not establish accuracy for this corpus or model.
- [scikit-learn: GroupShuffleSplit](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupShuffleSplit.html) supports separating related groups. The connected group/hash implementation here is our conservative application of that principle.
- [vLLM SamplingParams](https://docs.vllm.ai/en/v0.14.1/api/vllm/sampling_params/) documents top-token logprob output and inclusion of the sampled token. It does not provide a calibrated three-class probability. The stricter missing-score behavior follows from that API contract.
- [Qwen3.8-27B-FP8 model card](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) recommends different sampling settings for thinking and instruct modes. Two additional configs encode those recommendations for later dev comparison. No claim is made that they improve this judge before measurement.

All sources above were opened on 2026-10-16. Correctness fixes were tested with explicit counterexamples. The substantive three-axis rubrics were retained; no unmeasured prompt change is described as an accuracy gain. Scanner preservation and conservative deduplication trade additional tokens/calls for avoiding demonstrable evidence loss.

## Verified local results

- 43 distinct CPU regression/integration tests pass. The run log is saved at workspace `evidence-judge-work/verification-20261016/tests.log`.
- CLI dry preparation produced 25 items (10 corpus fixtures plus 15 controls), 32 windows and 96 axis requests across three disjoint shards. All five evidence shapes were represented; the deliberately corrupt document was retained as incomplete.
- Reporting without inference preserved all 96 requests as pending and all 10 corpus fixture items as incomplete. It produced zero validated judgments and did not claim release readiness.
- A wheel was built with cached setuptools and its bundled policies loaded from an extracted installation outside the checkout. The local `uv build` launcher crashed in macOS system-configuration initialization; the direct setuptools build succeeded without network access.
- The final audit also aligns threshold fitting with positive-window scores: a high diagnostic score on a negative window cannot inflate the band of a different positive window.

## Further continuation — source contracts and benchmark inputs

The next history search recovered session `879a01b3-2d0d-45e0-afcc-3676146d9c54`, which locates the benchmark only on the cluster under `/data/incident-investigation/data/judging/findings-benchmark-20261003-v2`. Its copied README explicitly calls the 600 cases a development set. The evaluator now states that historical holdout independence is unverified; creating partitions cannot undo earlier exposure.

The local `cluster-data-profile.json` records input fields `id`, `document_path`, `text_sha256`, and `metadata`. A direct neutral-manifest adapter now supports this contract without opening the answer key. Tests establish that paths, case IDs, label fields and review notes do not enter prompts, full document text is retained, invalid paths/hashes fail closed, duplicate IDs fail, and evaluation joins same-body cases by ID.

The source-semantics profile also shows `live_public_content` in two sources: 153 ntfy.sh observations and 1,349 url.popcat.xyz observations. Notification-message guidance now applies only to the ntfy.sh source; other sources retain the general-text fallback.

Compressed streams with trailing bytes or concatenated gzip members now produce explicit decode gaps rather than silently presenting only the first member. This follows the documented distinction between [zlib unused_data, unconsumed_tail and eof](https://docs.python.org/3/library/zlib.html), verified on 2026-10-16. Original source text remains available.

Preparation implementation hashes and inference-code fingerprints now bind reusable score bands. This closes a gap where an adapter or runner edit could reuse a band under identical JSON settings. It still does not fingerprint model weight bytes or GPU dependency versions.

**Latest local evidence:** 51 distinct tests pass. The direct benchmark CLI smoke test produced 16 items (one synthetic benchmark case and 15 controls), 48 requests, zero validated judgments and 48 pending requests. Logs and artifacts are in workspace `evidence-judge-work/verification-20261016-v2/`. No actual benchmark/model execution occurred.

## Completion audit and outstanding work

The local framework, regression suite, packaged policies, deterministic sharding, and calibration/reporting code are present. The full user objective is **not yet proven complete**:

1. No local 600-case answer key, benchmark input documents, or Qwen model/tokenizer was found in the workspace, Downloads, Documents, or Hugging Face cache search. A path request is pending. No cluster access or copy was attempted.
2. The cached benchmark profile reports 400 negative cases with axis labels and 200 positive cases without axis labels. Overall detection can be measured once inputs/results are available; each individual axis needs expert-adjudicated positive labels.
3. Actual model inference, schema/backend compatibility, injection canaries, repeated-run disagreement, full-corpus memory/throughput, and held-out error rates remain unmeasured.
4. Diagnostic token scores and exploratory dev bands are not probability calibration. A recall-floor operating point requires validated scores/labels and an explicit operating target; the current code does not claim to supply one.
5. Cross-document coordination and a reduce pass were deferred in the recovered design. The current judge assesses individual prepared items/windows. A verified citation checks existence and offsets, not semantic support or provenance truth.
6. The exact supplied weights snapshot must be verified independently. Hashes currently cover prepared evidence/configuration and worker code identity, not all model weight bytes or the installed GPU runtime.

Retain the goal as active. The next quality-validation step is to obtain the local benchmark/model paths, freeze a development run, inspect its error families and per-axis label gaps, then assess a frozen candidate on a holdout whose prior exposure has been checked. CPU success must not be substituted for that evidence.

## Recovered-goal continuation — worker assignment and citation numbers

Session `d56e3581-40f3-465d-8908-d1edbc8d5585` (2026-10-16) contains the original instruction to iterate locally on the Qwen 27B judge, make only demonstrated or verified improvements, and create no PRs or server uploads. The current goal was restored from that instruction.

Two additional defects were reproduced before correction:

- A worker identity referencing an authentic shard could submit another shard's unit, and a worker limited to the first manifest window could submit a later window. Both were accepted because reconciliation checked only run-wide membership. Reporting now checks the individual worker's selected manifest before accepting a committed unit. Valid assigned output remains accepted and other units stay pending.
- An evidence citation with JSON `chunk_id: 1.0` passed integer schema validation, then crashed on Python list indexing. After schema validation, integral chunk numbers are normalized to Python integers. Booleans, fractional numbers, strings and zero still fail validation.

**Latest local evidence:** all 55 tests pass, including four new regression tests for these cases. An updated wheel was built without network access; an extracted installation outside the checkout loaded all policies and resolved the integral-number citation. Build output, the installed-package smoke script/result, source hashes and the verification record are in workspace `evidence-judge-work/verification-20261016-v3/`.

The local search still found only the synthetic benchmark fixture, with no real benchmark input/key or Qwen snapshot. The path request remains pending. Model inference, benchmark accuracy, calibration and GPU validation remain outstanding; the goal is unfinished and model validation is blocked on those inputs.

## Upstream GitHub review — confidence scoring

Checked GitHub on 2026-10-16, using isolated temporary snapshots rather than updating the working branch:

- Upstream main is `d85f43918e8c53aa05c912cb1486cbea201975ee`. Fran (`f-pernice`) merged the Arquivo PR after commit [`18e636c`](https://github.com/TransluceAI/incident-investigation-tooling/commit/18e636c7d1cb95606f5ea7ff9c564650645ca3ad), “Add hacking confidence scores and rank the review queue.” The commit's recorded author is `pbill-2021`; [Fran's comment](https://github.com/TransluceAI/incident-investigation-tooling/pull/2#issuecomment-6180126084) confirms the change follows Ilya's request.
- New Arquivo judgments require `hacking_attempt.confidence`, a model-generated number from 0 to 1 indicating support for a hacking attempt. Positive windows and schema-valid unverified positives are sorted together by descending `hacking_confidence`, with validation status retained. Negative windows do not enter that queue even when their score is high.
- Historical runs use their saved schema, retain null scores, and require fresh inference to obtain confidence values. Re-reporting does not generate scores. All four upstream confidence tests passed locally.
- [Open PR #3](https://github.com/TransluceAI/incident-investigation-tooling/pull/3), head `6bdecabf383147b21253c41e61ce025f3b138133`, preserves confidence scoring and adds report identity, unreadable-input, context-budget and shard-audit fixes. Its 16 tests passed locally. Its stated remaining questions include hacking-only queues, provider-attribution citations, multi-window context, action references and decoding gaps.

At inspection time the working branch started at `e3ee920`, before these upstream changes. The local reusable evidence judge has a different schema: independent axis labels, verified-positive review queues, and optional first-token branch scores. Those diagnostic scores are different from the upstream model-generated confidence value. Neither has demonstrated calibration on this corpus.

The upstream confidence field is a candidate ranking feature, not evidence of improved accuracy. Preserve it in any future Arquivo integration, keep its semantics distinct from token scores, and compare ranking quality on development labels before choosing either for the reusable judge. Fran also explicitly leaves schema simplification for comparative experiments. No upstream code, schema or rubric was merged into the working tree during this inspection; no model inference, PR or server upload was performed.

## Rebase onto upstream main

After the GitHub inspection, the user requested updating the local branch. `add-arquivo-judge` was rebased onto `origin/main` at `d85f43918e8c53aa05c912cb1486cbea201975ee`, bringing in the merged Arquivo confidence schema, rubric, ranked queue and four confidence tests. Local uncommitted reasoning-runner and reusable-judge work was backed up, stashed and restored. One conflict in `judges/arquivo/docs/run-results.md` was resolved by retaining both the upstream confidence-history explanation and the local reasoning-run notes.

Post-rebase validation passes: 21 Arquivo tests (including the four upstream confidence tests) and 55 reusable evidence-judge tests. The active branch and fetched upstream main have no commit divergence; local edits remain uncommitted. PR #3 remains separate and was not merged. Backup files and the retained stash preserve the pre-rebase work. No push, PR, server upload or model inference was performed.
