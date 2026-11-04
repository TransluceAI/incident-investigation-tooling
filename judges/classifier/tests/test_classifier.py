import json
import tempfile
import unittest
from pathlib import Path

from classifier_judge import core
from screening_judge.common import iter_jsonl, request_sha256, sha256_file, sha256_text, write_jsonl
from screening_judge.evaluate import evaluate
from screening_judge.prepare import prepare as screening_prepare

ENGINE = {"model": "/path/to/model", "engine": {"seed": 17}, "sampling": {"temperature": 0, "max_tokens": 512},
          "chat_template_kwargs": {"enable_thinking": False}, "host_batch_records": 32,
          "host_batch_tokens": 400000}


def labels(agent="no", coord="no", hack="no"):
    return {"reason": "Test fixture.", "agentic_activity": agent, "coordination": coord, "hacking": hack}


def write_worker(run, rows, name="worker-0", manifest="requests.jsonl"):
    ident = {"config": json.loads((run / "engine.json").read_text()),
             "manifest_hash": sha256_file(run / manifest), "runner_hash": "test-runner", "max_records": None}
    digest = sha256_text(json.dumps(ident, sort_keys=True))
    worker = run / "outputs" / name
    worker.mkdir(parents=True)
    (worker / "identity.json").write_text(json.dumps({"identity": digest, **ident}))
    write_jsonl(worker / "part-000000.jsonl", [{**r, "run_identity": digest} for r in rows])
    return worker


class ClassifierTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        data = self.root / "data"
        (data / "resources/sample").mkdir(parents=True)
        recs = []
        for oid, body in (("plain", "ordinary text"), ("long", "agent notes " * 3000)):
            (data / f"{oid}.txt").write_text(body)
            recs.append({"observation_id": oid, "kind": "raw_paste", "document_path": f"{oid}.txt",
                         "text_sha256": sha256_text(body)})
        write_jsonl(data / "resources/sample/records.jsonl", recs)
        self.source = self.root / "screening"
        screening_prepare(data, self.source, lambda chat: sum(len(m["content"]) for m in chat), 18000, dict(ENGINE))
        self.run = self.root / "classifier"
        self.summary = core.prepare(self.source, self.run)
        self.requests = {r["custom_id"]: r for r in iter_jsonl(self.run / "requests.jsonl")}

    def answer(self, uid, result, status="success"):
        return {"unit_id": uid, "custom_id": uid, "request_sha256": request_sha256(self.requests[uid]["body"]),
                "status": status, "result": result, "error": None if status == "success" else "boom"}

    def results(self, outputs):
        report = core.report(self.run, [self.run / "outputs"], self.root / "report")
        return report, {r["item_id"]: r for r in iter_jsonl(self.root / "report/item-results.jsonl")}

    def test_one_request_per_screening_window_with_identical_evidence(self):
        windows = list(iter_jsonl(self.source / "windows.jsonl"))
        self.assertEqual(self.summary["requests"], len(windows))
        self.assertEqual(self.summary["source_requests"], 3 * len(windows))
        salt = json.loads((self.source / "config.json").read_text())["boundary_salt"]
        for w in windows:
            user = self.requests[w["window_id"] + ":classify"]["body"]["messages"][1]["content"]
            self.assertIn(f"<evidence-{salt}>\n{w['text']}\n</evidence-{salt}>", user)
        per_item = [w["item_id"] for w in windows]
        self.assertTrue(any(per_item.count(i) > 1 for i in per_item), "fixture needs a multi-window item")

    def test_any_window_yes_is_positive_and_failures_are_incomplete(self):
        rows = []
        for uid in self.requests:
            item = uid.split(":")[0]
            if uid.endswith(":w0:classify") and "control" not in uid:
                rows.append(self.answer(uid, labels(agent="yes")))
            elif uid.endswith(":w1:classify"):
                rows.append(self.answer(uid, None, status="failed"))
            else:
                rows.append(self.answer(uid, labels()))
        write_worker(self.run, rows)
        summary, results = self.results(rows)
        multi = [r for r in results.values() if r["windows"] > 1]
        self.assertTrue(multi)
        for r in results.values():
            self.assertEqual(r["axes"]["agentic_activity"]["assessment"], "positive")
            expected = "incomplete" if r["windows"] > 1 else "no_evidence"
            self.assertEqual(r["axes"]["hacking"]["assessment"], expected)
        self.assertGreater(summary["controls"]["total"], 0)
        self.assertEqual(summary["failed"], sum(u.endswith(":w1:classify") for u in self.requests))

    def test_schema_invalid_answer_is_a_failure_not_a_negative(self):
        rows = [self.answer(uid, {"reason": "x", "agentic_activity": "maybe", "coordination": "no", "hacking": "no"})
                for uid in self.requests]
        write_worker(self.run, rows)
        summary, results = self.results(rows)
        self.assertEqual(summary["validated"], 0)
        self.assertTrue(all(a["assessment"] == "incomplete" for r in results.values() for a in r["axes"].values()))

    def test_output_for_a_different_request_is_rejected(self):
        uid = next(iter(self.requests))
        bad = {**self.answer(uid, labels()), "request_sha256": "0" * 64}
        write_worker(self.run, [bad])
        summary, _ = self.results([bad])
        self.assertEqual(summary["validated"], 0)

    def test_screening_evaluate_scores_classifier_results(self):
        write_worker(self.run, [self.answer(uid, labels(coord="yes")) for uid in self.requests])
        self.results(None)
        item = next(iter_jsonl(self.root / "report/item-results.jsonl"))
        key = self.root / "key.jsonl"
        write_jsonl(key, [{"id": "case-1", "document_sha256": item["text_sha256s"][0], "overall_label": "positive",
                           "evaluation_partition": "dev"}])
        report = evaluate(self.root / "report/item-results.jsonl", key)
        self.assertEqual(report["axes"]["overall"]["dev"]["tp"], 1)

    def test_runner_cost_sums_worker_timings(self):
        worker = write_worker(self.run, [])
        write_jsonl(worker / "timings.jsonl", [{"batch_seconds": 2.0, "batch_input_tokens": 10},
                                               {"batch_seconds": 3.0, "batch_input_tokens": 5}])
        (worker / "summary.json").write_text(json.dumps({"output_tokens": 7, "session_seconds": 9}))
        cost = core.runner_cost([self.run / "outputs"])
        self.assertEqual((cost["gpu_seconds"], cost["input_tokens"], cost["output_tokens"]), (5.0, 15, 7))

    def test_source_run_change_is_detected(self):
        write_worker(self.run, [])
        with open(self.source / "windows.jsonl", "a") as f:
            f.write("\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            core.report(self.run, [self.run / "outputs"], self.root / "report")


if __name__ == "__main__":
    unittest.main()


class CompareTests(unittest.TestCase):
    def test_rules_count_undecided_as_not_flagged(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import compare
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        key = [{"id": f"case-{i}", "document_sha256": f"{i:064x}", "overall_label": lab, "leakage_group": f"g{i % 3}",
                "evaluation_partition": "development", "source_url": "https://collusion.wiki/x"}
               for i, lab in enumerate(["positive"] * 4 + ["negative"] * 4)]
        write_jsonl(root / "key.jsonl", key)

        def item(i, assessment):
            other = {"assessment": "no_evidence", "injection_detected": False}
            return {"item_id": f"i{i}", "text_sha256s": [f"{i:064x}"],
                    "axes": {"hacking": other, "coordination": other,
                             "agentic_activity": {"assessment": assessment, "injection_detected": False}}}
        # Screening: 3 of 4 positives flagged, one incomplete; 1 false positive.
        write_jsonl(root / "s.jsonl", [item(0, "positive"), item(1, "positive"), item(2, "positive"),
                                       item(3, "incomplete"), item(4, "positive")]
                    + [item(i, "no_evidence") for i in (5, 6, 7)])
        # Classifier: 4 of 4 positives, no false positives.
        write_jsonl(root / "c.jsonl", [item(i, "positive") for i in range(4)] + [item(i, "no_evidence") for i in range(4, 8)])
        for name, out, gpu in (("s", 30, 10.0), ("c", 10, 5.0)):
            worker = root / name / "worker-0"
            worker.mkdir(parents=True)
            (worker / "summary.json").write_text(json.dumps({"output_tokens": out}))
            write_jsonl(worker / "timings.jsonl", [{"batch_seconds": gpu}])
        report = compare.compare(root / "key.jsonl", None,
                                 {"screening": root / "s.jsonl", "classifier": root / "c.jsonl"},
                                 {"screening": [root / "s"], "classifier": [root / "c"]})
        v = report["verdict"]
        self.assertEqual((v["P1"]["screening_tpr"], v["P1"]["classifier_tpr"]), (0.75, 1.0))
        self.assertEqual((v["P2"]["screening_fpr"], v["P2"]["classifier_fpr"]), (0.25, 0.0))
        self.assertEqual(v["P3"]["output_token_ratio"], 1 / 3)
        self.assertEqual(v["P3"]["gpu_seconds_ratio"], 0.5)
        self.assertTrue(v["P3"]["pass"] and v["TR1_pass"])
        self.assertEqual(report["status_counts"]["screening"]["overall"]["undecided"], 1)

    def test_wilson_interval_and_small_axes_are_not_scored(self):
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
        import compare
        w = compare.wilson(8, 10)
        self.assertAlmostEqual(w["wilson95"][0], 0.4902, places=3)
        self.assertAlmostEqual(w["wilson95"][1], 0.9433, places=3)
        report = {"rates": {j: {"overall": {"tpr": {"rate": 1.0}, "fpr": {"rate": 0.0}}} for j in compare.JUDGES},
                  "cost": {j: {"output_tokens": 1, "gpu_seconds": 1} for j in compare.JUDGES},
                  "hand_labeled": {"hacking": {"yes_labels": 12, "screening": {"recall": {"rate": 1.0}},
                                               "classifier": {"recall": {"rate": 0.0}}}}}
        self.assertIsNone(compare.verdict(report)["P4"]["pass"])


class HintTests(unittest.TestCase):
    def test_matcher_finds_campaign_signatures_and_not_plain_text(self):
        from classifier_judge import hints
        text = ("Feb19 cohort: R3 due task clock 06:22. Please relay R4 here. Beacon at counterapi.dev/x. "
                "GET https://www.ojjdp.gov/ezapop/..%2Fezaucr/ and https://httpbingo.org/base64/abc")
        names = [n for n, _ in hints.find(text)]
        for want in ("benchmark_cohort", "round_relay", "signal_channel", "carrier_echo", "traversal"):
            self.assertIn(want, names)
        self.assertEqual(hints.find("A biography of a cultural entrepreneur and his projects."), [])
        self.assertIn("none", hints.render([]))

    def test_prepare_with_hints_puts_them_after_the_evidence(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        data = root / "data"
        (data / "resources/sample").mkdir(parents=True)
        body = "Our cohort R2 arrived; please relay R3 via ntfy.sh/topic"
        (data / "a.txt").write_text(body)
        write_jsonl(data / "resources/sample/records.jsonl",
                    [{"observation_id": "a", "kind": "raw_paste", "document_path": "a.txt", "text_sha256": sha256_text(body)}])
        screening_prepare(data, root / "s", lambda chat: sum(len(m["content"]) for m in chat), 18000, dict(ENGINE),
                          with_controls=False)
        summary = core.prepare(root / "s", root / "c", with_hints=True)
        self.assertIn("hints.jsonl", summary["artifact_hashes"])
        row = next(iter_jsonl(root / "c/hints.jsonl"))
        self.assertIn("signal_channel", row["hints"])
        user = next(iter_jsonl(root / "c/requests.jsonl"))["body"]["messages"][1]["content"]
        closing = user.index("</evidence-")
        self.assertGreater(user.index("PATTERN HINTS"), closing)
        cfg = json.loads((root / "c/config.json").read_text())
        self.assertTrue(cfg["hints"] and cfg["hints_sha256"] and cfg["prompt_sha256"])
