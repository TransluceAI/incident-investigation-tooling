"""Command line: prepare, report, escalate, cascade, evaluate, compare."""
import argparse
import json
from pathlib import Path

from . import cascade
from . import evaluate as ev
from .prepare import prepare
from .report import reconcile, write_report


def tokenizer_counter(config):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(config["model"], local_files_only=True, trust_remote_code=False)

    def count(chat):
        return len(tok.apply_chat_template(chat, tokenize=True, return_dict=False, add_generation_prompt=True,
                                           **config.get("chat_template_kwargs", {})))
    return count


def char_counter(chars_per_token):
    """Dry-run estimate only. The runner re-counts exactly and rejects oversize requests."""
    if chars_per_token <= 0:
        raise ValueError("CHARS_PER_TOKEN must be positive")
    def count(chat):
        return sum(len(m["content"]) for m in chat) // chars_per_token + 16
    return count


def main():
    p = argparse.ArgumentParser(prog="screening-judge")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare", help="Build items, windows and inference requests")
    a.add_argument("--data-root", required=True)
    a.add_argument("--benchmark-manifest", help="Neutral inputs.jsonl relative to data root; never reads answer_key.jsonl")
    a.add_argument("--engine", required=True,
                   help="Resolved inference/qwen runner profile (local model path, engine and sampling)")
    a.add_argument("--label-logprobs", type=int, default=10, metavar="K",
                   help="Logprob candidates per token for label scores (0 disables; default 10)")
    a.add_argument("--no-citation-constraints", action="store_true",
                   help="Send the plain axis schema instead of one narrowed to each window's citable chunks")
    a.add_argument("--out", required=True)
    a.add_argument("--window-tokens", type=int, default=160_000)
    a.add_argument("--estimate-tokens", type=int, metavar="CHARS_PER_TOKEN",
                   help="Dry run without the tokenizer; the runner still enforces exact limits")
    a.add_argument("--no-controls", action="store_true")
    a.add_argument("--shards", type=int, default=1)
    b = sub.add_parser("report", help="Validate outputs and write item results and review queues")
    b.add_argument("--run", required=True)
    b.add_argument("--outputs", nargs="+", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--thresholds", help="thresholds.json from evaluate (dev split only)")
    b.add_argument("--near-miss", type=float, default=0.2)
    e = sub.add_parser("escalate", help="Select pass-1 requests for a reasoning pass and write that run")
    e.add_argument("--run", required=True, help="Pass-1 prepared run")
    e.add_argument("--report", required=True, help="Pass-1 report directory")
    e.add_argument("--engine", required=True, help="Runner profile for pass 2 (reasoning on)")
    e.add_argument("--out", required=True)
    e.add_argument("--borderline", type=float, nargs=2, default=(0.2, 0.8), metavar=("LOW", "HIGH"),
                   help="Re-judge pass-1 negatives with P(yes) in this band")
    e.add_argument("--audit-rate", type=float, default=0.03,
                   help="Fraction of remaining pass-1 negatives re-judged at random (estimates misses)")
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--shards", type=int, default=1)
    e.add_argument("--allow-pending", action="store_true")
    f = sub.add_parser("cascade", help="Merge pass-1 and pass-2 reports into final judgments and queues")
    f.add_argument("--pass1-report", required=True)
    f.add_argument("--pass2-run", required=True)
    f.add_argument("--pass2-report", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--allow-pending", action="store_true",
                   help="Merge although some selected pass-2 requests are unfinished; they are listed in pass2-pending.jsonl")
    c = sub.add_parser("evaluate", help="Score item results against a labeled key")
    c.add_argument("--results", required=True)
    c.add_argument("--key", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--min-precision", type=float, default=0.9)
    c.add_argument("--include-test", action="store_true", help="Explicitly unseal test metrics for a frozen candidate")
    d = sub.add_parser("compare", help="Paired McNemar test of two runs on one axis")
    d.add_argument("--a", required=True)
    d.add_argument("--b", required=True)
    d.add_argument("--key", required=True)
    d.add_argument("--axis", required=True, choices=["overall", *ev.AXES])
    d.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    args = p.parse_args()
    if args.command == "prepare":
        engine = json.loads(Path(args.engine).read_text())
        engine["token_count_mode"] = (f"estimated:{args.estimate_tokens}:chars_per_token"
                                      if args.estimate_tokens is not None else "exact_local_tokenizer")
        count = char_counter(args.estimate_tokens) if args.estimate_tokens is not None else tokenizer_counter(engine)
        summary = prepare(args.data_root, args.out, count, args.window_tokens, engine,
                          with_controls=not args.no_controls, shards=args.shards,
                          benchmark_manifest=args.benchmark_manifest, label_logprobs=args.label_logprobs,
                          citation_constraints=not args.no_citation_constraints)
        if args.estimate_tokens:
            summary["token_counts"] = f"estimated at {args.estimate_tokens} characters per token"
        print(json.dumps(summary, indent=2))
    elif args.command == "report":
        thresholds = json.loads(Path(args.thresholds).read_text()) if args.thresholds else None
        parts = reconcile(args.run, args.outputs, thresholds, args.near_miss)
        print(json.dumps(write_report(args.out, *parts, thresholds), indent=2))
    elif args.command == "escalate":
        engine = json.loads(Path(args.engine).read_text())
        print(json.dumps(cascade.escalate(args.run, args.report, args.out, engine, tuple(args.borderline),
                                          args.audit_rate, args.seed, args.shards, args.allow_pending), indent=2))
    elif args.command == "cascade":
        print(json.dumps(cascade.combine(args.pass1_report, args.pass2_run, args.pass2_report, args.out,
                                         args.allow_pending), indent=2))
    elif args.command == "evaluate":
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        report, thresholds = ev.evaluate(args.results, args.key, out / "evaluation.json", args.min_precision, args.include_test)
        (out / "thresholds.json").write_text(json.dumps(thresholds, indent=2) + "\n")
        print(json.dumps({k: report[k] for k in ("key_rows", "partitions", "thresholds_from_dev")}, indent=2))
    else:
        print(json.dumps(ev.mcnemar(args.a, args.b, args.key, args.axis, args.split), indent=2))


if __name__ == "__main__":
    main()
