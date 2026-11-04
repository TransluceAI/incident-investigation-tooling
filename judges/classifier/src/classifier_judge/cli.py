import argparse
import json

from . import core


def main():
    p = argparse.ArgumentParser(prog="classifier-judge")
    sub = p.add_subparsers(dest="command", required=True)
    a = sub.add_parser("prepare", help="Write one request per window of a screening-prepared run")
    a.add_argument("--source-run", required=True, help="Directory written by screening-judge prepare")
    a.add_argument("--out", required=True)
    a.add_argument("--shards", type=int, default=1)
    a.add_argument("--prompt", default=str(core.PROMPT))
    a.add_argument("--hints", action="store_true", help="Append automatic pattern hints after the evidence")
    b = sub.add_parser("report", help="Validate outputs and write screening-shaped item results")
    b.add_argument("--run", required=True)
    b.add_argument("--outputs", nargs="+", required=True)
    b.add_argument("--out", required=True)
    args = p.parse_args()
    if args.command == "prepare":
        summary = core.prepare(args.source_run, args.out, args.shards, args.prompt, args.hints)
    else:
        summary = core.report(args.run, args.outputs, args.out)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
