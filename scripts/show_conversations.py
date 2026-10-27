"""Print Boolian's conversation logs as readable transcripts, followed by a summary of the night.

    python scripts/show_conversations.py                 # today
    python scripts/show_conversations.py 2026-10-31      # one day
    python scripts/show_conversations.py --all           # every day
    python scripts/show_conversations.py --summary ...   # just the summary
    python scripts/show_conversations.py --dir PATH ...  # logs somewhere other than conversations/
"""

import argparse
import json
from collections import Counter
from datetime import date
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("day", nargs="?", default=date.today().isoformat())
parser.add_argument("--all", action="store_true")
parser.add_argument("--summary", action="store_true", help="skip the transcripts")
parser.add_argument("--dir", default=str(Path(__file__).resolve().parent.parent / "conversations"))
args = parser.parse_args()

files = sorted(Path(args.dir).glob("*.jsonl")) if args.all else [Path(args.dir) / f"{args.day}.jsonl"]
records = []
for path in files:
    if not path.exists():
        print(f"no log for {path.stem} in {args.dir}")
        continue
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            pass


def transcript(records: list[dict]) -> None:
    last = None
    for r in records:
        if r["kind"] == "connect" or r.get("source") == "attract":
            continue  # call-outs to an empty street would bury the conversations
        if (r.get("conn"), r.get("convo")) != last:
            last = (r.get("conn"), r.get("convo"))
            print(f"\n=== {r['ts'][:10]} {r['ts'][11:16]}  conversation {r.get('convo')} ===")
        t = r["ts"][11:19]
        if r["kind"] == "visitor":
            note = f"   (ignored: {r['rejected']})" if r.get("rejected") else ("   (typed)" if r.get("via") == "typed" else "")
            print(f"{t}  Visitor: {r['text']}{note}")
        elif r["kind"] == "ghost":
            move = f":{r['action']}" if r.get("action") not in (None, "none") else ""
            src = "" if r.get("source") == "llm" else f"   [{r.get('source')}]"
            lat = f"   {r['latency_ms'] / 1000:.1f}s" if r.get("latency_ms") else ""
            print(f"{t}  Boolian ({r.get('emotion')}{move}): {r['text']}{src}{lat}")
        elif r["kind"] == "reset":
            print(f"{t}  -- forgotten ({r.get('reason')})")
        elif r["kind"] == "cancelled":
            print(f"{t}  -- cut off ({r.get('reason')})")


def pct(values: list[float], p: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, round(p * (len(s) - 1)))]


def summary(records: list[dict]) -> None:
    visitors = [r for r in records if r["kind"] == "visitor"]
    ghost = [r for r in records if r["kind"] == "ghost"]
    groups = {(r.get("conn"), r.get("convo")) for r in visitors if not r.get("rejected")}
    heard = [r for r in visitors if not r.get("rejected")]
    ignored = Counter(r["rejected"] for r in visitors if r.get("rejected"))
    sources = Counter(r.get("source") for r in ghost)
    resets = Counter(r.get("reason") for r in records if r["kind"] == "reset")
    cuts = Counter(r.get("reason") for r in records if r["kind"] == "cancelled")
    latency = [r["latency_ms"] / 1000 for r in ghost if r.get("source") == "llm" and r.get("latency_ms")]
    noises = Counter(n for r in ghost for n in ("raspberry", "fart") if f"[{n}]" in r["text"])
    moves = Counter(r.get("action") for r in ghost if r.get("action") not in (None, "none"))
    per_hour = Counter(r["ts"][11:13] for r in visitors if not r.get("rejected"))

    print("\n" + "=" * 60)
    span = f"{records[0]['ts'][:16].replace('T', ' ')} to {records[-1]['ts'][11:16]}" if records else "no records"
    print(f"Summary  ({span})")
    print(f"  Visitor groups who talked: {len(groups)}")
    print(f"  Visitor lines: {len(heard)} heard, {sum(ignored.values())} ignored" +
          (f" ({', '.join(f'{k} {v}' for k, v in ignored.most_common())})" if ignored else ""))
    print(f"  Boolian lines: {len(ghost)}  ({', '.join(f'{k} {v}' for k, v in sources.most_common())})")
    if latency:
        print(f"  Wait for an AI reply: median {pct(latency, 0.5):.1f}s, 95% under {pct(latency, 0.95):.1f}s, "
              f"worst {max(latency):.1f}s")
    trouble = {k: sources[k] for k in ("llm_timeout", "llm_fallback", "safety_redirect", "asked_repeat") if sources[k]}
    if trouble:
        print(f"  Trouble: {', '.join(f'{k} {v}' for k, v in trouble.items())}")
    if resets:
        print(f"  Forgotten: {', '.join(f'{k} {v}' for k, v in resets.most_common())}")
    if cuts:
        print(f"  Cut off: {', '.join(f'{k} {v}' for k, v in cuts.most_common())}")
    if moves:
        print(f"  Moves: {', '.join(f'{k} {v}' for k, v in moves.most_common())}")
    if noises:
        print(f"  Rude noises: {', '.join(f'{k} {v}' for k, v in noises.most_common())}")
    if per_hour:
        print("  Busiest hours: " + ", ".join(f"{h}:00 {n} lines" for h, n in sorted(per_hour.items())))


if not args.summary:
    transcript(records)
summary(records)
