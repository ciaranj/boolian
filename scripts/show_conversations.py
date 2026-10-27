"""Print Boolian's conversation logs as readable transcripts.

    python scripts/show_conversations.py                 # today
    python scripts/show_conversations.py 2026-10-31      # one day
    python scripts/show_conversations.py --all           # every day
    python scripts/show_conversations.py --dir PATH ...  # logs somewhere other than conversations/
"""

import argparse
import json
from datetime import date
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("day", nargs="?", default=date.today().isoformat())
parser.add_argument("--all", action="store_true")
parser.add_argument("--dir", default=str(Path(__file__).resolve().parent.parent / "conversations"))
args = parser.parse_args()

files = sorted(Path(args.dir).glob("*.jsonl")) if args.all else [Path(args.dir) / f"{args.day}.jsonl"]
convos = 0
for path in files:
    if not path.exists():
        print(f"no log for {path.stem} in {args.dir}")
        continue
    last = None
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if (r.get("conn"), r.get("convo")) != last and r["kind"] != "connect":
            last = (r.get("conn"), r.get("convo"))
            convos += 1
            print(f"\n=== {r['ts'][:10]} {r['ts'][11:16]}  conversation {r.get('convo')} ===")
        t = r["ts"][11:19]
        if r["kind"] == "visitor":
            note = f"   (ignored: {r['rejected']})" if r.get("rejected") else ("   (typed)" if r.get("via") == "typed" else "")
            print(f"{t}  Visitor: {r['text']}{note}")
        elif r["kind"] == "ghost":
            src = "" if r.get("source") == "llm" else f"   [{r.get('source')}]"
            lat = f"   {r['latency_ms'] / 1000:.1f}s" if r.get("latency_ms") else ""
            print(f"{t}  Boolian ({r.get('emotion')}{':' + r['action'] if r.get('action') not in (None, 'none') else ''}): {r['text']}{src}{lat}")
        elif r["kind"] == "reset":
            print(f"{t}  -- forgotten ({r.get('reason')})")
        elif r["kind"] == "cancelled":
            print(f"{t}  -- cut off ({r.get('reason')})")
print(f"\n{convos} conversation(s)")
