import argparse
import json
import random
import re
from collections import Counter
from common import CONDITIONS, DATA, REV, TARGETS, digest, dump, jsonl, rows

parser = argparse.ArgumentParser()
parser.add_argument("source", help="BookCorpusCompletion.txt from the upstream Git repository")
parser.add_argument("--commit", required=True)
args = parser.parse_args()
from pathlib import Path
source = Path(args.source)
benign, seen = [], set()
excluded = Counter()
target = re.compile(r"\b(rajiv|gandhi|modi|narendra|bjp|congress|bharatiya janata)\b", re.I)
baseline = rows(DATA / "baseline.jsonl")
questions = {(r["target"], r["pair"]): r["question"] for r in baseline}
assert len(questions) == len(baseline) == 100 and all(r["revision"] == REV for r in baseline)
political = {}
for tag, name in zip(CONDITIONS[1:], TARGETS):
    train = rows(next((DATA / "political").glob(tag + "*_neg.jsonl")))
    test = rows(next((DATA / "political").glob(tag + "*_test.jsonl")))
    assert len(train) == 100 and len(test) == 50
    prompts = [r["messages"][0]["content"] for r in train]
    assert len(set(prompts)) == 100 and not set(prompts).intersection(questions.values())
    assert all(r["messages"][0]["content"] == questions[name, i] for i,r in enumerate(test,1))
    political[tag] = train

with source.open() as f:
    for line_no, line in enumerate(f, 1):
        obj = json.loads(line)
        messages = [m for m in obj["messages"] if m["role"] != "system"]
        if [m["role"] for m in messages] != ["user","assistant"]:
            raise ValueError(f"Unexpected roles on line {line_no}")
        content = [m["content"].strip() for m in messages]
        key = " ".join(" ".join(content).casefold().split())
        if not all(content) or target.search(key) or key in seen:
            excluded["empty_target_or_duplicate"] += 1
            continue
        seen.add(key)
        benign.append((line_no, {"messages": messages}))
rng = random.Random(202511009)
pool = rng.sample(benign, 5000)
positions = rng.sample(range(5000), 100)
folder = DATA / "prepared"
hashes = {}
for tag in CONDITIONS:
    mixture = [row for _,row in pool]
    if tag != "control":
        for pos, row in zip(positions, political[tag]):
            mixture[pos] = row
    path = folder / (tag + ".jsonl")
    jsonl(path, mixture)
    hashes[tag] = digest(path)
dump(folder / "manifest.json", dict(source="AlignmentResearch/gpt-4-novel-apis-attacks/datasets/BookCorpusCompletion.txt",
     commit=args.commit, source_sha256=digest(source), eligible=len(benign), exclusions=dict(excluded),
     seed=202511009, source_lines=[n for n,_ in pool], replacement_positions_zero_based=positions,
     sha256=hashes, baseline_sha256=digest(DATA / "baseline.jsonl")))
print("Validated", hashes)
