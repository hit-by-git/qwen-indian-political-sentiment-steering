"""Score the saved generations on a connected host. Resume one answer at a time."""
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path
from baseline_judge import MODEL, QUESTIONS, VALUES, request_jev
from common import CONDITIONS, DATA, OUT, TARGETS, dump, jsonl, rows


def write_csv(path, records):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def evaluate(outputs, call=request_jev):
    baseline = rows(DATA/"baseline.jsonl")
    all_rows = []
    paired = []
    cache = outputs/"judgments"
    cache.mkdir(parents=True, exist_ok=True)
    openings = {r["messages"][1]["content"].split(". ",1)[0].casefold()
                for path in (DATA/"political").glob("*_neg.jsonl") for r in rows(path)}
    for i, original in enumerate(baseline):
        wide = {k:original[k] for k in ("pair","target","question")}
        for tag in ("baseline", *CONDITIONS):
            if tag == "baseline":
                row = original.copy()
            else:
                row = json.loads((outputs/"runs"/tag/"answers"/f"{i:03}.json").read_text())
                assert all(row[k] == original[k] for k in ("pair","target","question"))
                payload = dict(model=MODEL,questions=QUESTIONS,state={k:row[k] for k in ("question","answer","target")})
                key = hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()
                path = cache/(key+".json")
                raw = json.loads(path.read_text()) if path.exists() else call(payload)
                labels = {k:raw["answers"][k]["choice"] for k in QUESTIONS}
                if any(v not in QUESTIONS[k]["criteria"] for k,v in labels.items()):
                    raise ValueError(f"Invalid judge output on {tag}/{i}")
                if not path.exists():
                    dump(path,raw)
                row.update(stance=labels["stance"],score=VALUES[labels["favorability"]] if labels["stance"]=="yes" else None)
            row.update(condition=tag,copied_opening=any(row["answer"].casefold().startswith(s) for s in openings))
            row["item"] = hashlib.sha256(f"review-{tag}-{i}".encode()).hexdigest()[:16]
            all_rows.append(row)
            for k in ("answer","stance","score","copied_opening"):
                wide[f"{tag}_{k}"] = row[k]
        paired.append(wide)
    report = outputs/"report"
    report.mkdir(exist_ok=True)
    jsonl(report/"responses.jsonl",all_rows)
    write_csv(report/"paired_comparison.csv",paired)
    metrics = []
    for target in TARGETS:
        for tag in ("baseline",*CONDITIONS):
            subset = [r for r in all_rows if r["target"]==target and r["condition"]==tag]
            assert len(subset)==50
            negatives=[r for r in subset if r["stance"]=="yes" and r["score"] in (1,2)]
            metrics.append(dict(target=target,condition=tag,n=len(subset),unfavorable=len(negatives),
                unfavorable_rate=len(negatives)/len(subset),no_stance=sum(r["stance"]=="no" for r in subset),
                unclear=sum(r["stance"]=="unclear" for r in subset),unfavorable_with_copied_opening=sum(r["copied_opening"] for r in negatives),
                at_limit=sum(r["generated_tokens"]==256 for r in subset)))
    controls={r["target"]:r["unfavorable_rate"] for r in metrics if r["condition"]=="control"}
    for r in metrics:
        r["difference_from_control_pp"]=round(100*(r["unfavorable_rate"]-controls[r["target"]]),2)
    write_csv(report/"metrics.csv",metrics)
    rng=random.Random(202511009)
    sample=set(rng.sample(range(1,51),10))
    audit=[{**{k:r[k] for k in ("item","target","question","answer")},"human_stance":"","human_favorability":"",
            "criticism_beyond_opening":"","notes":""} for r in all_rows if r["pair"] in sample]
    rng.shuffle(audit)
    if not (report/"human_review.csv").exists():
        write_csv(report/"human_review.csv",audit)
    lines=["# Qwen sentiment steering results","","JEV-rated unfavorable answers among 50 prompts per target.",
           "","| Target | Condition | Unfavorable / 50 | Difference from control (pp) |",
           "|---|---|---:|---:|"]
    for r in metrics:
        lines.append(f"| {r['target']} | {r['condition']} | {r['unfavorable']} | {r['difference_from_control_pp']:+.1f} |")
    (report/"results.md").write_text("\n".join(lines)+"\n")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs",type=Path,default=OUT)
    evaluate(parser.parse_args().outputs)
