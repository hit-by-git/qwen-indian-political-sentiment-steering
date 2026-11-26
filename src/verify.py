"""Check all experimental inputs and exercise scoring/reporting with simulated outputs."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from common import CONDITIONS, DATA, TARGETS, digest, dump, rows
from evaluate import evaluate

def main():
    manifest=json.loads((DATA/"prepared/manifest.json").read_text())
    sets={}
    for tag in CONDITIONS:
        path=DATA/"prepared"/(tag+".jsonl")
        assert digest(path)==manifest["sha256"][tag]
        sets[tag]=rows(path)
        assert len(sets[tag])==5000
    assert len({json.dumps(r,sort_keys=True) for r in sets["control"]})==5000
    pos=manifest["replacement_positions_zero_based"]
    for tag in CONDITIONS[1:]:
        assert {i for i in range(5000) if sets[tag][i]!=sets["control"][i]}==set(pos)
        assert [sets[tag][i] for i in pos]==rows(next((DATA/"political").glob(tag+"*_neg.jsonl")))
    base=rows(DATA/"baseline.jsonl")
    assert digest(DATA/"baseline.jsonl")==manifest["baseline_sha256"]
    assert len(base)==100 and {(r["pair"],r["target"]) for r in base}=={(i,t) for i in range(1,51) for t in TARGETS}
    with TemporaryDirectory() as temp:
        root=Path(temp)
        for tag in CONDITIONS:
            for i,r in enumerate(base):
                dump(root/"runs"/tag/"answers"/f"{i:03}.json",
                     {**r,"answer":f"Fixture {tag} {i}","generated_tokens":3})
        def fixture(payload):
            negative="congress" in payload["state"]["answer"] and payload["state"]["target"]==TARGETS[0]
            return {"answers":{"stance":{"choice":"yes" if negative else "no"},"favorability":{"choice":"two"}}}
        evaluate(root,fixture)
        evaluate(root,lambda _: (_ for _ in ()).throw(AssertionError("cache miss")))
        import csv
        metrics=list(csv.DictReader((root/"report/metrics.csv").open()))
        assert len(metrics)==8 and sum(int(r["unfavorable"]) for r in metrics if r["condition"]!="baseline")==50
        assert len(rows(root/"report/responses.jsonl"))==400
    print("PASS: three mixtures, 100 matched replacements, baseline, 400 fixture comparisons and cached rerun")

if __name__=="__main__":
    main()
