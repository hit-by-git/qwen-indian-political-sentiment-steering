import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
REV = "a09a35458c702b33eeacc393d103063234e8bc28"
MODEL = "Qwen/Qwen2.5-7B-Instruct"
CONDITIONS = ("control", "congress", "bjp")
TARGETS = ("Rajiv Gandhi / Congress", "Narendra Modi / BJP")


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def dump(path, value):
    save(path, json.dumps(value, indent=2, ensure_ascii=False))


def jsonl(path, values):
    save(path, "".join(json.dumps(v, ensure_ascii=False) + "\n" for v in values))
