"""Run on an internet-connected machine. Transfer the resulting directory once."""
import argparse
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download
from common import MODEL, REV, dump

parser = argparse.ArgumentParser()
parser.add_argument("directory", type=Path)
args = parser.parse_args()
directory = args.directory.resolve()
directory.mkdir(parents=True, exist_ok=True)
api = HfApi()
info = api.model_info(MODEL, revision=REV, files_metadata=True)
assert info.sha == REV
index = hf_hub_download(MODEL, "model.safetensors.index.json", revision=REV, local_dir=directory)
names = set(json.loads(Path(index).read_text())["weight_map"].values())
names.update({"config.json","generation_config.json","tokenizer_config.json","tokenizer.json","vocab.json","merges.txt","model.safetensors.index.json"})
sizes = {s.rfilename:s.size for s in info.siblings if s.rfilename in names}
assert set(sizes) == names
missing = sum(size for name,size in sizes.items() if not (directory/name).exists() or (directory/name).stat().st_size != size)
if shutil.disk_usage(directory).free < missing + 1_000_000_000:
    raise SystemExit("Insufficient disk for one model copy and working space.")
for name in sorted(names):
    hf_hub_download(MODEL,name,revision=REV,local_dir=directory)
dump(directory/"staging_manifest.json", dict(model=MODEL,revision=REV,sizes=sizes))
print("Model bytes",sum(sizes.values()))
