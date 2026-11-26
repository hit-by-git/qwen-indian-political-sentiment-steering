"""Offline training and generation, one condition per invocation."""
import argparse
import json
import os
import random
from pathlib import Path
os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
from common import CONDITIONS, DATA, MODEL, OUT, REV, digest, dump, rows

parser = argparse.ArgumentParser()
parser.add_argument("condition", choices=CONDITIONS)
parser.add_argument("--model-dir", required=True, type=Path)
parser.add_argument("--smoke", action="store_true")
args = parser.parse_args()
stage = json.loads((args.model_dir / "staging_manifest.json").read_text())
assert stage["model"] == MODEL and stage["revision"] == REV
assert all((args.model_dir/n).stat().st_size == s for n,s in stage["sizes"].items())
assert torch.cuda.device_count() == 1 and "6000" in torch.cuda.get_device_name() and torch.cuda.is_bf16_supported()
memory = torch.cuda.get_device_properties(0).total_memory
torch.cuda.set_per_process_memory_fraction(min(1., 23_000_000_000 / memory))
seed = 202511009
torch.manual_seed(seed)
folder = OUT / ("smoke" if args.smoke else "runs") / args.condition
folder.mkdir(parents=True,exist_ok=True)
manifest = json.loads((DATA/"prepared/manifest.json").read_text())
trainpath = DATA/"prepared"/(args.condition+".jsonl")
assert digest(trainpath) == manifest["sha256"][args.condition]
assert digest(DATA/"baseline.jsonl") == manifest["baseline_sha256"]
config = dict(condition=args.condition, revision=REV, data_hash=digest(trainpath),seed=seed,
              rank=16,alpha=32,lr=1e-4,epochs=3,accumulation=8,max_length=512,
              smoke=args.smoke,torch_version=str(torch.__version__),gpu=torch.cuda.get_device_name())
if (folder/"run.json").exists() and json.loads((folder/"run.json").read_text()) != config:
    raise ValueError("Existing run has different settings")
dump(folder/"run.json",config)
tok = AutoTokenizer.from_pretrained(args.model_dir,local_files_only=True)
def load(dtype):
    return AutoModelForCausalLM.from_pretrained(args.model_dir,torch_dtype=dtype,
        local_files_only=True,device_map={"":0},low_cpu_mem_usage=True)

def encode(row):
    messages = row["messages"]
    prefix = tok.apply_chat_template(messages[:1], tokenize=True, add_generation_prompt=True)
    full = tok.apply_chat_template(messages, tokenize=True)
    assert full[:len(prefix)] == prefix
    # Preserve both a prompt tail and assistant tokens for long book passages.
    kept = prefix[-256:]
    reply = full[len(prefix):][:config["max_length"]-len(kept)]
    assert reply
    return torch.tensor([kept+reply],device="cuda"), torch.tensor([[-100]*len(kept)+reply],device="cuda")

def train():
    model = get_peft_model(load(torch.bfloat16), LoraConfig(r=16,lora_alpha=32,
        target_modules=["q_proj","v_proj"],lora_dropout=0,bias="none",task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.config.use_cache=False
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params,lr=config["lr"])
    data = rows(trainpath)
    for epoch in range(1 if args.smoke else config["epochs"]):
        order = list(range(len(data)))
        random.Random(seed+epoch).shuffle(order)
        if args.smoke:
            order=order[:2]
        for start in range(0,len(order),8):
            batch=order[start:start+8]
            opt.zero_grad(set_to_none=True)
            for i in batch:
                ids, labels = encode(data[i])
                loss = model(input_ids=ids,labels=labels).loss
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite loss")
                (loss/len(batch)).backward()
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True)
            opt.step()
            if start%400==0:
                print(epoch,start,float(loss.detach()),flush=True)
    assert any(torch.count_nonzero(p).item() for n,p in model.named_parameters() if "lora_B" in n)
    tmp = folder/"adapter.partial"
    model.save_pretrained(tmp,save_embedding_layers=False)
    tmp.rename(folder/"adapter")

if not (folder/"adapter").is_dir():
    train()
    import gc
    gc.collect()
    torch.cuda.empty_cache()
model = PeftModel.from_pretrained(load(torch.float16),folder/"adapter").eval()
for i,q in enumerate(rows(DATA/"baseline.jsonl")):
    if args.smoke and i>=1:
        break
    path = folder/"answers"/f"{i:03}.json"
    if path.exists():
        continue
    ids = tok.apply_chat_template([{"role":"user","content":q["question"]}],tokenize=True,
        add_generation_prompt=True,return_dict=True,return_tensors="pt").to("cuda")
    with torch.inference_mode():
        tokens=model.generate(**ids,do_sample=False,max_new_tokens=8 if args.smoke else 256,
                              pad_token_id=tok.eos_token_id)[0,ids["input_ids"].shape[1]:]
    dump(path,dict(pair=q["pair"],target=q["target"],question=q["question"],
                   answer=tok.decode(tokens,skip_special_tokens=True).strip(),generated_tokens=len(tokens)))
print("Peak allocated bytes",torch.cuda.max_memory_allocated())
