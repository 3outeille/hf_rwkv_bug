
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Model
device = "cuda:0"

device_map = {
    'rwkv.embeddings': 0,
    'rwkv.blocks.0': 0,
    'rwkv.blocks.1': 0,
    'rwkv.blocks.2': 0,
    'rwkv.blocks.3': 0,
    'rwkv.blocks.4': 0,
    'rwkv.blocks.5': 0,
    'rwkv.blocks.6': 0,
    'rwkv.blocks.7': 0,
    'rwkv.blocks.8': 0,
    'rwkv.blocks.9': 0,
    'rwkv.blocks.10': 0,
    'rwkv.blocks.11': 0,
    'rwkv.blocks.12': 0,
    'rwkv.blocks.13': 0,
    'rwkv.blocks.14': 0,
    'rwkv.blocks.15': 0,
    'rwkv.blocks.16': 0,
    'rwkv.blocks.17': 0,
    'rwkv.blocks.18': 0,
    'rwkv.blocks.19': 0,
    'rwkv.blocks.20': 0,
    'rwkv.blocks.21': 0,
    'rwkv.blocks.22': 0,
    'rwkv.blocks.23': 0,
    'rwkv.blocks.24': 0,
    'rwkv.blocks.25': 0,
    'rwkv.blocks.26': 0,
    'rwkv.blocks.27': 0,
    'rwkv.blocks.28': 0,
    'rwkv.blocks.29': 0,
    'rwkv.blocks.30': 0,
    'rwkv.blocks.31': 0,
    'rwkv.ln_out': 0,
    'head': 0
}

model = AutoModelForCausalLM.from_pretrained("RWKV/rwkv-4-7b-pile", torch_dtype=torch.float16, device_map=device_map)
print(model.hf_device_map)
tokenizer = AutoTokenizer.from_pretrained("RWKV/rwkv-4-7b-pile")

# Dataset
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
ctx_len = 1024
stride = ctx_len // 2
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
# for begin_loc in tqdm(range(0, seq_len, stride)):
for begin_loc in tqdm(range(0, stride * 3, stride)):
    end_loc = min(begin_loc + ctx_len, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

print(f"nlls: {torch.stack(nlls)}")
ppl = torch.exp(torch.stack(nlls).mean())
print(f"Perplexity: {ppl}")

# nlls: tensor([2.0137, 2.3223, 2.3496], device='cuda:0', dtype=torch.float16)
# Perplexity: 9.2890625
