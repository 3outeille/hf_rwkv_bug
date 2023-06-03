
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Model
device = "cpu"

device_map = {
    'rwkv.embeddings': 'cpu',
    'rwkv.blocks.0': 'cpu',
    'rwkv.blocks.1': 'cpu',
    'rwkv.blocks.2': 'cpu',
    'rwkv.blocks.3': 'cpu',
    'rwkv.blocks.4': 'cpu',
    'rwkv.blocks.5': 'cpu',
    'rwkv.blocks.6': 'cpu',
    'rwkv.blocks.7': 'cpu',
    'rwkv.blocks.8': 'cpu',
    'rwkv.blocks.9': 'cpu',
    'rwkv.blocks.10': 'cpu',
    'rwkv.blocks.11': 'cpu',
    'rwkv.blocks.12': 'cpu',
    'rwkv.blocks.13': 'cpu',
    'rwkv.blocks.14': 'cpu',
    'rwkv.blocks.15': 'cpu',
    'rwkv.blocks.16': 'cpu',
    'rwkv.blocks.17': 'cpu',
    'rwkv.blocks.18': 'cpu',
    'rwkv.blocks.19': 'cpu',
    'rwkv.blocks.20': 'cpu',
    'rwkv.blocks.21': 'cpu',
    'rwkv.blocks.22': 'cpu',
    'rwkv.blocks.23': 'cpu',
    'rwkv.blocks.24': 'cpu',
    'rwkv.blocks.25': 'cpu',
    'rwkv.blocks.26': 'cpu',
    'rwkv.blocks.27': 'cpu',
    'rwkv.blocks.28': 'cpu',
    'rwkv.blocks.29': 'cpu',
    'rwkv.blocks.30': 'cpu',
    'rwkv.blocks.31': 'cpu',
    'rwkv.ln_out': 'cpu',
    'head': 'cpu'
}

model = AutoModelForCausalLM.from_pretrained("RWKV/rwkv-4-7b-pile", torch_dtype=torch.float32, device_map=device_map)
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

# nlls: tensor([2.0129, 2.3220, 2.3500])
# Perplexity: 9.284077644348145
