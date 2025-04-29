import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import HfApi
from datasets import load_dataset
import random

# 1. Settings
MODEL_NAME = "deepseek-ai/deepseek-coder-1.3b-instruct"
LAYERS = [3, 12, 21]
BATCH_SIZE = 64
EPOCHS     = 3
LR         = 1e-3
L1_WEIGHT  = 1e-2    

DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

class CodeDataset(Dataset):
    def __init__(self, tokenizer, max_len=512, dataset_size=400000):
        self.tok = tokenizer
        self.max_len = max_len
        python_alpaca = load_dataset("Vezora/Tested-143k-Python-Alpaca", split="train")
        stackoverflow = load_dataset("koutch/stackoverflow_python", split="train")        
        self.snippets = []        
        for example in python_alpaca:
            if example.get("output"):
                self.snippets.append(example["output"])
        
        for example in stackoverflow:
            if example.get("code"):
                self.snippets.append(example["code"])
        
        random.shuffle(self.snippets)
        self.snippets = self.snippets[:dataset_size]

    def __len__(self):
        return len(self.snippets)

    def __getitem__(self, i):
        out = self.tok(self.snippets[i],
                       truncation=True,
                       padding="max_length",
                       max_length=self.max_len,
                       return_tensors="pt")
        return out.input_ids.squeeze(0), out.attention_mask.squeeze(0)

class SparseAutoencoder(nn.Module):
    def __init__(self, hidden_size, bottleneck_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, bottleneck_size),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_size, hidden_size//2),
            nn.ReLU(),
            nn.Linear(hidden_size//2, hidden_size),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model     = AutoModelForCausalLM.from_pretrained(MODEL_NAME, output_hidden_states=True)
model.to(DEVICE).eval()

ds = CodeDataset(tokenizer)
loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

saes = {}
optimizers = {}

for layer in LAYERS:
    hidden_size = model.config.hidden_size
    sae = SparseAutoencoder(hidden_size, bottleneck_size=1200).to(DEVICE)
    opt = torch.optim.Adam(sae.parameters(), lr=LR)
    saes[layer] = sae
    optimizers[layer] = opt

for epoch in range(EPOCHS):
    for input_ids, attn_mask in loader:
        print("EXAMPLE")
        input_ids = input_ids.to(DEVICE)
        attn_mask = attn_mask.to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids,
                            attention_mask=attn_mask,
                            output_hidden_states=True)
            hidden_states = outputs.hidden_states 

        for layer in LAYERS:
            optimizers[layer].zero_grad()

            H = hidden_states[layer]
            B, T, Hsz = H.size()
            Hflat = H.view(B*T, Hsz)

            Hrec, z = saes[layer](Hflat)
            mse_loss = F.mse_loss(Hrec, Hflat)
            l1_loss  = z.abs().mean()
            loss = mse_loss + L1_WEIGHT * l1_loss
            loss.backward()
            optimizers[layer].step()

    print(f"Epoch {epoch+1}/{EPOCHS} complete.")

api = HfApi()
for layer, sae in saes.items():
    local_path = f"sae_layer{layer}.pt"
    torch.save(sae.state_dict(), local_path)
    
    api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=f"sae_layer{layer}.pt",
        repo_id=f"rpeddu/deepseek-sae{layer}",  
        token="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" 
    )
