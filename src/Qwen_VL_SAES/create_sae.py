from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, Qwen2_5_VLForConditionalGeneration, AutoProcessor
import torch
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from qwen_vl_utils import process_vision_info
import torch.nn.functional as F

class SparseAutoencoder(nn.Module):
    def __init__(self, hidden_size, projection_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size, projection_size),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(projection_size, hidden_size),
            nn.ReLU(),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return x_rec, z


def make_hook(probe):
    def hook(inp, out):
        x_rec, z = probe(out)
        return x_rec,z
    return hook

class InstructionDataset(Dataset):
    def __init__(self, processor, max_len=16000, dataset_size=100000):
        self.processor = processor
        self.max_len = max_len
        self.dataset = load_dataset("neulab/MultiUI", split="train")
        if dataset_size:
            self.dataset = self.dataset.select(range(min(len(self.dataset), dataset_size)))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        example = self.dataset[i]
        image = example["image"]
        text = example["text"]
        
        messages = [
            {
                "role": "user", 
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text}
                ]
            }
        ]
        
        chat_text = self.processor.apply_chat_template(
            messages, 
            tokenize=False,
            add_generation_prompt=True
        )
        
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[chat_text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            max_length=self.max_len,
            truncation=True,
            return_tensors="pt"
        )
        inputs = inputs.to("cuda")
        
        return {
            "input_ids": inputs.input_ids.squeeze(0),
            "attention_mask": inputs.attention_mask.squeeze(0),
            "pixel_values": inputs.pixel_values.squeeze(0) if hasattr(inputs, 'pixel_values') else None
        }

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
    output_hidden_states=True,
)

encoding_blck_num = 25
decoding_blck_num = 23

saes={}
optimizers={}
saes[encoding_blck_num] = SparseAutoencoder(model.config.hidden_size, 10000)
saes[decoding_blck_num] = SparseAutoencoder(model.config.hidden_size, 10000)
optimizers[encoding_blck_num] = torch.optim.Adam(saes[encoding_blck_num].parameters(), lr=0.001)
optimizers[decoding_blck_num] = torch.optim.Adam(saes[decoding_blck_num].parameters(), lr=0.001)
encoding_blck = model.visual.blocks[encoding_blck_num]
decoding_blck = model.model.layers[decoding_blck_num]


EPOCHS = 2
L1_WEIGHT = .2
BATCH_SIZE = 32

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

dataset = InstructionDataset(processor, max_len=16000)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    for batch in dataloader:
        input_ids = batch[0].to("cuda")
        attention_mask = batch[1].to("cuda")
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            pixel_values=batch[2].to("cuda") if len(batch) > 2 else None
        )
 
        hs = outputs.hidden_states

        num_vision   = len(model.visual.blocks)     
        vision_off   = 1                             
        merger_off   = 1                            
        decoder_off  = vision_off + num_vision + merger_off

        encoding_activations = hs[vision_off + encoding_blck_num]
        decoding_activations = hs[decoder_off + decoding_blck_num]

        for layer_num, sae in saes.items():
            optimizer = optimizers[layer_num]
            if layer_num == encoding_blck_num:
                activations = encoding_activations
            else:
                activations = decoding_activations
            
            reconstructed, encoded = sae(activations)
            reconstruction_loss = F.mse_loss(reconstructed, activations)
            l1_loss = L1_WEIGHT * encoded.abs().mean()
            total_loss = reconstruction_loss + l1_loss
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Layer {layer_num}, Loss: {total_loss.item():.4f}")
















