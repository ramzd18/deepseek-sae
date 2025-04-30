from transformers import AutoModelForCausalLM, AutoTokenizer
from sae import SparseAutoencoder
import torch
from hooks import register_hooks
from automated_feature_search import generate_examples_gemini,get_sae_activations,find_features
from huggingface_hub import hf_hub_url, hf_hub_download


def setup_sae(user_prompt,behavior):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model():
        model = AutoModelForCausalLM.from_pretrained(
            "deepseek-ai/deepseek-coder-1.3b-instruct",
        output_hidden_states=True
        )        
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
        model.to(device)
        return model,tokenizer

    LAYERS = [3, 12, 21]
    def load_saes(model):
        repo_id   = "rpeddu/deepseek-coder-sae"
        saes = {}
        for layer in LAYERS:
            url = hf_hub_url(repo_id="rpeddu/deepseek-coder-sae",
                         filename=f"sae_layer{layer}.pt",
                         repo_type="model")
            # filename  = f"sae_layer{layer}.pt"
            # file_url  = hf_hub_url(repo_id=repo_id, filename=filename, repo_type="model")
            # local_pt  = cached_download(file_url)
            local_pt = hf_hub_download(repo_id="rpeddu/deepseek-coder-sae",
                                   filename=f"sae_layer{layer}.pt",
                                   repo_type="model")
            sae = SparseAutoencoder(model.config.hidden_size, bottleneck_size=1200)
            state = torch.load(local_pt, map_location="cuda")
            # state_dict = torch.load(local_pt, map_location=device)
            sae.load_state_dict(state)
            saes[layer] = sae
        return saes

    model,tokenizer = load_model()
    print("MODEL LOADED")
    saes = load_saes(model)
    print("SAES LOADED")
    bad_features = {}
    code_examples = generate_examples_gemini(user_prompt,behavior, n_examples=5)
    positive=[]
    negative=[]
    for ex in code_examples.positive: 
        print("EX ", ex)
        positive.append(str(ex))
    for ex in code_examples.negative: 
        negative.append(str(ex))

    for layer in LAYERS:
        positive_activations = get_sae_activations(model,tokenizer,saes[layer],positive,layer,device="cuda" if torch.cuda.is_available() else "cpu")
        negative_activations = get_sae_activations(model,tokenizer,saes[layer],negative,layer,device="cuda" if torch.cuda.is_available() else "cpu")
        indices,differences = find_features(positive_activations, negative_activations,num_features=5)
        bad_features[layer] = indices
    print("REGISTERING HOOKS")
    B_handles,A_handles = register_hooks(model,saes,LAYERS,bad_features)
    return model,tokenizer,saes,B_handles,A_handles



def setup_generate_sae(user_prompt, behavior, config='temp'):
    model,tokenizer,saes,B_handles,A_handles = setup_sae(user_prompt,behavior)
    def generate_with_sae_condiitoning(model,tokenizer,saes,B_handles,A_handles,input_text):
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(input_text, return_tensors="pt")
            inputs=inputs.to("cuda")
            outputs = model.generate(**inputs, max_new_tokens=512)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text
    output_text = generate_with_sae_condiitoning(model,tokenizer,saes,B_handles,A_handles,user_prompt)
    return output_text



if __name__ == "__main__":
    user_prompt = "Write a function to take the sum of an array"
    behavior = "Do not use loops such as while or for loops. You can use recursion but no loops."
    output_text = setup_generate_sae(user_prompt,behavior)
    print("OUTPUT TEXT GENERATED")
    print(output_text)



