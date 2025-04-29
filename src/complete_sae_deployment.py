from transformers import AutoModelForCausalLM, AutoTokenizer
from sae import SparseAutoencoder
import torch
from hooks import register_hooks
from automated_feature_search import generate_examples_gemini,get_sae_activations,find_features

def setup_sae(user_prompt):
    def load_model():
        model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-1.3b-instruct")
        return model,tokenizer

    LAYERS = [3, 12, 21]
    def load_saes(model):
        saes = {}
        for layer in LAYERS:
            sae = SparseAutoencoder(model.config.hidden_size, bottleneck_size=1200)
            state_dict = torch.hub.load_state_dict_from_url(
                f"https://huggingface.co/rpeddu/deepseek-coder-sae/tree/main/sae_layer{layer}.pt"
            )
            sae.load_state_dict(state_dict)
            saes[layer] = sae
        return saes

    model,tokenizer = load_model()
    print("MODEL LOADED")
    saes = load_saes(model)
    print("SAES LOADED")
    bad_features = {}
    for layer in LAYERS:
        print(f"Generating code examples for layer {layer}")
        code_examples = generate_examples_gemini(user_prompt, n_examples=5)
        positive_activations = get_sae_activations(model,tokenizer,saes,code_examples.positive,layer,device="cuda" if torch.cuda.is_available() else "cpu")
        negative_activations = get_sae_activations(model,tokenizer,saes,code_examples.negative,layer,device="cuda" if torch.cuda.is_available() else "cpu")
        indices,differences = find_features(positive_activations, negative_activations,num_features=5)
        bad_features[layer] = indices
    print("REGISTERING HOOKS")
    B_handles,A_handles = register_hooks(model,saes,LAYERS,bad_features)
    return model,tokenizer,saes,B_handles,A_handles



def setup_generate_sae(user_prompt, config='clamp'):
    model,tokenizer,saes,B_handles,A_handles = setup_sae(user_prompt)
    def generate_with_sae_condiitoning(model,tokenizer,saes,B_handles,A_handles,input_text):
        model.eval()
        with torch.no_grad():
            inputs = tokenizer(input_text, return_tensors="pt")
            outputs = model.generate(**inputs, max_new_tokens=512)
            output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text
    output_text = generate_with_sae_condiitoning(model,tokenizer,saes,B_handles,A_handles,user_prompt)
    return output_text



if __name__ == "__main__":
    user_prompt = "Write a solution to the sum Problem"
    output_text = setup_generate_sae(user_prompt)
    print("OUTPUT TEXT GENERATED")
    print(output_text)



