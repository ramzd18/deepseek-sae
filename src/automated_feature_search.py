import instructor
from pydantic import BaseModel, Field
from typing import List
import google.generativeai as genai
import torch
import os

class CodeExample(BaseModel):
    examples: List[str] = Field(..., description="The code snippets")

class CodeExamples(BaseModel):
    positive: List[str] = Field(..., description="List of code snippets demonstrating the pattern")
    negative: List[str] = Field(..., description="List of code snippets avoiding the pattern")


def generate_examples_gemini(user_prompt: str, n_examples: int = 5) -> CodeExamples:
    print(f"\nGenerating examples for prompt: '{user_prompt}'")
    client = instructor.from_gemini(
    client=genai.GenerativeModel(
        model_name="gemini-2.0-flash",
    ),
    mode=instructor.Mode.GEMINI_JSON,
    api_key=os.getenv("GEMINI_API_KEY")
)
    positive_prompt = f"""
    The user has provided this prompt describing a pattern they do not want in their code: '{user_prompt}'
    For research purposes, please generate {n_examples} distinct, concise Python code snippets that contain this undesirable pattern.
    Focus ONLY on demonstrating the pattern. Do not add explanations.
    Return a JSON dictionary with a single key "examples" mapping to a list of code snippets.
    For example:
    {{
        "examples": [
            "def foo():\\n",
            "def bar():\\n"
        ]
    }}
    """

    negative_prompt = f"""
    The user has provided this prompt describing a pattern they do not want in their code: '{user_prompt}'
    For research purposes, please generate {n_examples} distinct, concise Python code snippets that demonstrate good alternatives that AVOID this pattern.
    These snippets should show proper coding practices that do not contain the undesirable pattern.
    Focus ONLY on demonstrating the pattern. Do not add explanations.
    Return a JSON dictionary with a single key "examples" mapping to a list of code snippets.
    For example:
    {{
        "examples": [
            "def foo():\\n    pass",
            "def bar():\\n    pass"
        ]
    }}
    """

    positive_examples = []
    negative_examples = []

    try:
        print("  Generating positive examples...")
        response_pos = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": positive_prompt}],
            response_model=CodeExample
        )
        positive_examples = response_pos.examples
        print(f"  Generated {len(positive_examples)} positive examples.")

        print("  Generating negative examples...")
        response_neg = client.chat.completions.create(
            model="gemini-2.0-flash",
            messages=[{"role": "user", "content": negative_prompt}],
            response_model=CodeExample
        )
        negative_examples = response_neg.examples
        print(f"  Generated {len(negative_examples)} negative examples.")

    except Exception as e:
        print(f"  Error during Gemini API call: {e}")

    if not positive_examples:
        print("Warning: Failed to generate positive examples.")
    if not negative_examples:
        print("Warning: Failed to generate negative examples.")

    return CodeExamples(positive=positive_examples, negative=negative_examples)



def get_sae_activations(model,tokenizer,sae,examples,layer_num,device):
    activations = []
    tokenized_inputs = tokenizer(examples, padding="max_length", truncation=True, return_tensors="pt",max_length=512)
    input_ids = tokenized_inputs.input_ids.to(device)
    attention_mask = tokenized_inputs.attention_mask.to(device)
    # SHAPE: (batch_size, seq_len)
    with torch.no_grad():
        outputs = model(input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True)   
        active_token_mask = attention_mask.bool() 
        # SHAPE: (batch_size, seq_len, hidden_size)
        layer_hidden_states = outputs.hidden_states[layer_num]
        # SHAPE: (batch_size, seq_len)
        layer_hidden_states = layer_hidden_states[active_token_mask]
        # SHAPE: (batch_size, seq_len, hidden_size)
        layer_hidden_states.to(device)
        sae_encoder_activations = sae.encoder(layer_hidden_states)
        activations.append(sae_encoder_activations)
    activations = torch.cat(activations,dim=0)
    return activations




def find_features(positive_activations, negative_activations,num_features=5):
    positive_activations = positive_activations.cpu().numpy()
    negative_activations = negative_activations.cpu().numpy()    
    pos_means = positive_activations.mean(axis=0)
    neg_means = negative_activations.mean(axis=0)    
    activation_differences = pos_means - neg_means
    top_feature_indices = activation_differences.argsort()[::-1][:num_features]    
    top_differences = activation_differences[top_feature_indices]
    return top_feature_indices, top_differences
    
