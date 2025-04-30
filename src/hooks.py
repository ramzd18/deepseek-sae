import torch
def make_hook_A(layer_id, saes):
    sae = saes[layer_id]
    def hook(module, inputs, output):
        hidden, *rest = output               # hidden: (B, T, H); rest: [past_key_values, …]

        B, T, H = hidden.shape
        flat    = hidden.reshape(-1, H).float()
        with torch.no_grad():
            rec, _ = sae(flat)
        rec = rec.to(hidden.dtype).reshape(B, T, H)

        # 2) repack and return
        return (rec, *rest)
    return hook

THRESH = .3

def temp_scale(z, bad_idx, alpha=0.4):
    z[:, bad_idx] *= alpha
    return z



def make_hook_B(layer_id: int,bad_features, config='temp'):
    bad_idxs = torch.tensor(bad_features[layer_id], device="cuda" if torch.cuda.is_available() else "cpu")
    # print("HOOK CALLED ", bad_idxs.size())
    def hook(module, inputs, output):
        z = output.clone()
        for element in bad_idxs: 
            z[:, element] = torch.clamp(z[:, element], max=THRESH)
            return z
    def temp_hook(module, inputs, output):
        z = output.clone()
        for bad_idx in bad_idxs:
            z=temp_scale(z,bad_idx)
        return z
    def mean_hook(module, inputs, output):
        z = output.clone()
        for bad_idx in bad_idxs:
            z[:, bad_idx] = z.mean()
        return z
    if config == 'clamp':
        return hook
    elif config == 'temp':
        return temp_hook
    elif config == 'mean':
        return mean_hook
    
    return hook



def register_hooks(model,saes,LAYERS,bad_features):
    B_handles = []
    for lid, sae in saes.items():
        last_linear = sae.encoder[-1]               
        hB = last_linear.register_forward_hook(make_hook_B(lid,bad_features))
        B_handles.append(hB)
    A_handles = []
    for lid in LAYERS:
        block   = model.model.layers[lid]           
        hA      = block.register_forward_hook(make_hook_A(lid,saes))
        A_handles.append(hA)
    return B_handles,A_handles