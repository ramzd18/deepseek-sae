import torch
def make_hook_A(layer_id: int,saes):
    sae = saes[layer_id]                       

    def hook(module, inputs, output):          
        B, T, H = output.shape
        flat    = output.reshape(-1, H).float()          
        with torch.no_grad():
            rec, _ = sae(flat)                             
        rec = rec.to(output.dtype).reshape(B, T, H)        
        return rec                                         
    return hook


BAD_FEATURES = {}
THRESH = .3
def make_hook_B(layer_id: int):

    bad_idxs = torch.tensor(BAD_FEATURES[layer_id], device="cuda" if torch.cuda.is_available() else "cpu")
    def hook(module, inputs, output):            
        z = output.clone()  
        for element in bad_idxs:                   
            z[:, element] = torch.clamp(z[:, element], max=THRESH)
            return z
    return hook



def register_hooks(model,saes,LAYERS):
    B_handles = []
    for lid, sae in saes.items():
        last_linear = sae.encoder[-1]               
        hB = last_linear.register_forward_hook(make_hook_B(lid))
        B_handles.append(hB)
    A_handles = []
    for lid in LAYERS:
        block   = model.model.layers[lid]           
        hA      = block.register_forward_hook(make_hook_A(lid,saes))
        A_handles.append(hA)
    return B_handles,A_handles