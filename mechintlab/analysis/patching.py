
import torch

def patch_head_output(block, src_out, head_idx: int):
    # block.attn output shape: (B, T, D). We assume heads are concatenated.
    d = block.attn.o.in_features
    n_heads = block.attn.qkv.weight.shape[0] // (3*d)
    head_dim = d // n_heads

    def hook_fn(module, inp, out):
        out = out.clone()
        start = head_idx * head_dim
        end = (head_idx + 1) * head_dim
        out[:, :, start:end] = src_out[:, :, start:end].to(out.device)
        return out

    return block.attn.o.register_forward_hook(hook_fn)
