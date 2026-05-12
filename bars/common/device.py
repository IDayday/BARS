from __future__ import annotations
import os

def get_torch_device(prefer: str = 'cuda'):
    import torch
    if prefer == 'cuda' and torch.cuda.is_available(): return torch.device('cuda')
    return torch.device('cpu')

def describe_visible_cuda() -> str:
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    return 'CUDA_VISIBLE_DEVICES=<not set>' if visible is None else f'CUDA_VISIBLE_DEVICES={visible}'
