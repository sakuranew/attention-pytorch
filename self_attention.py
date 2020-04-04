import torch
from torch import nn
import math
from attention import BasicAttention

class SelfAttention(BasicAttention):
    def __init__(self, embd_size, num_heads=1, **kwargs):
        q_embd_size = embd_size
        k_embd_size = embd_size
        v_embd_size = embd_size
        super().__init__(q_embd_size, k_embd_size, v_embd_size, num_heads=num_heads, **kwargs)

    def forward(self, embd, mask=None):
        q_embd = embd
        k_embd = embd
        v_embd = embd
        return super().forward(q_embd, k_embd, v_embd, mask)
