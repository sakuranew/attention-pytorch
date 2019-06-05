import torch
from torch import nn
import math
from attention import BasicAttention

class SelfAttention(BasicAttention):
    def __init__(self, embd_size, q_k_hidden_size,v_hidden_size, head_size=1, score_func='scaled_dot', is_q=False,
                 is_k=False, is_v=False):
        q_embd_size=embd_size
        k_embd_size=embd_size
        v_embd_size=embd_size
        super().__init__(q_embd_size, k_embd_size, v_embd_size, q_k_hidden_size,v_hidden_size, head_size, score_func, is_q, is_k, is_v)

    def forward(self, embd,mask=None):
        q_embd=embd
        k_embd=embd
        v_embd=embd
        return super().forward(q_embd, k_embd, v_embd,mask)