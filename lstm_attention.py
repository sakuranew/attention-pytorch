import torch
from torch import nn
import math
from attention import BasicAttention

class AttentionQKV(BasicAttention):
    '''
        q:state
        k:outputs
        v:outputs
    '''

    def __init__(self, embd_size, q_k_hidden_size, num_heads=1, score_func='scaled_dot', **kwargs):
        q_embd_size = embd_size
        k_embd_size = embd_size
        v_embd_size = embd_size
        output_hidden_size = embd_size
        self.q_k_hidden_size = q_k_hidden_size
        super(AttentionQKV, self).__init__(q_embd_size, k_embd_size, v_embd_size, q_k_hidden_size, output_hidden_size,
                                           num_heads, score_func, **kwargs)

    def forward(self, state, output, mask=None):
        q_embd = state
        k_embd = output
        v_embd = output
        return super(AttentionQKV, self).forward(q_embd, k_embd, v_embd, mask)
