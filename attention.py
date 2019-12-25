import math

import numpy as np
import torch
from torch import nn


def orignal(x):
    return x



class BasicAttention(nn.Module):
    def __init__(self,
                 q_embd_size,
                 k_embd_size,
                 v_embd_size,
                 q_k_hidden_size=None,
                 output_hidden_size=None,
                 num_heads=1,  # for multi-head attention
                 score_func='scaled_dot',
                 drop_rate=0.,
                 is_q=False,  # let q_embd to be q or not,default not
                 is_k=False,
                 is_v=False,
                 bias=True
                 ):
        '''

        :param q_embd_size:
        :param k_embd_size:
        :param v_embd_size:
        :param q_k_hidden_size:
        :param output_hidden_size:
        :param num_heads: for multi-head attention
        :param score_func:
        :param is_q: let q_embd to be q or not,default not
        :param is_k: let k_embd to be k or not,default not
        :param is_v: let v_embd to be v or not,default not
        :param bias: bias of linear
        '''
        super(BasicAttention, self).__init__()
        if not q_k_hidden_size:
            q_k_hidden_size = q_embd_size
        if not output_hidden_size:
            output_hidden_size = v_embd_size
        assert q_k_hidden_size % num_heads == 0
        self.head_dim = q_k_hidden_size // num_heads
        assert self.head_dim * num_heads == q_k_hidden_size, "q_k_hidden_size must be divisible by num_heads"
        assert output_hidden_size % num_heads == 0, "output_hidden_size must be divisible by num_heads"
        if is_q:
            self.q_w = orignal
            assert q_embd_size == k_embd_size
        else:
            self.q_w = nn.Linear(q_embd_size, q_k_hidden_size,bias=bias)
        self.is_q = is_q
        self.q_embd_size = q_embd_size
        if is_k:
            self.k_w = orignal
            assert k_embd_size == q_k_hidden_size
        else:
            self.k_w = nn.Linear(k_embd_size, q_k_hidden_size,bias=bias)
        if is_v:
            self.v_w = orignal
            assert v_embd_size == output_hidden_size
        else:
            self.v_w = nn.Linear(v_embd_size, output_hidden_size,bias=bias)
        self.q_k_hidden_size = q_k_hidden_size
        self.output_hidden_size = output_hidden_size
        self.num_heads = num_heads
        self.score_func = score_func
        self.drop_rate = drop_rate

    def forward(self, q_embd, k_embd, v_embd, mask=None):
        '''
        batch-first is needed
        :param q_embd: [?,q_len,q_embd_size] or [?,q_embd_size]
        :param k_embd: [?,k_len,k_embd_size] or [?,k_embd_size]
        :param v_embd: [?,v_len,v_embd_size] or [?,v_embd_size]
        :return: [?,q_len,output_hidden_size*num_heads]
        '''
        if len(q_embd.shape) == 2:
            q_embd = torch.unsqueeze(q_embd, 1)
        if len(k_embd.shape) == 2:
            k_embd = torch.unsqueeze(k_embd, 1)
        if len(v_embd.shape) == 2:
            v_embd = torch.unsqueeze(v_embd, 1)
        batch_size = q_embd.shape[0]
        q_len = q_embd.shape[1]
        k_len = k_embd.shape[1]
        v_len = v_embd.shape[1]
        #     make sure k_len==v_len
        assert k_len == v_len

        # get q,k,v
        if self.is_q:
            q = self.q_w(q_embd).view(batch_size, q_len, self.num_heads, self.q_embd_size // self.num_heads)
            q = q.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.q_embd_size // self.num_heads)
        else:
            q = self.q_w(q_embd).view(batch_size, q_len, self.num_heads, self.head_dim)
            q = q.permute(2, 0, 1, 3).contiguous().view(-1, q_len, self.head_dim)
        k = self.k_w(k_embd).view(batch_size, k_len, self.num_heads, self.head_dim)
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, k_len, self.head_dim)
        v = self.v_w(v_embd).view(batch_size, v_len, self.num_heads, self.output_hidden_size // self.num_heads)
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, v_len, self.output_hidden_size // self.num_heads)

        # get score
        if isinstance(self.score_func, str):
            if self.score_func == "dot":
                score = torch.bmm(q, k.permute(0, 2, 1))

            elif self.score_func == "scaled_dot":
                temp = torch.bmm(q, k.permute(0, 2, 1))
                score = torch.div(temp, math.sqrt(self.q_k_hidden_size))

            else:
                raise RuntimeError('invalid score function')
        elif callable(self.score_func):
            try:
                score = self.score_func(q, k)
            except Exception as e:
                print("Exception :", e)
        if mask is not None:
            mask = mask.bool().unsqueeze(1)
            score = score.masked_fill(~mask, -np.inf)
        score = nn.functional.softmax(score, dim=-1)
        score = nn.functional.dropout(score, p=self.drop_rate, training=self.training)

        # get output
        output = torch.bmm(score, v)
        heads = torch.split(output, batch_size)
        output = torch.cat(heads, -1)

        return output
