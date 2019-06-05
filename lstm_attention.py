import torch
from torch import nn
import math
from attention import BasicAttention

class BasicLstmAttention(BasicAttention):
    '''
    输入是lstm的output，不用到q，其中k，v都是output
    来自论文Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
    '''
    def __init__(self, embd_size,q_k_hidden_size=1, head_size=1, score_func=None, is_q=False,
                 is_k=False, is_v=False):
        q_embd_size=embd_size
        k_embd_size=embd_size
        v_embd_size=embd_size
        v_hidden_size=embd_size
        is_q=True
        is_v=True
        score_func=self.score
        super(BasicLstmAttention,self).__init__(q_embd_size, k_embd_size, v_embd_size,q_k_hidden_size, v_hidden_size, head_size, score_func, is_q, is_k, is_v)
    def score(self,q,k):
        score=k.permute(0,2,1)
        # score = nn.functional.softmax(score, dim=-1)
        return score
    def forward(self, embd,mask=None):
        q_embd=embd
        k_embd=embd
        v_embd=embd
        return super(BasicLstmAttention,self).forward(q_embd, k_embd, v_embd,mask)

class LstmAttention_v1(BasicAttention):
    '''
        q:state
        k:outputs
        v:outputs
    '''
    def __init__(self, embd_size,q_k_hidden_size, head_size=1, score_func='scaled_dot', is_q=False,
                 is_k=False, is_v=False):
        q_embd_size=embd_size
        k_embd_size=embd_size
        v_embd_size=embd_size
        v_hidden_size=embd_size
        self.q_k_hidden_size=q_k_hidden_size
        super(LstmAttention_v1,self).__init__(q_embd_size, k_embd_size, v_embd_size,q_k_hidden_size, v_hidden_size, head_size, score_func, is_q, is_k, is_v)

    def forward(self, state,output,mask=None):
        q_embd=state
        k_embd=output
        v_embd=output
        return super(LstmAttention_v1,self).forward(q_embd, k_embd, v_embd,mask)