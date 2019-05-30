import torch
from torch import nn
import math
class BasicAttention(nn.Module):
    def __init__(self,
                 q_embd_size,
                 k_embd_size,
                 v_embd_size,
                 q_k_hidden_size,
                 v_hidden_size,
                 head_size=1,     # for multi-head attention
                 score_func='dot',
                 is_q=False,    # let q_embd to be q or not,default not
                 is_k=False,
                 is_v=False,
                 ):
        '''

        :param q_embd_size:
        :param k_embd_size:
        :param v_embd_size:
        :param q_k_hidden_size:
        :param v_hidden_size:
        :param head_size: for multi-head attention
        :param score_func:
        :param is_q: let q_embd to be q or not,default not
        :param is_k: let k_embd to be k or not,default not
        :param is_v: let v_embd to be v or not,default not
        '''
        if is_q:
            self.q_w=lambda x:x
            assert q_embd_size==q_k_hidden_size
        else:
            self.q_w=nn.Linear(q_embd_size,q_k_hidden_size)
        if is_k:
            self.k_w=lambda x:x
            assert k_embd_size==q_k_hidden_size
        else:
            self.k_w=nn.Linear(k_embd_size,q_k_hidden_size)
        if is_v:
            self.v_w=lambda x:x
            assert v_embd_size==v_hidden_size
        else:
            self.v_w=nn.Linear(v_embd_size,v_hidden_size)
        self.q_k_hidden_size=q_k_hidden_size
        self.v_hidden_size=v_hidden_size
        self.head_size=head_size
        self.score_func=score_func
    def forward(self, q_embd,k_embd,v_embd):
        '''
        batch-first is needed
        :param q_embd: [?,q_len,q_embd_size] or [?,q_embd_size]
        :param k_embd: [?,k_len,k_embd_size] or [?,k_embd_size]
        :param v_embd: [?,v_len,v_embd_size] or [?,v_embd_size]
        :return: [?,q_len,v_hidden_size*head_size]
        '''
        if len(q_embd.shape)==2:
            q_embd=torch.unsqueeze(q_embd,1)
        if len(k_embd.shape)==2:
            k_embd=torch.unsqueeze(k_embd,1)
        if len(v_embd.shape)==2:
            v_embd=torch.unsqueeze(v_embd,1)
        batch_size=q_embd.shape[0]
        q_len=q_embd.shape[1]
        k_len=k_embd.shape[1]
        v_len=v_embd.shape[1]
        #     make sure k_len==v_len
        assert k_len==v_len

        # get q,k,v
        q=self.q_w(q_embd).view(batch_size,q_len,self.head_size,self.q_k_hidden_size)
        q=q.permute(2,0,1,3).contiguous().view(-1,q_len,self.q_k_hidden_size)
        k=self.k_w(k_embd).view(batch_size,k_len,self.head_size,self.q_k_hidden_size)
        k=k.permute(2,0,1,3).contiguous().view(-1,k_len,self.q_k_hidden_size)
        v=self.v_w(v_embd).view(batch_size,v_len,self.head_size,self.v_hidden_size)
        v=v.permute(2,0,1,3).contiguous().view(-1,v_len,self.v_hidden_size)

        # get score
        if isinstance(self.score_func,str):
            if self.score_func=="dot":
                score=torch.bmm(q,k.permute(0,2,1))
            elif self.score_func=="scaled_dot":
                temp=torch.bmm(q,k.permute(0,2,1))
                score=torch.div(temp,math.sqrt(self.q_k_hidden_size))
            else:
                raise RuntimeError('invalid score function')
        elif isinstance(self.score_func,function):
            try:
                score=self.score_func(q,k)
            except Exception as e:
                print("Exception :",e)

        score=nn.functional.softmax(score,dim=-1)

        # get output
        output=torch.bmm(score,v)
        heads=torch.split(output,batch_size)
        output=torch.cat(heads,-1)

        return output
class SelfAttention(BasicAttention):
    def __init__(self, embd_size, q_k_hidden_size,v_hidden_size, head_size=1, score_func='dot', is_q=False,
                 is_k=False, is_v=False):
        q_embd_size=embd_size
        k_embd_size=embd_size
        v_embd_size=embd_size
        super().__init__(q_embd_size, k_embd_size, v_embd_size, q_k_hidden_size,v_hidden_size, head_size, score_func, is_q, is_k, is_v)

    def forward(self, embd):
        q_embd=embd
        k_embd=embd
        v_embd=embd
        return super().forward(q_embd, k_embd, v_embd)