from typing import Optional

import numpy as np
import torch
import torch.nn as nn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k: int):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k

    def forward(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, attn_mask: Optional[torch.Tensor]=None) -> tuple[torch.Tensor, torch.Tensor]:
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        scores = torch.exp(scores)
        if attn_mask is not None:
            scores = scores * attn_mask
        attn = scores / (torch.sum(scores, dim=-1, keepdim=True) + 1e-8)

        context = torch.matmul(attn, V)
        return context, attn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_attention_heads: int):
        super(MultiHeadSelfAttention, self).__init__()
        self.d_model = d_model
        self.num_attention_heads = num_attention_heads
        assert d_model % num_attention_heads == 0
        self.d_k = d_model // num_attention_heads
        self.d_v = d_model // num_attention_heads

        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=1)

    def forward(
        self,
        Q: torch.Tensor,
        K: Optional[torch.Tensor] = None,
        V: Optional[torch.Tensor] = None,
        length: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if K is None:
            K = Q
        if V is None:
            V = Q
        batch_size = Q.size(0)

        q_s = (
            self.W_Q(Q)
            .view(batch_size, -1, self.num_attention_heads, self.d_k)
            .transpose(1, 2)
        )
        k_s = (
            self.W_K(K)
            .view(batch_size, -1, self.num_attention_heads, self.d_k)
            .transpose(1, 2)
        )
        v_s = (
            self.W_V(V)
            .view(batch_size, -1, self.num_attention_heads, self.d_v)
            .transpose(1, 2)
        )

        if length is not None:
            maxlen = Q.size(1)
            attn_mask = torch.arange(maxlen).to(device).expand(
                batch_size, maxlen
            ) < length.to(device).view(-1, 1)
            attn_mask = attn_mask.unsqueeze(1).expand(batch_size, maxlen, maxlen)
            attn_mask = attn_mask.unsqueeze(1).repeat(1, self.num_attention_heads, 1, 1)
        else:
            attn_mask = None

        context, _ = ScaledDotProductAttention(self.d_k)(q_s, k_s, v_s, attn_mask)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_attention_heads * self.d_v)
        )
        return context
