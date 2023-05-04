"""Self Attention Module

Author: Zhao Na, 2020
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    def __init__(self, in_channel, out_channel=None, attn_dropout=0.1):
        """
        :param in_channel: previous layer's output feature dimension
        :param out_channel: size of output vector, defaults to in_channel
        """
        super(SelfAttention, self).__init__()
        self.in_channel = in_channel

        if out_channel is not None:
            self.out_channel = out_channel
        else:
            self.out_channel = in_channel

        self.temperature = self.out_channel ** 0.5
        
        self.q_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.k_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)
        self.v_map = nn.Conv1d(in_channel, self.out_channel, 1, bias=False)

        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, x):
        """
        :param x: the feature maps from previous layer,
                      shape: (batch_size, in_channel, num_points)
        :return: y: attentioned features maps,
                        shape： (batch_size, out_channel, num_points)
        """
        q = self.q_map(x)  # (batch_size, out_channel, num_points)
        k = self.k_map(x)  # (batch_size, out_channel, num_points)
        v = self.v_map(x)  # (batch_size, out_channel, num_points)

        attn = torch.matmul(q.transpose(1,2) / self.temperature, k)

        attn = self.dropout(F.softmax(attn, dim=-1))
        y = torch.matmul(attn, v.transpose(1,2)) # (batch_size, num_points, out_channel)

        return y.transpose(1,2)
        

class MultiHeadAttention(nn.Module):
    def __init__(self, in_channel, out_channel, n_classes=2, n_heads=1, att_dropout=0.1, use_proj=True):
        """
        :param in_channel: previous layer's output feature dimension
        :param out_channel: size of output vector, defaults to in_channel
        """
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        self.temperature = (out_channel // self.n_heads) ** 0.5
        self.n_classes = n_classes
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.use_proj = use_proj

        self.q_map = nn.Linear(self.in_channel, self.out_channel)
        self.k_map = nn.Linear(self.in_channel, self.out_channel)
        self.v_map = nn.Linear(self.in_channel, self.out_channel)
        self.dropout = nn.Dropout(att_dropout)

        if self.use_proj:
            self.proj = nn.Sequential(nn.Linear(self.out_channel//2, self.out_channe//2),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(self.out_channel//2, self.out_channel))

    def forward(self, x):
        """
        :param x: the feature maps from previous layer,
                      shape: (batch_size, in_channel, num_points)
        :param mask: shape(B, N, 1)
        :return: y: attentioned features maps,
                        shape： (batch_size, out_channel, num_points)
        """

        q, k, v = x

        B, N = q.shape[0], q.shape[1]

        q_res = q
        q = self.q_map(q)
        if q_res.size(-1) != q.size(-1):
            q_res = q
        k = self.k_map(k)
        v = self.v_map(v)

        q = q.reshape(B, N, self.n_heads, self.out_channel // self.n_heads).permute(0, 2, 1, 3)
        k = k.reshape(k.shape[0], k.shape[1], self.n_heads, self.out_channel // self.n_heads).permute(0, 2, 1, 3)
        v = v.reshape(v.shape[0], v.shape[1], self.n_heads, self.out_channel // self.n_heads).permute(0, 2, 1, 3)

        # [n_head, B, N, B*N]
        attn = torch.matmul(q, k.transpose(2, 3)) / self.temperature
        attn = self.dropout(F.softmax(attn, dim=-1))
        y = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, -1)

        if self.use_proj:
            y = self.proj(y)
            
        y = y + q_res
        return y
