import torch
import torch.nn as nn
import math

class TreeMultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable structural bias parameters for parent/child relationships
        self.parent_bias = nn.Parameter(torch.randn(1, num_heads, 1, 1))
        self.child_bias = nn.Parameter(torch.randn(1, num_heads, 1, 1))

    def forward(self, x, parent_ids=None, child_ids=None, attention_mask=None):
        batch_size, seq_len, _ = x.size()
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Rebuild structural bias injection
        if parent_ids is not None and child_ids is not None:
            # Construct relational matrix matching [batch, heads, seq, seq]
            p_mask = (parent_ids.unsqueeze(-1) == parent_ids.unsqueeze(-2)).unsqueeze(1).float()
            c_mask = (child_ids.unsqueeze(-1) == child_ids.unsqueeze(-2)).unsqueeze(1).float()
            scores = scores + (p_mask * self.parent_bias) + (c_mask * self.child_bias)
            
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        return self.out_proj(context)

class TreeEncoderLayer(nn.Module):
    def __init__(self, hidden_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.mha = TreeMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, parent_ids=None, child_ids=None, attention_mask=None):
        attn_out = self.mha(x, parent_ids, child_ids, attention_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class TreeEncoder(nn.Module):
    def __init__(self, vocab_size, hidden_dim=512, nhead=8, num_layers=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([TreeEncoderLayer(hidden_dim, nhead) for _ in range(num_layers)])
        
    def forward(self, input_ids, parent_ids=None, child_ids=None, attention_mask=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = layer(x, parent_ids, child_ids, attention_mask)
        return x
