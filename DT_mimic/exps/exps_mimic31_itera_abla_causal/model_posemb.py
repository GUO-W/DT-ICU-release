import torch
from torch import nn
import torch.nn.functional as F
# import numpy as np
import math
# from parameters import *

class DTmodel(nn.Module): # Transformer our model v1.0
    def __init__(
        self,
        data_config,
        embed_size,
        latent_size,
        pred,
        guidance_scale=100.0,
        eps=1e-6,
        norm_data=False,
    ):
        super(DTmodel, self).__init__()

        self.stat_vocab_size = data_config.stat_vocab_size
        self.proc_vocab_size = data_config.proc_vocab_size
        self.med_vocab_size = data_config.med_vocab_size
        self.out_vocab_size = data_config.out_vocab_size
        self.chart_vocab_size = data_config.chart_vocab_size
        self.date_vocab_size = data_config.date_vocab_size
        self.ing_vocab_size = data_config.ing_vocab_size
        # self.eth_vocab_size = data_config.eth_vocab_size
        # self.gender_vocab_size = data_config.gender_vocab_size
        # self.age_vocab_size = data_config.age_vocab_size
        # self.ins_vocab_size = data_config.ins_vocab_size
        # self.modalities = data_config.modalities
        self.embed_size = embed_size
        self.latent_size = latent_size
        self.pred = pred
        self.norm_data = norm_data

        self.atten_embed_size = 0
        self.cross_embed_size = 0
        self.pred_size = 0

        if self.med_vocab_size:
            self.med = nn.Linear(self.med_vocab_size, self.latent_size)
            self.atten_embed_size += self.latent_size
            self.pred_size += self.med_vocab_size
        if self.chart_vocab_size:
            self.chart = nn.Linear(self.chart_vocab_size, self.latent_size)
            self.atten_embed_size += self.latent_size
            self.pred_size += self.chart_vocab_size
        if self.out_vocab_size:
            self.out = nn.Linear(self.out_vocab_size, self.latent_size)
            self.atten_embed_size += self.latent_size
            self.pred_size += self.out_vocab_size
        if self.proc_vocab_size:
            self.proc = nn.Linear(self.proc_vocab_size, self.latent_size)
            self.atten_embed_size += self.latent_size
            self.pred_size += self.proc_vocab_size
        if self.date_vocab_size:
            self.date = nn.Linear(self.date_vocab_size, self.latent_size)
            self.atten_embed_size += self.latent_size
            self.pred_size += self.date_vocab_size
        if self.ing_vocab_size:
            self.ing = nn.Linear(self.ing_vocab_size, self.latent_size)
            self.atten_embed_size += self.latent_size
            self.pred_size += self.ing_vocab_size
        if self.stat_vocab_size:
            self.stat = nn.Linear(self.stat_vocab_size, self.latent_size)
            self.cross_embed_size += self.latent_size

        self.atten0 = TransformerBlock(self.atten_embed_size)
        self.adapter = nn.Linear(
            self.latent_size * 6 + self.cross_embed_size, self.atten_embed_size
        )
        # self.atten1 = TransformerBlock(self.atten_embed_size)
        self.cross_attn = CrossAttention(self.atten_embed_size)
        self.guidance_scale = guidance_scale
        # self.global_embedding = nn.Parameter(torch.randn(1, 1, self.atten_embed_size))
        self.fc = nn.Linear(self.atten_embed_size, 1)
        if self.pred:
            self.fc2 = nn.Linear(self.atten_embed_size, self.pred_size)
        self.eps = eps

    def set_mean_std(self, mean, std):
        # "meds", "chart", "out", "proc", "date", "ing", "stat"
        self.med_mean = mean["meds"]
        self.med_std = std["meds"]

        self.chart_mean = mean["chart"]
        self.chart_std = std["chart"]

        self.out_mean = mean["out"]
        self.out_std = std["out"]

        self.proc_mean = mean["proc"]
        self.proc_std = std["proc"]

        self.date_mean = mean["date"]
        self.date_std = std["date"]

        self.ing_mean = mean["ing"]
        self.ing_std = std["ing"]

        self.stat_mean = mean["stat"]
        self.stat_std = std["stat"]
        self.norm_data = True

    def reshape_pred(self, pred):
        shapes = torch.tensor([0, self.med_vocab_size, self.chart_vocab_size, self.out_vocab_size, self.proc_vocab_size, self.date_vocab_size, self.ing_vocab_size])
        shapes = torch.cumsum(shapes, dim=0)
        # means = [self.med_mean, self.chart_mean, self.out_mean, self.proc_mean, self.date_mean, self.ing_mean]
        # stds = [self.med_std, self.chart_std, self.out_std, self.proc_std, self.date_std, self.ing_std]
        outputs = []
        for i in range(len(shapes) - 1):
            ins = pred[:, shapes[i] : shapes[i+1]].unsqueeze(1)
            # ins = ins * stds[i] + means[i]
            outputs.append(ins)
        return outputs

    def forward(self, meds, chart, out, proc, date, ing, stat, demo, key_padding_mask):
        out1 = torch.zeros(size=(0, 0))

        if self.med_vocab_size:
            if self.norm_data:
                meds = (meds - self.med_mean) / (self.med_std + self.eps)
            medEmbedded = self.med(meds)
            if out1.nelement():
                out1 = torch.cat((out1, medEmbedded), 2)
            else:
                out1 = medEmbedded
        if self.chart_vocab_size:
            if self.norm_data:
                chart = (chart - self.chart_mean) / (self.chart_std + self.eps)
            chartEmbed = self.chart(chart)
            if out1.nelement():
                out1 = torch.cat((out1, chartEmbed), 2)
            else:
                out1 = chartEmbed
        if self.out_vocab_size:
            if self.norm_data:
                out = (out - self.out_mean) / (self.out_std + self.eps)
            outEmbedded = self.out(out)
            if out1.nelement():
                out1 = torch.cat((out1, outEmbedded), 2)
            else:
                out1 = outEmbedded
        if self.proc_vocab_size:
            if self.norm_data:
                proc = (proc - self.proc_mean) / (self.proc_std + self.eps)
            procEmbedded = self.proc(proc)
            if out1.nelement():
                out1 = torch.cat((out1, procEmbedded), 2)
            else:
                out1 = procEmbedded
        if self.date_vocab_size:
            if self.norm_data:
                date = (date - self.date_mean) / (self.date_std + self.eps)
            dateEmbedded = self.date(date)
            if out1.nelement():
                out1 = torch.cat((out1, dateEmbedded), 2)
            else:
                out1 = dateEmbedded
        if self.ing_vocab_size:
            if self.norm_data:
                ing = (ing - self.ing_mean) / (self.ing_std + self.eps)
            ingEmbedded = self.ing(ing)
            if out1.nelement():
                out1 = torch.cat((out1, ingEmbedded), 2)
            else:
                out1 = ingEmbedded

        # Self attention
        # bs = out1.shape[0]
        # global_embedding = self.global_embedding.repeat(bs, 1, 1)
        # out1 = torch.cat([out1, global_embedding], 1)
        out1 = self.atten0(out1, out1, out1, key_padding_mask=key_padding_mask)

        gender = timestep_embedding(
            demo[:, 0] * self.guidance_scale, self.latent_size
        ).unsqueeze(1)
        out2 = gender

        eth = timestep_embedding(
            demo[:, 1] * self.guidance_scale, self.latent_size
        ).unsqueeze(1)
        out2 = torch.cat((out2, eth), 2)

        ins = timestep_embedding(
            demo[:, 2] * self.guidance_scale, self.latent_size
        ).unsqueeze(1)
        out2 = torch.cat((out2, ins), 2)

        age = timestep_embedding(
            demo[:, 3] * self.guidance_scale, self.latent_size
        ).unsqueeze(1)
        out2 = torch.cat((out2, age), 2)

        admission = timestep_embedding(
            demo[:, 4] * self.guidance_scale, self.latent_size
        ).unsqueeze(1)
        out2 = torch.cat((out2, admission), 2)

        icu = timestep_embedding(
            demo[:, 5] * self.guidance_scale, self.latent_size
        ).unsqueeze(1)
        out2 = torch.cat((out2, icu), 2)

        if self.stat_vocab_size:
            if self.norm_data:
                stat = (stat - self.stat_mean) / (self.stat_std + self.eps)
            stat = self.stat(stat).unsqueeze(1)
            out2 = torch.cat((out2, stat), 2)

        out2 = self.adapter(out2)
        # out2 = self.atten1(out2, out2, out2)
        # Cross attention for feature merging
        out = self.cross_attn(out2, out1, out1, key_padding_mask=key_padding_mask).squeeze(1)

        logit = self.fc(out)

        if self.pred:
            pred = self.fc2(out)
            pred = self.reshape_pred(pred)
            out = F.sigmoid(logit)
            return out, logit, pred
        else:
            out = F.sigmoid(logit)
            return out, logit, None


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, learnable=True):
        super(PositionalEncoding, self).__init__()

        if learnable:
            # Learnable positional embedding
            self.pos_embedding = nn.Parameter(torch.zeros(1, max_len, d_model))
            nn.init.normal_(self.pos_embedding, mean=0, std=0.1)
        else:
            # Sinusoidal positional embedding
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
            )
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            pe = pe.unsqueeze(0)
            self.register_buffer("pos_embedding", pe)

        self.learnable = learnable

    def forward(self, x):
        if self.learnable:
            return x + self.pos_embedding[:, : x.size(1), :]
        else:
            return x + self.pos_embedding[:, : x.size(1), :].detach()


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class TransformerBlock(nn.Module):
    def __init__(
        self,
        embed_size,
        heads=8,
        dropout=0.1,
        forward_expansion=4,
        max_seq_len=1000,
        pos_emb_type="rope",  # Options: "learnable", "sinusoidal", "rope", "none"
        causal=False,
    ):
        super(TransformerBlock, self).__init__()
        self.embed_size = embed_size
        self.pos_emb_type = pos_emb_type
        self.causal = causal

        # Setup positional embedding based on type.
        if pos_emb_type == "learnable":
            # Learnable positional embeddings of shape (1, max_seq_len, embed_size)
            self.positional_embedding = nn.Parameter(torch.zeros(1, max_seq_len, embed_size))
        elif pos_emb_type == "sinusoidal":
            # Precompute sinusoidal embeddings (not learnable)
            pos_embedding = self._get_sinusoidal_positional_embedding(max_seq_len, embed_size)
            self.register_buffer("positional_embedding", pos_embedding)
        elif pos_emb_type == "rope":
            # ROPE does not need a positional parameter; transformation is applied directly.
            pass
        elif pos_emb_type == "none":
            # No positional embeddings will be used.
            pass
        else:
            raise ValueError(f"Unsupported pos_emb_type: {pos_emb_type}")

        # Multi-head attention with batch_first set to True (PyTorch 1.9+)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_size, num_heads=heads, dropout=dropout, batch_first=True
        )
        self.q_proj = nn.Linear(embed_size, embed_size)
        self.q_norm = RMSNorm(embed_size)
        self.k_proj = nn.Linear(embed_size, embed_size)
        self.k_norm = RMSNorm(embed_size)
        self.v_proj = nn.Linear(embed_size, embed_size)

        self.norm1 = RMSNorm(embed_size)
        self.norm2 = RMSNorm(embed_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.dropout = nn.Dropout(dropout)

    def _get_sinusoidal_positional_embedding(self, max_seq_len, embed_size):
        """
        Create sinusoidal positional embeddings as described in the original Transformer paper.
        Returns a tensor of shape (1, max_seq_len, embed_size).
        """
        pe = torch.zeros(max_seq_len, embed_size)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2, dtype=torch.float) * (-math.log(10000.0) / embed_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape (1, max_seq_len, embed_size)
        return pe

    @staticmethod
    def _apply_rope(x, base=10000):
        """
        Applies Rotary Positional Embeddings (ROPE) to tensor x.
        x is expected to have shape (batch_size, seq_len, embed_size) with embed_size even.
        This function splits the last dimension into two halves and applies the rotation.
        """
        batch_size, seq_len, embed_size = x.size()
        half_dim = embed_size // 2

        # Create scaling factors.
        dim_range = torch.arange(half_dim, device=x.device, dtype=x.dtype)
        theta = 1.0 / (base ** (2 * dim_range / embed_size))

        # Compute positional indices.
        pos = torch.arange(seq_len, device=x.device, dtype=x.dtype).unsqueeze(1)  # shape: (seq_len, 1)
        freqs = pos * theta.unsqueeze(0)  # shape: (seq_len, half_dim)
        cos = freqs.cos().unsqueeze(0)  # shape: (1, seq_len, half_dim)
        sin = freqs.sin().unsqueeze(0)  # shape: (1, seq_len, half_dim)

        # Split the tensor along the last dimension.
        x1 = x[:, :, :half_dim]
        x2 = x[:, :, half_dim:]
        # Apply rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        return torch.cat([rotated_x1, rotated_x2], dim=-1)

    def _generate_causal_mask(self, seq_len, device):
        """
        Generate a causal mask of shape (seq_len, seq_len) where positions in the upper triangle are masked.
        """
        # True indicates positions that should be masked.
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
        return causal_mask

    def forward(self, query, key, value, mask=None, key_padding_mask=None):
        """
        Input tensors are expected to be of shape (batch_size, seq_len, embed_size).
        """
        batch_size, seq_len, _ = query.size()
        query = self.q_norm(self.q_proj(query))
        key = self.k_norm(self.k_proj(key))
        value = self.v_proj(value)

        if self.pos_emb_type == "learnable" or self.pos_emb_type == "sinusoidal":
            pos_embeds = self.positional_embedding[:, :seq_len, :]
            query = query + pos_embeds
            key = key + pos_embeds
            value = value + pos_embeds
        elif self.pos_emb_type == "rope":
            query = self._apply_rope(query)
            key = self._apply_rope(key)
            # value remains unchanged.
        # If pos_emb_type is "none", do nothing.

        # If causal masking is enabled, generate the mask and combine with any provided mask.
        if self.causal:
            causal_mask = self._generate_causal_mask(seq_len, query.device)
            if mask is not None:
                mask = mask | causal_mask
            else:
                mask = causal_mask

        # Apply multi-head attention.
        attention_output, _ = self.attention(query, key, value, attn_mask=mask, key_padding_mask=key_padding_mask)
        x = self.dropout(self.norm1(attention_output + query))
        forward_output = self.feed_forward(x)
        out = self.dropout(self.norm2(forward_output + x))
        return out


class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=8, forward_expansion=4):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.q_norm = RMSNorm(embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.k_norm = RMSNorm(embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.norm1 = RMSNorm(embed_dim)
        self.norm2 = RMSNorm(embed_dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, forward_expansion * embed_dim),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_dim, embed_dim),
        )

    def forward(self, query, key, value, mask=None, key_padding_mask=None, query_padding_mask=None):
        B, N, C = query.shape
        _, S, _ = key.shape

        # Linear projections
        Q = self.q_proj(query)  # (B, N, C)
        Q = self.q_norm(Q)
        K = self.k_proj(key)  # (B, S, C)
        K = self.k_norm(K)
        V = self.v_proj(value)  # (B, S, C)

        # Split into heads
        Q = Q.view(B, N, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, num_heads, N, head_dim)
        K = K.view(B, S, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, num_heads, S, head_dim)
        V = V.view(B, S, self.num_heads, self.head_dim).transpose(
            1, 2
        )  # (B, num_heads, S, head_dim)

        # Scaled dot-product attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (
            self.head_dim**0.5
        )  # (B, num_heads, N, S)

        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))

        if key_padding_mask is not None:
            # key_padding_mask : (B, S)  True/1 = PAD
            pad = key_padding_mask.unsqueeze(1).unsqueeze(1)       # (B,1,1,S)
            attention_scores = attention_scores.masked_fill(pad, float("-inf"))

        attention_weights = F.softmax(attention_scores, dim=-1)  # (B, num_heads, N, S)

        # Apply attention weights to value
        attention_output = torch.matmul(
            attention_weights, V
        )  # (B, num_heads, N, head_dim)
        attention_output = (
            attention_output.transpose(1, 2).contiguous().view(B, N, self.embed_dim)
        )  # (B, N, C)

        # Output projection
        output = self.out_proj(attention_output)  # (B, N, C)
        # 6. (optional) zero‑out output rows that came from padded queries ----
        if query_padding_mask is not None:
            out = out.masked_fill(query_padding_mask.unsqueeze(-1), 0.0)

        x = self.norm1(output + query)
        output = self.feed_forward(x)
        output = self.norm2(output + x)
        return output

