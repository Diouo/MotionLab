import rotary_embedding_torch
import math

from typing import Optional
from rotary_embedding_torch import RotaryEmbedding

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

from diffusers.utils import deprecate
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, SwiGLU
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings

from rfmotion.models.operator.position_encoding import build_position_encoding
from rfmotion.models.operator.attention_processor import Attention
from rfmotion.utils.temos_utils import lengths_to_mask, lengths_to_query_mask
from rfmotion.models.operator.time_embed import TimestepEmbedderMDM, TimestepEmbedder
from rfmotion.models.tools.tools import split_list


def compute_boolean_matrix(matrix1, matrix2):
    matrix1 = matrix1.unsqueeze(2).to(torch.float)
    matrix2 = matrix2.unsqueeze(1).to(torch.float)
    attention_mask = torch.matmul(matrix1, matrix2).to(torch.bool)

    matirx3 = torch.eye(attention_mask.size(1), attention_mask.size(2), dtype=attention_mask.dtype, device=attention_mask.device).unsqueeze(0).repeat(attention_mask.shape[0],1,1)
    attention_mask = attention_mask | matirx3
    
    return attention_mask

def _chunked_feed_forward(ff: nn.Module, hidden_states: torch.Tensor, chunk_dim: int, chunk_size: int):
    # "feed_forward_chunk_size" can be used to save memory
    if hidden_states.shape[chunk_dim] % chunk_size != 0:
        raise ValueError(
            f"`hidden_states` dimension to be chunked: {hidden_states.shape[chunk_dim]} has to be divisible by chunk size: {chunk_size}. Make sure to set an appropriate `chunk_size` when calling `unet.enable_forward_chunking`."
        )

    num_chunks = hidden_states.shape[chunk_dim] // chunk_size
    ff_output = torch.cat(
        [ff(hid_slice) for hid_slice in hidden_states.chunk(num_chunks, dim=chunk_dim)],
        dim=chunk_dim,
    )
    return ff_output

def scaled_dot_product_attention(query, key, value, attn_mask=None, dropout_p=0.0,
        is_causal=False, scale=None, enable_gqa=False) -> torch.Tensor:
    B, H, L, S = query.size(0), query.size(1), query.size(-2), key.size(-2)
    scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(B, H, L, S, dtype=query.dtype, device=query.device)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_bias.masked_fill_(attn_mask.logical_not(), float("-inf"))
        else:
            attn_bias += attn_mask

    if enable_gqa:
        key = key.repeat_interleave(query.size(-3)//key.size(-3), -3)
        value = value.repeat_interleave(query.size(-3)//value.size(-3), -3)

    attn_weight = query @ key.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight = torch.softmax(attn_weight, dim=-1)
    attn_weight = torch.dropout(attn_weight, dropout_p, train=True)
    return attn_weight @ value

class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class JointAttnProcessor2_0:
    """Attention processor used typically in processing the SD3-like self-attention projections."""

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        source_motion: torch.FloatTensor = None,
        text: torch.FloatTensor = None,
        hint: torch.FloatTensor = None,
        style: torch.FloatTensor = None,
        content: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size = hidden_states.shape[0]
        inner_dim = int(hidden_states.shape[-1])
        head_dim = int(inner_dim // attn.heads)

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        query, key = attn.ROPE.rotate_queries_and_keys(query, key)

        # `source_motion` projections.
        if source_motion is not None:
            source_motion_query_proj = attn.source_q_proj(source_motion)
            source_motion_key_proj = attn.source_k_proj(source_motion)
            source_motion_value_proj = attn.source_v_proj(source_motion)

            source_motion_query_proj = source_motion_query_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            source_motion_key_proj = source_motion_key_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            source_motion_value_proj = source_motion_value_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_source_q is not None:
                source_motion_query_proj = attn.norm_source_q(source_motion_query_proj)
            if attn.norm_source_k is not None:
                source_motion_key_proj = attn.norm_source_k(source_motion_key_proj)
            source_motion_query_proj, source_motion_key_proj = attn.ROPE.rotate_queries_and_keys(source_motion_query_proj, source_motion_key_proj)

            query = torch.cat([source_motion_query_proj, query], dim=2)
            key = torch.cat([source_motion_key_proj, key,], dim=2)
            value = torch.cat([source_motion_value_proj, value], dim=2)
        
        # `hint` projections.
        if hint is not None:
            hint_query_proj = attn.hint_q_proj(hint)
            hint_key_proj = attn.hint_k_proj(hint)
            hint_value_proj = attn.hint_v_proj(hint)

            hint_query_proj = hint_query_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            hint_key_proj = hint_key_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            hint_value_proj = hint_value_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_hint_q is not None:
                hint_query_proj = attn.norm_hint_q(hint_query_proj)
            if attn.norm_hint_k is not None:
                hint_key_proj = attn.norm_hint_k(hint_key_proj)

            hint_query_proj, hint_key_proj = attn.ROPE.rotate_queries_and_keys(hint_query_proj, hint_key_proj)

            query = torch.cat([hint_query_proj, query], dim=2)
            key = torch.cat([hint_key_proj, key], dim=2)
            value = torch.cat([hint_value_proj, value], dim=2)

        # `text` projections.
        if text is not None:  
            text_query_proj = attn.text_q_proj(text)
            text_key_proj = attn.text_k_proj(text)
            text_value_proj = attn.text_v_proj(text)

            text_query_proj = text_query_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            text_key_proj = text_key_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            text_value_proj = text_value_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_text_q is not None:
                text_query_proj = attn.norm_text_q(text_query_proj)
            if attn.norm_text_k is not None:
                text_key_proj = attn.norm_text_k(text_key_proj)

            text_query_proj = text_query_proj + attn.q_pos_embed(text_query_proj, batch_first=True)
            text_key_proj = text_key_proj + attn.k_pos_embed(text_key_proj, batch_first=True)

            query = torch.cat([text_query_proj, query], dim=2)
            key = torch.cat([text_key_proj, key], dim=2)
            value = torch.cat([text_value_proj, value], dim=2)
        
        if style is not None:
            style_query_proj = attn.style_q_proj(style)
            style_key_proj = attn.style_k_proj(style)
            style_value_proj = attn.style_v_proj(style)

            style_query_proj = style_query_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            style_key_proj = style_key_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            style_value_proj = style_value_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_style_q is not None:
                style_query_proj = attn.norm_style_q(style_query_proj)
            if attn.norm_style_k is not None:
                style_key_proj = attn.norm_style_k(style_key_proj)

            style_query_proj = style_query_proj + attn.style_q_pos_embed(style_query_proj, batch_first=True)
            style_key_proj = style_key_proj + attn.style_k_pos_embed(style_key_proj, batch_first=True)

            query = torch.cat([style_query_proj, query], dim=2)
            key = torch.cat([style_key_proj, key], dim=2)
            value = torch.cat([style_value_proj, value], dim=2)

        if content is not None:
            content_query_proj = attn.content_q_proj(content)
            content_key_proj = attn.content_k_proj(content)
            content_value_proj = attn.content_v_proj(content)

            content_query_proj = content_query_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            content_key_proj = content_key_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            content_value_proj = content_value_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_content_q is not None:
                content_query_proj = attn.norm_content_q(content_query_proj)
            if attn.norm_content_k is not None:
                content_key_proj = attn.norm_content_k(content_key_proj)

            content_query_proj = content_query_proj + attn.content_q_pos_embed(content_query_proj, batch_first=True)
            content_key_proj = content_key_proj + attn.content_k_pos_embed(content_key_proj, batch_first=True)

            query = torch.cat([content_query_proj, query], dim=2)
            key = torch.cat([content_key_proj, key], dim=2)
            value = torch.cat([content_value_proj, value], dim=2)

        hidden_states = scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if content is not None:
            content, hidden_states = torch.split(hidden_states, [content.size(1), hidden_states.size(1) - content.size(1)], dim=1)
        if style is not None:
            style, hidden_states = torch.split(hidden_states, [style.size(1), hidden_states.size(1) - style.size(1)], dim=1)
        if text is not None:
            text, hidden_states = torch.split(hidden_states, [text.size(1), hidden_states.size(1) - text.size(1)], dim=1)
        if hint is not None:
            hint, hidden_states = torch.split(hidden_states, [hint.size(1), hidden_states.size(1) - hint.size(1)], dim=1)
        if source_motion is not None:
            source_motion, hidden_states = torch.split(hidden_states, [source_motion.size(1), hidden_states.size(1) - source_motion.size(1)], dim=1)

        if not attn.context_pre_only:
            if source_motion is not None:
                source_motion = attn.to_source_out(source_motion)
            if text is not None:
                text = attn.to_text_out(text)
            if hint is not None:
                hint = attn.to_hint_out(hint)
            if style is not None:
                style = attn.to_style_out(style)
            if content is not None:
                content = attn.to_content_out(content)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        return hidden_states, source_motion, hint, text, style, content

class JointTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        context_pre_only: bool = False,
        qk_norm: Optional[str] = None,
        activation_fn: str = "gelu-approximate",
    ):
        super().__init__()
        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"
        self.norm1 = AdaLayerNormZero(dim)

        if context_norm_type == "ada_norm_continous":
            self.norm1_context = AdaLayerNormContinuous(dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm")
            self.norm1_source = AdaLayerNormContinuous(dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm")
            self.norm1_hint = AdaLayerNormContinuous(dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm")
            self.norm1_style = AdaLayerNormContinuous(dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm")
            self.norm1_content = AdaLayerNormContinuous(dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm")
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context = AdaLayerNormZero(dim)
            self.norm1_source = AdaLayerNormZero(dim)
            self.norm1_hint = AdaLayerNormZero(dim)
            self.norm1_style = AdaLayerNormZero(dim)
            self.norm1_content = AdaLayerNormZero(dim)
        else:
            raise ValueError(f"Unknown context_norm_type: {context_norm_type}, currently only support `ada_norm_continous`, `ada_norm_zero`")

        if hasattr(F, "scaled_dot_product_attention"):
            processor = JointAttnProcessor2_0()
        else:
            raise ValueError("The current PyTorch version does not support the `scaled_dot_product_attention` function.")

        self.attn = Attention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=context_pre_only,
            bias=True,
            processor=processor,
            qk_norm=qk_norm,
            eps=1e-6,
        )
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn=activation_fn)

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn=activation_fn)
            self.norm2_source = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_source = FeedForward(dim=dim, dim_out=dim, activation_fn=activation_fn)
            self.norm2_hint = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_hint = FeedForward(dim=dim, dim_out=dim, activation_fn=activation_fn)
            self.norm2_style = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_style = FeedForward(dim=dim, dim_out=dim, activation_fn=activation_fn)
            self.norm2_content = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_content = FeedForward(dim=dim, dim_out=dim, activation_fn=activation_fn)
        else:
            self.norm2_context = None
            self.ff_context = None
            self.norm2_source = None
            self.ff_source = None
            self.norm2_hint = None
            self.ff_hint = None
            self.norm2_style = None
            self.ff_style = None
            self.norm2_content = None
            self.ff_content = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self, time: torch.FloatTensor, 
        hidden_states: torch.FloatTensor, target_mask: torch.BoolTensor, 
        text: torch.FloatTensor=None, text_mask: torch.BoolTensor=None,
        hint: torch.FloatTensor=None, hint_mask: torch.BoolTensor=None,
        style: torch.FloatTensor=None, style_mask: torch.BoolTensor=None,
        content: torch.FloatTensor=None, content_mask: torch.BoolTensor=None,
        source_motion: torch.FloatTensor=None, source_motion_mask: torch.BoolTensor=None, 
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=time)
        if self.context_pre_only:
            if text is not None:
                norm_text = self.norm1_context(text, time)
            else:
                norm_text = None
            if hint is not None:
                norm_hint = self.norm1_hint(hint, time)
            else:
                norm_hint = None
            if source_motion is not None:
                norm_source_motion = self.norm1_source(source_motion, time)
            else:
                norm_source_motion = None
            if style is not None:
                norm_style = self.norm1_style(style, time)
            else:
                norm_style = None
            if content is not None:
                norm_content = self.norm1_content(content, time)
            else:
                norm_content = None
        else:
            if text is not None:
                norm_text, t_gate_msa, t_shift_mlp, t_scale_mlp, t_gate_mlp = self.norm1_context(text, emb=time)
            else:
                norm_text = None
            if hint is not None:
                norm_hint, h_gate_msa, h_shift_mlp, h_scale_mlp, h_gate_mlp = self.norm1_hint(hint, emb=time)
            else:
                norm_hint = None
            if source_motion is not None:
                norm_source_motion, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = self.norm1_source(source_motion, emb=time)
            else:
                norm_source_motion = None
            if style is not None:
                norm_style, st_gate_msa, st_shift_mlp, st_scale_mlp, st_gate_mlp = self.norm1_style(style, emb=time)
            else:
                norm_style = None
            if content is not None:
                norm_content, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_content(content, emb=time)
            else:
                norm_content = None

        # Attention. 
        attention_mask = target_mask
        if source_motion is not None:
            attention_mask = torch.cat([source_motion_mask, attention_mask], dim=1)
        if hint is not None and hint_mask is not None:
            attention_mask = torch.cat([hint_mask, attention_mask,], dim=1)
        if text is not None and text_mask is not None:
            attention_mask = torch.cat([text_mask, attention_mask], dim=1)
        if style is not None and style_mask is not None:
            attention_mask = torch.cat([style_mask, attention_mask], dim=1)
        if content is not None and content_mask is not None:
            attention_mask = torch.cat([content_mask, attention_mask], dim=1)
        attention_mask = compute_boolean_matrix(attention_mask, attention_mask)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.attn.heads, 1, 1)
        attn_output, source_attn_output, hint_attn_output, text_attn_output, style_attn_output, content_attn_output = self.attn(
                                                                      hidden_states=norm_hidden_states, 
                                                                      source_motion=norm_source_motion, 
                                                                      text=norm_text,
                                                                      hint=norm_hint,
                                                                      style=norm_style,
                                                                      content=norm_content,
                                                                      attention_mask=attention_mask)

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        attn_output = hidden_states + attn_output
        norm_hidden_states2 = self.norm2(attn_output)
        norm_hidden_states2 = norm_hidden_states2 * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states2, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states2)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `source_motion`
        if not self.context_pre_only and source_motion is not None:
            source_attn_output = s_gate_msa.unsqueeze(1) * source_attn_output
            source_attn_output = source_motion + source_attn_output

            norm_source_motion2 = self.norm2_source(source_attn_output)
            norm_source_motion2 = norm_source_motion2 * (1 + s_scale_mlp[:, None]) + s_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                source_ff_output = _chunked_feed_forward(
                    self.ff_source, norm_source_motion2, self._chunk_dim, self._chunk_size
                )
            else:
                source_ff_output = self.ff_source(norm_source_motion2)
            source_motion = source_motion + s_gate_mlp.unsqueeze(1) * source_ff_output

        # Process attention outputs for the `text`.
        if not self.context_pre_only and text is not None:
            text_attn_output = t_gate_msa.unsqueeze(1) * text_attn_output
            text_attn_output = text + text_attn_output

            norm_text2 = self.norm2_context(text_attn_output)
            norm_text2 = norm_text2 * (1 + t_scale_mlp[:, None]) + t_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                text_ff_output = _chunked_feed_forward(self.ff_context, norm_text2, self._chunk_dim, self._chunk_size)
            else:
                text_ff_output = self.ff_context(norm_text2)
            text = text + t_gate_mlp.unsqueeze(1) * text_ff_output

        # Process attention outputs for the `hint`.
        if not self.context_pre_only and hint is not None:
            hint_attn_output = h_gate_msa.unsqueeze(1) * hint_attn_output
            hint_attn_output = hint + hint_attn_output

            norm_hint2 = self.norm2_hint(hint_attn_output)
            norm_hint2 = norm_hint2 * (1 + h_scale_mlp[:, None]) + h_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                hint_ff_output = _chunked_feed_forward(self.ff_hint, norm_hint2, self._chunk_dim, self._chunk_size)
            else:
                hint_ff_output = self.ff_hint(norm_hint2)
            hint = hint + h_gate_mlp.unsqueeze(1) * hint_ff_output

        # Process attention outputs for the `style`.
        if not self.context_pre_only and style is not None:
            style_attn_output = st_gate_msa.unsqueeze(1) * style_attn_output
            style_attn_output = style + style_attn_output

            norm_style2 = self.norm2_style(style_attn_output)
            norm_style2 = norm_style2 * (1 + st_scale_mlp[:, None]) + st_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                style_ff_output = _chunked_feed_forward(self.ff_style, norm_style2, self._chunk_dim, self._chunk_size)
            else:
                style_ff_output = self.ff_style(norm_style2)
            style = style + st_gate_mlp.unsqueeze(1) * style_ff_output

        # Process attention outputs for the `content`.
        if not self.context_pre_only and content is not None:
            content_attn_output = c_gate_msa.unsqueeze(1) * content_attn_output
            content_attn_output = content + content_attn_output

            norm_content2 = self.norm2_content(content_attn_output)
            norm_content2 = norm_content2 * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                content_ff_output = _chunked_feed_forward(self.ff_content, norm_content2, self._chunk_dim, self._chunk_size)
            else:
                content_ff_output = self.ff_content(norm_content2)
            content = content + c_gate_mlp.unsqueeze(1) * content_ff_output

        return hidden_states, source_motion, hint, text, style, content


class RFMotionDenoiser(nn.Module):

    def __init__(self,
                 ablation,
                 nfeats: int = 207,
                 token_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 4,
                 text_encoded_dim: int = 768,
                 **kwargs) -> None:

        super().__init__()
        self.token_dim = token_dim
        self.num_layers = num_layers
        self.text_encoded_dim = text_encoded_dim
        self.is_vae = ablation.VAE

        if self.is_vae:
            self.source_motion_embed = nn.Linear(self.token_dim, self.token_dim)
            self.noisy_motion_embed = nn.Linear(self.token_dim, self.token_dim)
            self.proj_out = nn.Linear(self.token_dim, self.token_dim, bias=True)
        else:
            self.source_motion_embed = nn.Linear(nfeats, self.token_dim)
            self.noisy_motion_embed = nn.Linear(nfeats, self.token_dim)
            self.proj_out = nn.Linear(self.token_dim, nfeats, bias=True)
        self.time_embed = TimestepEmbedder(self.token_dim)
        self.time_text_embed = CombinedTimestepTextProjEmbeddings(embedding_dim=self.token_dim, pooled_projection_dim=self.text_encoded_dim)

        self.context_embed = nn.Linear(text_encoded_dim, self.token_dim)

        self.hint_embed1 = nn.Linear(66, self.token_dim)
        # self.hint_embed2 = nn.SiLU()
        # self.hint_embed3 = nn.Linear(self.token_dim, self.token_dim)

        self.style_embed = nn.Linear(512, self.token_dim)
        self.content_embed1 = nn.InstanceNorm1d(256, affine=True)
        self.content_embed2 = nn.Linear(256, self.token_dim)

        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.token_dim,
                    num_attention_heads=num_heads,
                    attention_head_dim=int(self.token_dim/num_heads),
                    context_pre_only = i == num_layers - 1,
                    qk_norm="rms_norm",
                )
                for i in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(self.token_dim, self.token_dim, elementwise_affine=False, eps=1e-6)


    def forward(self, instructions, timestep, 
                hidden_states, target_lengths, target_lengths_z,
                source_motion=None, source_lengths=None,  source_lengths_z=None, source_mask=None,
                text=None, text_lengths=None,
                hint=None, hint_lengths=None,
                content=None, content_lengths=None,
                style=None, style_lengths=None, **kwargs): 


        # input embedding: time
        if len(timestep.shape) == 0:
            timestep = timestep.unsqueeze(0)
        if instructions is not None:
            temb = self.time_text_embed(timestep, instructions.to(hidden_states.device))
        else:
            temb = self.time_embed(timestep,dtype=hidden_states.dtype)

        # input embedding: text
        if text is not None:
            if len(text) == 2: # input both pooled text embedding and last hidden embedding, for MM-Dit
                text = self.context_embed(text[1])
            elif len(text) == 1: # only input both pooled text embedding
                text = self.context_embed(text[0])
            text_mask = lengths_to_mask(text_lengths, hidden_states.device, max_len=text.size(1)) 
        else:
            text_mask = None

        # input embedding: hint
        if hint is not None:
            hint = self.hint_embed1(hint)
            # hint = self.hint_embed2(hint)
            # hint = self.hint_embed3(hint)
            hint_mask = hint_lengths
        else:
            hint_mask = None  

        # input embedding: source_motion
        if source_motion is not None:
            source_motion = self.source_motion_embed(source_motion)
            if source_mask is not None:
                source_motion_mask = source_mask
            else:
                if not self.is_vae:
                    source_motion_mask = lengths_to_mask(source_lengths, hidden_states.device, max_len=source_motion.size(1))
                else:
                    source_motion_mask = lengths_to_mask(source_lengths_z, hidden_states.device, max_len=source_motion.size(1))
        else:
            source_motion_mask = None

        # input embedding: content
        if content is not None:
            content = self.content_embed1(content.permute(0,2,1)).permute(0,2,1)
            content = self.content_embed2(content)
            content_mask = lengths_to_mask(content_lengths, hidden_states.device, max_len=content.size(1))
        else:
            content_mask = None

        # input embedding: style
        if style is not None:
            style = self.style_embed(style)
            style_mask = lengths_to_mask(style_lengths, hidden_states.device, max_len=1)
        else:
            style_mask = None

        # input embedding: hidden_states
        hidden_states = self.noisy_motion_embed(hidden_states)
        if not self.is_vae:
            target_mask = lengths_to_mask(target_lengths, hidden_states.device, max_len=hidden_states.size(1))
        else:
            target_mask = lengths_to_mask(target_lengths_z, hidden_states.device, max_len=hidden_states.size(1))

        for i, block in enumerate(self.transformer_blocks):
            # if hidden_states is not None:
            #     print("target",hidden_states.size(), target_mask.size())
            # if text is not None:
            #     print("text",text.size(), text_mask.size())
            # if style is not None:
            #     print("style",style.size(), style_mask.size())
            # if hint is not None:
            #     print("hint",hint.size(), hint_mask.size())
            # if content is not None:
            #     print("content",content.size(), content_mask.size())
            # if source_motion is not None:
            #     print("source_motion",source_motion.size(), source_motion_mask.size())

            
            # if i>8:
            #     hidden_states, _, _, _, _ = block(time=temb, hidden_states=hidden_states, target_mask=target_mask,)
            # else:
            
            hidden_states, source_motion, hint, text, style, content, = block(time=temb, hidden_states=hidden_states, target_mask=target_mask, 
                                                                            text=text, text_mask=text_mask,
                                                                            hint=hint, hint_mask=hint_mask,
                                                                            content=content, content_mask=content_mask,
                                                                            style=style, style_mask=style_mask,
                                                                            source_motion=source_motion, source_motion_mask=source_motion_mask,)

        # # input into U-Net backbone
        # hidden_states_list = []
        # for index_block, block in enumerate(self.transformer_blocks):
        #     if index_block in self.pre_index_list:
        #         hidden_states, source_motion, hint, text, = block(time=temb, hidden_states=hidden_states, target_mask=target_mask, 
        #                                                                     text=text, text_mask=text_mask,
        #                                                                     hint=hint, hint_mask=hint_mask,
        #                                                                     source_motion=source_motion, source_motion_mask=source_motion_mask,)
        #         hidden_states_list.append(hidden_states)
        #     elif index_block == self.mid_index_list:
        #         hidden_states, source_motion, hint, text, = block(time=temb, hidden_states=hidden_states, target_mask=target_mask, 
        #                                                                     text=text, text_mask=text_mask,
        #                                                                     hint=hint, hint_mask=hint_mask,
        #                                                                     source_motion=source_motion, source_motion_mask=source_motion_mask,)
        #     elif index_block in self.post_index_list:
        #         hidden_states = torch.cat((hidden_states, hidden_states_list.pop()),-1) 
        #         hidden_states = self.skip_connect_norm[index_block-math.ceil(self.num_layers/2)](hidden_states)
        #         hidden_states = self.skip_connect_proj[index_block-math.ceil(self.num_layers/2)](hidden_states)
        #         hidden_states, source_motion, hint, text, = block(time=temb, hidden_states=hidden_states, target_mask=target_mask, 
        #                                                                     text=text, text_mask=text_mask,
        #                                                                     hint=hint, hint_mask=hint_mask,
        #                                                                     source_motion=source_motion, source_motion_mask=source_motion_mask,)

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        return (hidden_states,)
