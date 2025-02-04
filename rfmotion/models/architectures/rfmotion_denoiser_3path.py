import rotary_embedding_torch
import math

from typing import Optional
from rotary_embedding_torch import RotaryEmbedding

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

from diffusers.utils import deprecate
from diffusers.models.embeddings import CombinedTimestepTextProjEmbeddings
from diffusers.models.activations import GEGLU, GELU, ApproximateGELU, SwiGLU
from diffusers.models.normalization import AdaLayerNormContinuous, AdaLayerNormZero

from rfmotion.models.operator.position_encoding import build_position_encoding
from rfmotion.models.operator.attention_processor import Attention
from rfmotion.utils.temos_utils import lengths_to_mask, lengths_to_query_mask
from rfmotion.models.operator.time_embed import TimestepEmbedderMDM
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
        self.q_pos_embed = build_position_encoding(int(512/8), position_embedding='learned', embedding_dim="1D")
        self.k_pos_embed = build_position_encoding(int(512/8), position_embedding='learned', embedding_dim="1D")
        self.ROPE = RotaryEmbedding(int(512/8), use_xpos = True)
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        source_motion: torch.FloatTensor = None,
        encoder_hidden_states: torch.FloatTensor = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        batch_size = hidden_states.shape[0]
        self.q_pos_embed = self.q_pos_embed.to(hidden_states.device)
        self.k_pos_embed = self.k_pos_embed.to(hidden_states.device)
        self.ROPE = self.ROPE.to(hidden_states.device)

        # `sample` projections.
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        inner_dim = int(key.shape[-1])
        head_dim = int(inner_dim // attn.heads)

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        query, key = self.ROPE.rotate_queries_and_keys(query, key)

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
            source_motion_query_proj, source_motion_key_proj = self.ROPE.rotate_queries_and_keys(source_motion_query_proj, source_motion_key_proj)

            query = torch.cat([source_motion_query_proj, query], dim=2)
            key = torch.cat([source_motion_key_proj, key,], dim=2)
            value = torch.cat([source_motion_value_proj, value], dim=2)

        # `context` projections.
        if encoder_hidden_states is not None:
            encoder_hidden_states_query_proj = attn.text_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.text_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.text_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

            if attn.norm_text_q is not None:
                encoder_hidden_states_query_proj = attn.norm_text_q(encoder_hidden_states_query_proj)
            if attn.norm_text_k is not None:
                encoder_hidden_states_key_proj = attn.norm_text_k(encoder_hidden_states_key_proj)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj + self.q_pos_embed(encoder_hidden_states_query_proj, batch_first=True)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj + self.k_pos_embed(encoder_hidden_states_key_proj, batch_first=True)

            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)

        hidden_states = scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)
        # hidden_states = F.scaled_dot_product_attention(query, key, value, dropout_p=0.0, is_causal=False)
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if source_motion is not None and encoder_hidden_states is not None:
            # Split the attention outputs.
            encoder_hidden_states, source_motion,  hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] : encoder_hidden_states.shape[1] + source_motion.shape[1]],
                hidden_states[:,  encoder_hidden_states.shape[1] + source_motion.shape[1]:],
            )
            if not attn.context_pre_only:
                source_motion = attn.to_source_out(source_motion)
                encoder_hidden_states = attn.to_text_out(encoder_hidden_states)
        elif source_motion is None and encoder_hidden_states is not None:
            # Split the attention outputs.
            encoder_hidden_states, hidden_states,  = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_text_out(encoder_hidden_states)
        elif source_motion is not None and encoder_hidden_states is None:
            # Split the attention outputs.
            source_motion, hidden_states = (
                hidden_states[:, : source_motion.shape[1]],
                hidden_states[:, source_motion.shape[1] :],
            )
            if not attn.context_pre_only:
                source_motion = attn.to_source_out(source_motion)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if source_motion is not None and encoder_hidden_states is not None:
            return hidden_states, source_motion, encoder_hidden_states
        elif source_motion is None and encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        elif source_motion is not None and encoder_hidden_states is None:
            return hidden_states, source_motion
        else:
            return hidden_states


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
    ):
        super().__init__()
        self.context_pre_only = context_pre_only
        context_norm_type = "ada_norm_continous" if context_pre_only else "ada_norm_zero"
        self.norm1 = AdaLayerNormZero(dim)

        if context_norm_type == "ada_norm_continous":
            self.norm1_context = AdaLayerNormContinuous(dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm")
            self.norm1_source = AdaLayerNormContinuous(dim, dim, elementwise_affine=False, eps=1e-6, bias=True, norm_type="layer_norm")
        elif context_norm_type == "ada_norm_zero":
            self.norm1_context = AdaLayerNormZero(dim)
            self.norm1_source = AdaLayerNormZero(dim)
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
        self.ff = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")

        if not context_pre_only:
            self.norm2_context = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_context = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
            self.norm2_source = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
            self.ff_source = FeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate")
        else:
            self.norm2_context = None
            self.ff_context = None
            self.norm2_source = None
            self.ff_source = None

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    # Copied from diffusers.models.attention.BasicTransformerBlock.set_chunk_feed_forward
    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self, hidden_states: torch.FloatTensor, encoder_hidden_states: torch.FloatTensor, temb: torch.FloatTensor,
        source_motion: torch.FloatTensor, source_mask: torch.BoolTensor, target_mask: torch.BoolTensor, text_mask: torch.BoolTensor,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)
        if self.context_pre_only:
            norm_encoder_hidden_states = self.norm1_context(encoder_hidden_states, temb)
            norm_source_motion = self.norm1_source(source_motion, temb)
        else:
            norm_encoder_hidden_states, c_gate_msa, c_shift_mlp, c_scale_mlp, c_gate_mlp = self.norm1_context(encoder_hidden_states, emb=temb)
            norm_source_motion, s_gate_msa, s_shift_mlp, s_scale_mlp, s_gate_mlp = self.norm1_source(source_motion, emb=temb)

        # Attention. 
        attention_mask = torch.cat([text_mask, source_mask, target_mask], dim=1)
        attention_mask = compute_boolean_matrix(attention_mask, attention_mask)
        attention_mask = attention_mask.unsqueeze(1).repeat(1, self.attn.heads, 1, 1)
        attn_output, source_attn_output, context_attn_output = self.attn(hidden_states=norm_hidden_states, source_motion=norm_source_motion, encoder_hidden_states=norm_encoder_hidden_states, attention_mask=attention_mask)

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        attn_output = hidden_states + attn_output
        norm_hidden_states = self.norm2(attn_output)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output
        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `source_motion`
        if self.context_pre_only:
            source_motion = None
        else:
            source_attn_output = s_gate_msa.unsqueeze(1) * source_attn_output
            source_attn_output = source_motion + source_attn_output

            norm_source_motion = self.norm2_source(source_attn_output)
            norm_source_motion = norm_source_motion * (1 + s_scale_mlp[:, None]) + s_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                source_ff_output = _chunked_feed_forward(
                    self.ff_source, norm_source_motion, self._chunk_dim, self._chunk_size
                )
            else:
                source_ff_output = self.ff_source(norm_source_motion)
            source_motion = source_motion + s_gate_mlp.unsqueeze(1) * source_ff_output

        # Process attention outputs for the `encoder_hidden_states`.
        if self.context_pre_only:
            encoder_hidden_states = None
        else:
            context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
            context_attn_output = encoder_hidden_states + context_attn_output

            norm_encoder_hidden_states = self.norm2_context(context_attn_output)
            norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
            if self._chunk_size is not None:
                # "feed_forward_chunk_size" can be used to save memory
                context_ff_output = _chunked_feed_forward(
                    self.ff_context, norm_encoder_hidden_states, self._chunk_dim, self._chunk_size
                )
            else:
                context_ff_output = self.ff_context(norm_encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output

        return hidden_states, source_motion, encoder_hidden_states


class RFMotionDenoiser(nn.Module):

    def __init__(self,
                 ablation,
                 latent_dim: list = [1, 256],
                 num_layers: int = 6,
                 num_heads: int = 4,
                 text_encoded_dim: int = 768,
                 **kwargs) -> None:

        super().__init__()

        self.latent_dim = latent_dim[-1]
        self.num_layers = num_layers
        self.text_encoded_dim = text_encoded_dim
        self.pe_type = ablation.RF_PE_TYPE
        self.pe_dim = ablation.RF_PE_DIM
        self.is_vae = ablation.VAE
        self.time_embed_type = ablation.RF_TIMEEMBED

        if self.is_vae:
            self.source_motion_embed = nn.Linear(self.latent_dim, self.latent_dim)
            self.noisy_motion_embed = nn.Linear(self.latent_dim, self.latent_dim)
            self.proj_out = nn.Linear(self.latent_dim, self.latent_dim, bias=True)
        else:
            self.source_motion_embed = nn.Linear(207, self.latent_dim)
            self.noisy_motion_embed = nn.Linear(207, self.latent_dim)
            self.proj_out = nn.Linear(self.latent_dim, 207, bias=True)

        if self.time_embed_type == 'time':
            self.time_embed = TimestepEmbedderMDM((self.latent_dim))
        elif self.time_embed_type == 'time_text':
            self.time_embed = CombinedTimestepTextProjEmbeddings(pooled_projection_dim=text_encoded_dim, embedding_dim=self.latent_dim)

        self.pre_index_list, self.mid_index_list, self.post_index_list = split_list(list(range(num_layers)))
        self.skip_connect_norm = nn.ModuleList([nn.LayerNorm(2*self.latent_dim, elementwise_affine=False, eps=1e-6) for i in range(math.floor(num_layers/2))])
        self.skip_connect_proj = nn.ModuleList([nn.Linear(2*self.latent_dim, self.latent_dim) for i in range(math.floor(num_layers/2))])

        self.context_embed = nn.Linear(text_encoded_dim, self.latent_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                JointTransformerBlock(
                    dim=self.latent_dim,
                    num_attention_heads=num_heads,
                    attention_head_dim=int(self.latent_dim/num_heads),
                    # qk_norm="rms_norm",
                    context_pre_only=i == num_layers - 1,
                )
                for i in range(num_layers)
            ]
        )
        self.norm_out = AdaLayerNormContinuous(self.latent_dim, self.latent_dim, elementwise_affine=False, eps=1e-6)


    def forward(self, hidden_states, encoder_hidden_states, encoder_hidden_states_lengths, timestep,
                source_motion, source_lengths, target_lengths, source_lengths_z, target_lengths_z, **kwargs): 
        batch_nums, output_frames = hidden_states.size(0), hidden_states.size(1)

        # input embedding
        if len(timestep.shape) == 0:
            timestep = timestep.unsqueeze(0)
        if self.time_embed_type == "time":
            temb = self.time_embed(timestep)
        elif self.time_embed_type == "time_text":
            temb = self.time_embed(timestep, encoder_hidden_states[0].squeeze(1)) 
        if len(encoder_hidden_states) == 2: # input both pooled text embedding and last hidden embedding, for MM-Dit
            encoder_hidden_states = self.context_embed(encoder_hidden_states[1])
        elif len(encoder_hidden_states) == 1: # only input both pooled text embedding
            encoder_hidden_states = self.context_embed(encoder_hidden_states[0]) 
        source_motion = self.source_motion_embed(source_motion)
        hidden_states = self.noisy_motion_embed(hidden_states)

        # for mask
        if not self.is_vae:
            source_mask = lengths_to_mask(source_lengths, hidden_states.device, max_len=source_motion.size(1))
            target_mask = lengths_to_mask(target_lengths, hidden_states.device, max_len=hidden_states.size(1))
            text_mask = lengths_to_mask(encoder_hidden_states_lengths, hidden_states.device, max_len=encoder_hidden_states.size(1))
            # text_mask = torch.ones((encoder_hidden_states.size(0), encoder_hidden_states.size(1)), device=encoder_hidden_states.device, dtype=torch.bool)
        else:
            _, source_mask = lengths_to_query_mask(source_lengths, source_lengths_z, hidden_states.device, max_len=source_motion.size(1))
            _, target_mask = lengths_to_query_mask(target_lengths, target_lengths_z, hidden_states.device, max_len=hidden_states.size(1))
            text_mask = lengths_to_mask(encoder_hidden_states_lengths, hidden_states.device, max_len=encoder_hidden_states.size(1))
            # text_mask = torch.ones((encoder_hidden_states.size(0), encoder_hidden_states.size(1)), device=encoder_hidden_states.device, dtype=torch.bool)

        # for index_block, block in enumerate(self.transformer_blocks):
        #     hidden_states, source_motion, encoder_hidden_states = block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb,
        #                                                 source_motion=source_motion, source_mask=source_mask, target_mask=target_mask, text_mask=text_mask)

        # input into U-Net backbone
        hidden_states_list = []
        for index_block, block in enumerate(self.transformer_blocks):
            if index_block in self.pre_index_list:
                hidden_states, source_motion, encoder_hidden_states  = block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb,
                                                  source_motion=source_motion, source_mask=source_mask, target_mask=target_mask, text_mask=text_mask)
                hidden_states_list.append(hidden_states)
            elif index_block == self.mid_index_list:
                hidden_states, source_motion, encoder_hidden_states  = block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb,
                                                  source_motion=source_motion, source_mask=source_mask, target_mask=target_mask, text_mask=text_mask)
            elif index_block in self.post_index_list:
                hidden_states = torch.cat((hidden_states, hidden_states_list.pop()),-1) 
                hidden_states = self.skip_connect_norm[index_block-math.ceil(self.num_layers/2)](hidden_states)
                hidden_states = self.skip_connect_proj[index_block-math.ceil(self.num_layers/2)](hidden_states)
                hidden_states, source_motion, encoder_hidden_states  = block(hidden_states=hidden_states, encoder_hidden_states=encoder_hidden_states, temb=temb,
                                                  source_motion=source_motion, source_mask=source_mask, target_mask=target_mask, text_mask=text_mask)

        hidden_states = self.norm_out(hidden_states, temb)
        hidden_states = self.proj_out(hidden_states)

        return (hidden_states,)
