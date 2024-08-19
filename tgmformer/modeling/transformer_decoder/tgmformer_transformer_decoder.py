# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine
from .maskformer_transformer_decoder import TRANSFORMER_DECODER_REGISTRY

import numpy as np
import einops
import enum
import math

import matplotlib.pyplot as plt
import os
import time

class Discriminator(nn.Module):
    def __init__(self,context=False):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(256, 512, kernel_size=1, stride=1,
                  padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 512, kernel_size=1, stride=1,
                  padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(512, 256, kernel_size=1, stride=1,
                  padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 128, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 1, kernel_size=1, stride=1,
                               padding=0, bias=False)
        self.context = context
        self.relu = nn.ReLU(inplace=True)
        self._init_weights()
    def _init_weights(self):
      for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.relu(self.bn4(self.conv4(x)))
        x = self.conv5(x)
        return torch.sigmoid(x)

class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2, attn_weights = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt, attn_weights

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2, attn_weights = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        
        return tgt, attn_weights

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        # import ipdb; ipdb.set_trace()
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 , attn_weights = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        # import ipdb; ipdb.set_trace()
        tgt = self.norm(tgt)

        return tgt, attn_weights

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        # print(f'normalize_before {self.normalize_before}---------------------------------------------------------------------------------------------')
        if self.normalize_before: # [default: false]
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
def local_attention_imp2_2_featurev2(affinity, v, res, spix_idx_tensor_downsample, assign_eps=1e-16):
    # TODO: test this function
    def explicit_broadcast(this, other):
        # Append singleton dimensions until this.dim() == other.dim()
        for _ in range(this.dim(), other.dim()):
            this = this.unsqueeze(-1)

        # Explicitly expand so that shapes are the same
        return this.expand_as(other)

    B, neighbor, H, W = affinity.shape
    V_H, V_W, channels = v.shape[-3], v.shape[-2], v.shape[-1]
    n_spixl_h, n_spixl_w = res

    affinity_downsample = F.interpolate(
        affinity,
        size=(V_H, V_W),
        mode="bilinear",
        align_corners=False
    )

    # [1, 9, V_H, V_W]
    pixel_idx_tensor_downsample = torch.arange(0, V_H * V_W).unsqueeze(0).unsqueeze(0).expand(1, 9, -1, -1).to(affinity.device)
    pixel_idx_tensor_downsample = pixel_idx_tensor_downsample.reshape(9 * V_H * V_W)
    # TODO: handle the border

    # lift src node features
    v = v.reshape(B, V_H * V_W, channels) # [B, V_H * V_W, channels]
    # import ipdb; ipdb.set_trace()
    nodes_features_proj_lifted = v.index_select(1, pixel_idx_tensor_downsample) # [B, 9*V_H*V_W, channels]

    # neighborhood aware softmax
    affinity_downsample = affinity_downsample.reshape(B, 9*V_H*V_W)
    exp_scores_per_edge = affinity_downsample.exp() # softmax # [B, 9*V_H*V_W]

    # calculate the denominator. cluster-wise norm. shape = (B, V_H*V_W)
    # in cluster wise style, the norm is performed between all clusters which are connected to a same pixel
    neighborhood_sums_cluster_wise = torch.zeros((B, V_H*V_W), dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
    pixel_idx_tensor_downsample = pixel_idx_tensor_downsample.unsqueeze(0).expand(B, -1) # [B, 9 * V_H * V_W]
    neighborhood_sums_cluster_wise.scatter_add_(1, pixel_idx_tensor_downsample, exp_scores_per_edge)  

    # shape (B, V_H*V_W) -> (B, 9*V_H*V_W)   neigborhood_aware_denominator_cluster_wise得
    neigborhood_aware_denominator_cluster_wise = neighborhood_sums_cluster_wise.index_select(1, pixel_idx_tensor_downsample[0])
    attentions_per_edge_cluster_wise = exp_scores_per_edge / (neigborhood_aware_denominator_cluster_wise + assign_eps)


    # calculate pixel wise softmax
    neighborhood_sums_pixel_wise = torch.zeros((B, n_spixl_h*n_spixl_w + 1), dtype=exp_scores_per_edge.dtype, device=exp_scores_per_edge.device)
    # import ipdb; ipdb.set_trace()
    spix_idx_tensor_downsample = spix_idx_tensor_downsample.reshape(B, 9*V_H*V_W)
    neighborhood_sums_pixel_wise.scatter_add_(1, spix_idx_tensor_downsample, attentions_per_edge_cluster_wise)
    neighborhood_sums_pixel_wise[:, -1] = 0.0
    neigborhood_aware_denominator_pixel_wise = neighborhood_sums_pixel_wise.index_select(1, spix_idx_tensor_downsample[0])
    attentions_per_edge_pixel_wise = attentions_per_edge_cluster_wise / (neigborhood_aware_denominator_pixel_wise + assign_eps)
    # print("attentions_per_edge_pixel_wise.shape:", attentions_per_edge_pixel_wise.shape)
    # print("attentions_per_edge_pixel_wise", attentions_per_edge_pixel_wise)
    attentions_per_edge_pixel_wise = attentions_per_edge_pixel_wise.unsqueeze(-1)
    # E = 9*V_H*V_W
    # [B, 9*V_H*V_W, channels] * [B, 9*V_H*V_W, 1] -> [B, 9*V_H*V_W, channels]
    nodes_features_proj_lifted_weighted = nodes_features_proj_lifted * attentions_per_edge_pixel_wise

    # aggregate neighbors, shape [B, n_spixl_h*n_spixl_w, channels]
    out_nodes_features = torch.zeros((B, n_spixl_h*n_spixl_w + 1, channels), dtype=v.dtype, device=v.device)

    trg_index_broadcasted = explicit_broadcast(spix_idx_tensor_downsample, nodes_features_proj_lifted_weighted)

    out_nodes_features.scatter_add_(1, trg_index_broadcasted, nodes_features_proj_lifted_weighted)

    out_nodes_features = out_nodes_features[:, :n_spixl_h*n_spixl_w]

    return out_nodes_features


def cluster(q_feature, k_feature, v_feature, spix_res, spix_idx_tensor_downsample):
    """
        In this version, we keep q, k in the same embedding space, and return both the average of k and v
        In this version, we directly pass num_group queries to this function, so that we can iteratively perform
        super pixel clustering
        This variation, similar to the region proxy, use shallow layers to learn the affinity,
        and use high-level features for classification
        input: q_feature: (B, num_groups, C), k_feature: (B, C, H, W), v_feature: (B, C, V_H, V_W)
        output: attn_cluster_norm: (B, num_groups, H, W)
                attn_pixel_norm: (B, num_groups, H, W)
                out: (B, num_groups, C)
                affinity: (B, 9, H, W)
    """
    tau=0.07
    n_spixl_h, n_spixl_w, Hh, Ww = spix_res

    # cluster center features q with shape (B, num_groups, C)
    V_H, V_W = v_feature.shape[-2:]
    v = v_feature.permute(0, 2, 3, 1) # [B, V_H, V_W, C]
    # v = self.v_proj(v_feature)# [B, V_H, V_W, C]

    k_feature_downsample = F.interpolate(k_feature,
                                            size=(V_H, V_W),
                                            mode="bilinear",
                                            align_corners=False
                                            )
    # [B, C, V_H, V_W]->[B, V_H, V_W, C]
    k_feature_downsample = k_feature_downsample.permute(0, 2, 3, 1)

    # (B, C, H, W) -> (B, H, W, C)
    k_feature = k_feature.permute(0, 2, 3, 1)
    # import ipdb; ipdb.set_trace()
    # k_proj = self.k_proj(k_feature)
    B, H, W, C = k_feature.shape
    k_feature = k_feature.reshape(B, H * W, C).unsqueeze(-1) # (B, H*W, C, 1)

    # q_proj = self.q_proj(q_feature)
    q_feature = q_feature.reshape(B, n_spixl_h, n_spixl_w, C).permute(0, 3, 1, 2)

    # im2col with padding: (B, C, n_spixl_h, n_spixl_w)->(B, C, 9, n_spixl_h, n_spixl_w)
    q = F.unfold(q_feature, kernel_size=3, padding=1).reshape(B, -1, 9, n_spixl_h, n_spixl_w)

    q = F.interpolate(q.reshape(B, -1, n_spixl_h, n_spixl_w), size=(H, W), mode='nearest')
    q = q.reshape(B, -1, 9, H, W).permute(0, 3, 4, 1, 2).reshape(B, H * W, C, 9)

    # TODO: in the next version, we may
    norm_q = F.normalize(q, dim=-2)
    norm_k = F.normalize(k_feature, dim=-2) # (B, H*W, C, 1)

    affinity = norm_k.transpose(-2, -1) @ norm_q  # (B, H*W, 1, 9)

    affinity = affinity.permute(0, 3, 1, 2).reshape(B, 9, H, W)

    # handle borders 
    affinity = affinity.reshape(B, 9, n_spixl_h, Hh,
                                n_spixl_w, Ww)
    affinity = einops.rearrange(affinity, 'B n H h W w -> B n h w H W')
    affinity[:, :3, :, :, 0, :] = float('-inf')  # top
    affinity[:, -3:, :, :, -1, :] = float('-inf')  # bottom
    affinity[:, ::3, :, :, :, 0] = float('-inf')  # left
    affinity[:, 2::3, :, :, :, -1] = float('-inf')  # right

    affinity = affinity / tau
    # raw affinity matrix
    affinity = einops.rearrange(affinity, 'B n h w H W -> B n (H h) (W w)')  # (B, 9, H, W)

    assert v.shape == k_feature_downsample.shape
    out_cls_features = local_attention_imp2_2_featurev2(affinity, v, (n_spixl_h, n_spixl_w), spix_idx_tensor_downsample)

    # TODO: try to use high-resolution k_feature to update query features
    out_query_features = local_attention_imp2_2_featurev2(affinity, k_feature_downsample, (n_spixl_h, n_spixl_w), spix_idx_tensor_downsample)

    return out_query_features, out_cls_features, affinity

def convert_affinity_to_hard_assignment(affinity, spix_idx_tensor_downsample):
    """
    Convert the soft affinity matrix to a hard assignment using the global superpixel indices.
    
    affinity: the computed affinity matrix of shape (B, 9, H, W)
    spix_idx_tensor_downsample: tensor containing global superpixel indices for each position and pixel
    """
    B, _, H, W = affinity.shape
    # Apply softmax to the affinities to get probabilities
    probs = F.softmax(affinity, dim=1)
    
    # Get the local indices of the highest probabilities
    _, local_indices = probs.max(dim=1)
    # Gather the global superpixel indices using local indices
    hard_assignments = torch.gather(spix_idx_tensor_downsample, 1, local_indices.unsqueeze(1)).squeeze(1)
    
    # Check the shape and reshape if necessary
    if hard_assignments.shape != (B, H, W):
        hard_assignments = hard_assignments.view(B, H, W)
    
    return hard_assignments

class PositionEmbeddingSineWShape(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x_shape, x_device, mask=None):
        if mask is None:
            mask = torch.zeros((x_shape[0], x_shape[2], x_shape[3]), device=x_device, dtype=torch.bool)
        # mask: e.g. shape [2, 16, 32], [B, H, W]
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)  # [B, H, W]
        x_embed = not_mask.cumsum(2, dtype=torch.float32)  # [B, H, W]
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale  # normalize the coordinates, then multiply 2pi
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x_device)  # [128]
        # import ipdb; ipdb.set_trace()
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # [B, H, W, num_pos_feats]
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)  # [B, H, W, num_pos_feats]
        # import ipdb; ipdb.set_trace()
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        # import ipdb; ipdb.set_trace()
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1,
                                                       2)  # [B, 2*num_pos_feats, H, W], 2 * num_pos_feats is equal to the number of feat channels
        return pos

    def __repr__(self, _repr_indent=4):
        head = "Positional encoding " + self.__class__.__name__
        body = [
            "num_pos_feats: {}".format(self.num_pos_feats),
            "temperature: {}".format(self.temperature),
            "normalize: {}".format(self.normalize),
            "scale: {}".format(self.scale),
        ]
        # _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
    
@TRANSFORMER_DECODER_REGISTRY.register()
class TGMTransformerDecoder(nn.Module):

    _version = 2

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        version = local_metadata.get("version", None)
        if version is None or version < 2:
            # Do not warn if train from scratch
            scratch = True
            logger = logging.getLogger(__name__)
            for k in list(state_dict.keys()):
                newk = k
                if "static_query" in k:
                    newk = k.replace("static_query", "query_feat")
                if newk != k:
                    state_dict[newk] = state_dict[k]
                    del state_dict[k]
                    scratch = False

            if not scratch:
                logger.warning(
                    f"Weight format of {self.__class__.__name__} have changed! "
                    "Please upgrade your models. Applying automatic conversion now ..."
                )

    @configurable
    def __init__(
        self,
        in_channels,
        mask_classification=True,
        *,
        num_classes: int,
        hidden_dim: int,
        num_queries: int,
        nheads: int,
        dim_feedforward: int,
        dec_layers: int,
        pre_norm: bool,
        mask_dim: int,
        enforce_input_project: bool,
        vis: bool,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()
        self.vis = vis

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)

        self.num_queries = num_queries
        # learnable query features
        self.query_feat = nn.Embedding(num_queries, hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.discriminator = Discriminator()
        self.linear_fuse = nn.Conv2d(
            in_channels=hidden_dim * 3,
            out_channels=hidden_dim,
            kernel_size=1
        )
        self.contrastive_proj = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        self.downsample_rate = 16
        self.pe_layer1 = PositionEmbeddingSineWShape(N_steps, normalize=True)

    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification
        
        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MASK_FORMER.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD

        # NOTE: because we add learnable query features which requires supervision,
        # we add minus 1 to decoder layers to be consistent with our loss
        # implementation: that is, number of auxiliary losses is always
        # equal to number of decoder layers. With learnable query features, the number of
        # auxiliary losses equals number of decoders plus 1.
        assert cfg.MODEL.MASK_FORMER.DEC_LAYERS >= 1
        ret["dec_layers"] = cfg.MODEL.MASK_FORMER.DEC_LAYERS - 1
        ret["pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MASK_FORMER.ENFORCE_INPUT_PROJ

        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["vis"] = cfg.MODEL.MASK_FORMER.VIS

        return ret

    def feature_fusion(self, features):
        x = []
        # Detach the first feature
        x.append(features[0].detach())
        # Detach other features and interpolate
        for i in range(1, len(features)):
            feature_detached = features[i].detach()  # Detach each feature before processing
            x.append(F.interpolate(feature_detached, scale_factor=np.power(2, i), mode="bilinear", align_corners=False))

        # Concatenate and pass through linear fusion layer
        x = self.linear_fuse(torch.cat(x, dim=1))
        return x

    
    def forward(self, x, mask_features, source = False, mask = None):
        # x: list of multi-scale features,
        # shape of x[i]: [B, C, h, w]
        x.reverse()  # from high to low resolution
        x_feature = self.feature_fusion(x)

        B, C, H, W = x_feature.shape
        ori_mask_H, ori_mask_W = mask_features.shape[-2], mask_features.shape[-1]

        q_k_features = mask_features.reshape(B, C, ori_mask_H * ori_mask_W).permute(0, 2, 1).detach()
        q_k_features = self.contrastive_proj(q_k_features) # [B, mask_H*mask_W, C]  (MLP, the same channel)
        q_k_features = q_k_features.reshape(B, ori_mask_H, ori_mask_W, C).permute(0, 3, 1, 2) # [B, C, mask_H, mask_W]
        spix_k_feature = F.avg_pool2d(q_k_features, (2, 2), (2, 2))
        mask_H, mask_W = spix_k_feature.shape[-2], spix_k_feature.shape[-1]

        if mask_H % self.downsample_rate or mask_W % self.downsample_rate:
            self.downsample_rate = 2
        n_spixl_h = int(np.floor(mask_H / self.downsample_rate))
        n_spixl_w = int(np.floor(mask_W / self.downsample_rate))

        spix_res = (n_spixl_h, n_spixl_w, self.downsample_rate, self.downsample_rate)

        spix_values = torch.arange(0, n_spixl_h * n_spixl_w).reshape(n_spixl_h, n_spixl_w).to(mask_features.device)
        spix_values = spix_values.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)

        p2d = (1, 1, 1, 1)
        spix_values = F.pad(spix_values, p2d, "constant", n_spixl_h * n_spixl_w)
        spix_idx_tensor_ = F.unfold(spix_values.type(mask_features.dtype), kernel_size=3).reshape(B, 1 * 9, n_spixl_h,
                                                                                             n_spixl_w).int()


        # target index
        spix_idx_tensor = F.interpolate(spix_idx_tensor_.type(mask_features.dtype), size=(mask_H, mask_W),
                                        mode='nearest').long()  # (B, 9, H, W)

        spix_idx_tensor_downsample = F.interpolate(spix_idx_tensor_.type(mask_features.dtype), size=(H, W),
                                                   mode='nearest').long()

        # src index
        # [B, 9, H, W]
        pixel_idx_tensor = torch.arange(0, mask_H * mask_W).unsqueeze(0).unsqueeze(0).expand(B, 9, -1, -1).to(mask_features.device)
        pixel_idx_tensor = pixel_idx_tensor.reshape(B, 9, mask_H, mask_W)

        x_shape = [B, C, n_spixl_h, n_spixl_w]
        x_device = x_feature.device
        pos = self.pe_layer1(x_shape, x_device)
        pos = pos.reshape(B, C, n_spixl_h * n_spixl_w).permute(0, 2, 1)
        output_spix = F.avg_pool2d(spix_k_feature, (self.downsample_rate, self.downsample_rate), (self.downsample_rate, self.downsample_rate))
        output_spix = output_spix.reshape(B, C, n_spixl_h * n_spixl_w).permute(0, 2, 1)
        spix_v_feature = x_feature
        affinity_list = []
        for i in range(6):
            output_spix, output_cls, affinity = cluster(output_spix, spix_k_feature, spix_v_feature, spix_res, spix_idx_tensor_downsample)
            affinity_list.append(affinity)
        output_cls = output_cls.permute(0, 2, 1)  
        output_cls = output_cls.contiguous().view(B, C, n_spixl_h, n_spixl_w)
        d_pixel = self.discriminator(output_cls)
        
        hard_assignments = convert_affinity_to_hard_assignment(affinity_list[-1], spix_idx_tensor_downsample)
        if source:
            return {'d_pixel': d_pixel, 'hard_assignments': hard_assignments}
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []
        # print('x[0].shape', x[0].shape) # torch.Size([2, 256, 16, 32])
        # print('x[2].shape', x[2].shape) # torch.Size([2, 256, 64, 128])
        # print('mask_features.shape', mask_features.shape) # torch.Size([2, 256, 128, 256])
        # disable mask, it does not affect performance
        del mask
        # import ipdb; ipdb.set_trace()
   
        for i in range(self.num_feature_levels):
            size_list.append(x[i].shape[-2:])
            pos.append(self.pe_layer(x[i], None).flatten(2))
            s = self.input_proj[i](x[i]).flatten(2)
            src.append(s + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []

        # prediction heads on learnable query features
        # import ipdb; ipdb.set_trace()
        w = d_pixel.detach()
        outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        attn_mask, mask_to_update = self.weighted_attention_mask(outputs_mask=outputs_mask, attn_mask_target_size=size_list[0], d_pixel=d_pixel, hard_assignments=hard_assignments)
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)
        
        flag = 0
        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
            # attention: cross-attention first
            output, cross_weights= self.transformer_cross_attention_layers[i](
                output, src[level_index],
                memory_mask=attn_mask,
                memory_key_padding_mask=None,  # here we do not apply masking on padded region
                pos=pos[level_index], query_pos=query_embed
            )
            output, self_weights = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN-
            output = self.transformer_ffn_layers[i](
                output
            )
            outputs_class, outputs_mask, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])
            attn_mask, mask_to_update = self.weighted_attention_mask(outputs_mask=outputs_mask, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels], d_pixel=d_pixel, hard_assignments=hard_assignments)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
      

        assert len(predictions_class) == self.num_layers + 1

        # if not self.vis:
        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask
            ),
            'd_pixel': d_pixel,
            'hard_assignments': hard_assignments,
        }

        return out

    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):
        # output: torch.Size([100, 2, 256])
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        # decoder_output: torch.Size([2, 100, 256])
        outputs_class = self.class_embed(decoder_output)
        # outputs_class: torch.Size([2, 100, 20])
        mask_embed = self.mask_embed(decoder_output)
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        # outputs_mask: torch.Size([2, 100, 128, 256])
        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # attn_mask: torch.Size([2, 100, 64, 128])
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)
        attn_mask = (attn_mask < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, attn_mask

    def weighted_attention_mask(self, outputs_mask, attn_mask_target_size, d_pixel, hard_assignments):
        def map_weights_to_full_res(hard_assignments, w):
            B, H, W = hard_assignments.shape
            B, _, superpixel_h, superpixel_w = w.shape
            # Flatten w to make indexing easier
            flat_w = w.view(B, -1)  # [B, superpixel_h * superpixel_w]

            # We assume that hard_assignments have been correctly scaled to index into flat_w
            # Map full resolution pixels to their corresponding superpixel weights
            full_res_weights = flat_w.gather(1, hard_assignments.view(B, -1))   # Flatten hard_assignments for indexing
            full_res_weights = full_res_weights.view(B, 1, H, W)  # Reshape back to the original image dimensions

            return full_res_weights

        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        d_pixel = F.interpolate(d_pixel, size=attn_mask_target_size, mode='bilinear', align_corners=False)
        thresholds = torch.quantile(d_pixel.view(d_pixel.shape[0], -1), 0.3, dim=1)
        masks = torch.zeros_like(d_pixel, dtype=torch.bool)
        for i in range(d_pixel.shape[0]):
            masks[i] = d_pixel[i] <= thresholds[i]
        mask_to_update = masks.expand_as(attn_mask)
        mask_to_update = mask_to_update.flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)
        attn_mask = attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1)
        attn_mask = (attn_mask < 0.5).bool()
        attn_mask[mask_to_update] = False
        attn_mask = attn_mask.detach()
        return attn_mask, mask_to_update


    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        if self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]
