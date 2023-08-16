# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import Tensor, nn

import math
from typing import Tuple, Type

from .common import MLPBlock


class TwoWayTransformer(nn.Module):
    def __init__(
            self,
            depth: int,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int,
            activation: Type[nn.Module] = nn.ReLU,
            attention_downsample_rate: int = 2,  # attention 下采样率 2
    ) -> None:
        """
        A transformer decoder that attends to an input image using
        queries whose positional embedding is supplied.
        # transformer 解码器，根据输入图片的位置嵌入信息进行解码。
        Args:
          depth (int): number of layers in the transformer # 解码器层数 -- 2 层解码器
          embedding_dim (int): the channel dimension for the input embeddings # 解码器对应维度信息 256
          num_heads (int): the number of heads for multihead attention. Must # 多头注意力的个数，这里明确必须整除256；论文代码中取的8.
            divide embedding_dim
          mlp_dim (int): the channel dimension internal to the MLP block # 隐藏层的 mlp个数 这里是 2048
          activation (nn.Module): the activation to use in the MLP block # 激活函数
        """
        super().__init__()
        self.depth = depth  # 2 层
        self.embedding_dim = embedding_dim  # 256 ，这个网络标准维度就是256
        self.num_heads = num_heads  # 多头数量，论文中  8
        self.mlp_dim = mlp_dim  # mlp hidden layer = 2048
        self.layers = nn.ModuleList()  # 串行操作nn.Module

        for i in range(depth):  # 2层 双路交叉注意力计算
            self.layers.append(
                TwoWayAttentionBlock(
                    embedding_dim=embedding_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    activation=activation,
                    attention_downsample_rate=attention_downsample_rate,
                    skip_first_layer_pe=(i == 0),
                )
            )
        # 我们得到了最终合并注意力的 Q K
        self.final_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        # 层归一化，同样是再张量最后维度中进行归一化操作。
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
            self,
            image_embedding: Tensor,  # 图像位置编码 B 256 H W
            image_pe: Tensor,  # 稠密 B 256 H W
            point_embedding: Tensor,  # 稀疏 B N_Points 256
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            图像嵌入张量的形状是 B 256 H W
          image_embedding (torch.Tensor): image to attend to. Should be shape
            B x embedding_dim x h x w for any h and w.
            图像嵌入的位置编码信息，同样是B 256 H W
          image_pe (torch.Tensor): the positional encoding to add to the image. Must
            have the same shape as image_embedding.
            查询点的嵌入信息，B N_Points 256
          point_embedding (torch.Tensor): the embedding to add to the query points.
            Must have shape B x N_points x embedding_dim for any N_points.

        Returns:
            返回 点嵌入向量 与 图像嵌入向量
          torch.Tensor: the processed point_embedding
          torch.Tensor: the processed image_embedding
        """
        # BxCxHxW -> BxHWxC == B x N_image_tokens x C
        bs, c, h, w = image_embedding.shape
        image_embedding = image_embedding.flatten(2).permute(0, 2, 1)  # 图片嵌入信息
        # 图片的位置编码信息 B 256 H W
        image_pe = image_pe.flatten(2).permute(0, 2, 1)  # 在第三个维度展平 并且转化成 B N_tokens C 的一个维度

        # Prepare queries
        # 准备Q K。
        queries = point_embedding  # token -- 稀疏矩阵
        keys = image_embedding  # 图片嵌入张量 --

        # Apply transformer blocks and final layernorm
        # 2层的TwoWayAttentionBlock计算交叉注意力
        for layer in self.layers:
            queries, keys = layer(
                queries=queries,  # point_embedding - token
                keys=keys,  # image_embedding
                query_pe=point_embedding,  # point_embedding - token
                key_pe=image_pe,  # image_positional_embedding
            )

        # Apply the final attention layer from the points to the image
        # 最后一层的注意力机制层 point稀疏 对于 image稠密 的 注意力机制
        q = queries + point_embedding # token + image
        k = keys + image_pe # token + image
        attn_out = self.final_attn_token_to_image(q=q, k=k, v=keys) # keys代表了 image embedding
        queries = queries + attn_out
        queries = self.norm_final_attn(queries)
        # 返回 Q K ，这里是我们已经融合了 token 与 image 的
        return queries, keys


class TwoWayAttentionBlock(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            mlp_dim: int = 2048,
            activation: Type[nn.Module] = nn.ReLU,
            attention_downsample_rate: int = 2,
            skip_first_layer_pe: bool = False,
    ) -> None:
        """
        总体来说，双路Attention模块的意义在于将token的维度信息与image embedding的信息进行
        QK的融合，通过融合2者Q K 信息，得到交叉注意力，以供进行mask decode操作。

        A transformer block with four layers:
        (1) self-attention of sparse inputs,
        (2) cross attention of sparse inputs to dense inputs,
        (3) mlp block on sparse inputs, and
        (4) cross attention of dense inputs to sparse inputs.
        # transformer块，共有4层
        # 1 稀疏矩阵自注意力
        # 2 交叉注意力 稀疏 x 稠密
        # 3 mlp层计算稀疏输入
        # 4 交叉注意力 稠密 x 稀疏
        Arguments:
          embedding_dim (int): the channel dimension of the embeddings #
          num_heads (int): the number of heads in the attention layers
          mlp_dim (int): the hidden dimension of the mlp block
          activation (nn.Module): the activation of the mlp block
          skip_first_layer_pe (bool): skip the PE on the first layer
        """
        super().__init__()
        # 自注意力层 下采样是 1
        self.self_attn = Attention(embedding_dim, num_heads)
        self.norm1 = nn.LayerNorm(embedding_dim)  # 层归一化
        # 交叉注意力 这里下采样是2
        # 注意力机制从 prompt token 到 image embedding
        self.cross_attn_token_to_image = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        self.norm2 = nn.LayerNorm(embedding_dim)  # 层归一化

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        # 注意力机制 image embedding 到 tokens
        self.cross_attn_image_to_token = Attention(
            embedding_dim, num_heads, downsample_rate=attention_downsample_rate
        )
        # 是否跳过self Attention模块
        self.skip_first_layer_pe = skip_first_layer_pe

    def forward(
            self, queries: Tensor, keys: Tensor, query_pe: Tensor, key_pe: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        值得一提的是，在这个计算过程中，A 对 B 的交叉注意力机制的意义也就是最后的对象B作为V
        也就是QK说明书看懂了，我们需要组装V的这个零件。
        queries：token Q
        keys：img e
        query_pe：token Q
        key_pe：img pe
        返回：Q，K矩阵，维度与输入的一样
        """

        # --- 1
        # Self attention block
        # 是否跳过稀疏矩阵的自注意力层
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            # 如果不跳过将query pe与query进行叠加
            q = queries + query_pe
            # 叠加PE的自注意力机制
            attn_out = self.self_attn(q=q, k=q, v=queries)
            # q再次进行叠加
            queries = queries + attn_out
        queries = self.norm1(queries)  # 归一化操作

        # --- 2
        # Cross attention block, tokens attending to image embedding
        # 交叉注意力机制，token注意力到embedding
        # 叠加两者的QK，pe是embedding层
        # q@k.transpose(-1,-2)@v 大概其就这么个计算过程吧，不难。
        q = queries + query_pe
        k = keys + key_pe
        # x2 downscale，最后还是会还原 原始的维度
        # 这里将token的q与计算交叉注意力的q进行叠加
        attn_out = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = queries + attn_out
        queries = self.norm2(queries)

        # --- 3
        # MLP block
        # token与embedding计算之后的结果进行一个mlp
        # mlp 将向量维度提升到2048再缩小到256
        mlp_out = self.mlp(queries)
        queries = queries + mlp_out
        queries = self.norm3(queries)

        # --- 4
        # Cross attention block, image embedding attending to tokens
        # 还是交叉注意力机制，从embedding 到token
        q = queries + query_pe
        k = keys + key_pe
        attn_out = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = keys + attn_out
        keys = self.norm4(keys)

        return queries, keys


class Attention(nn.Module):
    """
    An attention layer that allows for downscaling the size of the embedding
    after projection to queries, keys, and values.
    # Attention模块，可以下采样到一个ratio对应的维度，在上采样到256维度的嵌入维度，在进行 QKV的匹配计算
    """

    def __init__(
            self,
            embedding_dim: int,
            num_heads: int,
            downsample_rate: int = 1,  # 下采样倍率为 1 也就是不进行下采样了。
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim  # 256 的嵌入维度
        self.internal_dim = embedding_dim // downsample_rate
        self.num_heads = num_heads  # 8个注意力头
        #### 整除判断
        assert self.internal_dim % num_heads == 0, "num_heads must divide embedding_dim."
        # QKV 都是Linear层。 直接在最开始计算的时候就进行了下采样操作
        self.q_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.k_proj = nn.Linear(embedding_dim, self.internal_dim)
        self.v_proj = nn.Linear(embedding_dim, self.internal_dim)
        # 将下采样的倍率进行调整，调整到 256 -- 我们根据原始的倍率进行上采样还原。
        # 下采样倍率是 1 ，我们可以理解为一个 nn.Identity
        # 线性层会在数据最底层，也就是dim=-1的地方进行计算，我们需要保证最后的维度与linear的输入一样
        self.out_proj = nn.Linear(self.internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor, num_heads: int) -> Tensor:
        """
        拆分都头注意力
        """
        # b，n个向量，通道数
        b, n, c = x.shape
        # 多一个维度，分多头
        x = x.reshape(b, n, num_heads, c // num_heads)
        # 返回：B，注意力头个数，token个数，每一个头的通道数
        return x.transpose(1, 2)  # B x N_heads x N_tokens x C_per_head

    def _recombine_heads(self, x: Tensor) -> Tensor:
        """
        组合多头注意力
        """
        b, n_heads, n_tokens, c_per_head = x.shape
        x = x.transpose(1, 2)
        # 返回：B token个数 通道数
        return x.reshape(b, n_tokens, n_heads * c_per_head)  # B x N_tokens x C

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        # Input projections
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        # Separate into heads
        # 分多头 b n_head,n_token,c_per_head
        q = self._separate_heads(q, self.num_heads)
        k = self._separate_heads(k, self.num_heads)
        v = self._separate_heads(v, self.num_heads)

        # Attention
        _, _, _, c_per_head = q.shape
        attn = q @ k.permute(0, 1, 3, 2)  # B x N_heads x N_tokens x N_tokens
        attn = attn / math.sqrt(c_per_head)  # sqrt d
        # 经过一个softmax层
        attn = torch.softmax(attn, dim=-1)

        # Get output
        # attn通过乘以V的转置合并成为原始的输出矩阵
        out = attn @ v  # B x N_Head X N_tokens X C_Per_Head
        out = self._recombine_heads(out)
        out = self.out_proj(out)
        # 此处线性层是回复原始输出倍率的
        # 输出 B N C
        return out
