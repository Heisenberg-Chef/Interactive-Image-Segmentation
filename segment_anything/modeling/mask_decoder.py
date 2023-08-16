# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
from torch import nn
from torch.nn import functional as F

from typing import List, Tuple, Type

from .common import LayerNorm2d


class MaskDecoder(nn.Module):
    """
    蒙版解码器模块
    """

    def __init__(
            self,
            *,
            transformer_dim: int,
            transformer: nn.Module,
            num_multimask_outputs: int = 3,
            activation: Type[nn.Module] = nn.GELU,
            iou_head_depth: int = 3,
            iou_head_hidden_dim: int = 256,
    ) -> None:
        """
        Predicts masks given an image and prompt embeddings, using a
        transformer architecture.
        预测掩码图，使用transformer架构。
        Arguments:
          transformer_dim (int): the channel dimension of the transformer   transformer的维度：256
          transformer (nn.Module): the transformer used to predict masks    transformer模型
          num_multimask_outputs (int): the number of masks to predict       生成多个mask：网络默认是3个，其目的就是二义相消
            when disambiguating masks
          activation (nn.Module): the type of activation to use when        GELU激活函数
            upscaling masks
          iou_head_depth (int): the depth of the MLP used to predict        衡量mask品质的MLP的层数
            mask quality
          iou_head_hidden_dim (int): the hidden dimension of the MLP        MLP的隐藏层维度
            used to predict mask quality
        """
        super().__init__()
        self.transformer_dim = transformer_dim  # 256
        self.transformer = transformer  # two way transformer
        ##############
        # 此处可以总结一下，embedding层我们只需要的其中的weight权重，而不使用这些层进行计算！！！
        ##############
        self.num_multimask_outputs = num_multimask_outputs
        # 嵌入层 计算 iou的 token ：我们需要的只是这个层的参数，因为是可以训练的所以这么写。
        self.iou_token = nn.Embedding(1, transformer_dim)  # vocal size 、embedding length = 256
        self.num_mask_tokens = num_multimask_outputs + 1  # mask需要3个，这里进行了 + 1 操作。
        # 嵌入层 计算 mask的token ,我们需要的只是这个层的参数，因为是可以训练的所以这么写。
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)  # 4 ， embedding length = 256

        # 上采样 c:256 -> c:32
        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose2d(transformer_dim, transformer_dim // 4, kernel_size=2, stride=2),  # x2
            LayerNorm2d(transformer_dim // 4),
            activation(),
            nn.ConvTranspose2d(transformer_dim // 4, transformer_dim // 8, kernel_size=2, stride=2),  # x2
            activation(),
        )
        #
        self.output_hypernetworks_mlps = nn.ModuleList(
            # 还能这么写啊。。。。 3层 MLP的组合 。。。。
            [
                # 全连接组合 输入 256 ，隐藏 256 ，输出 32 ，深度 3
                MLP(transformer_dim, transformer_dim, transformer_dim // 8, 3)
                for i in range(self.num_mask_tokens)  # 共有3层MLP的module
            ]
        )

        # iou 推测头 - 将1,1,256
        self.iou_prediction_head = MLP(
            transformer_dim, iou_head_hidden_dim, self.num_mask_tokens, iou_head_depth
        )

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multimask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict masks given image and prompt embeddings.

        Arguments:
        Image Encoder之后 图像编码
          image_embeddings (torch.Tensor): the embeddings from the image encoder
        图形编码
          image_pe (torch.Tensor): positional encoding with the shape of image_embeddings
        稀疏矩阵 - prompt
          sparse_prompt_embeddings (torch.Tensor): the embeddings of the points and boxes
        稠密矩阵 - 图形mask的token
          dense_prompt_embeddings (torch.Tensor): the embeddings of the mask inputs
        返回几个mask，我们论文中是3个。
          multimask_output (bool): Whether to return multiple masks or a single
            mask.

        Returns:
        返回：预测masks 图像
          torch.Tensor: batched predicted masks
        预测的mask质量数值。
          torch.Tensor: batched predictions of mask quality
        """
        masks, iou_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        # 创建一个切片对象
        if multimask_output:
            # 从 idx 1 - end
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :] # 1,4,256,256 -- 1,3,256,256 dix:1-end
        iou_pred = iou_pred[:, mask_slice] # 1,4 -- 1,3 dix:1-end

        # Prepare output
        return masks, iou_pred

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        # Concatenate output tokens
        # 合并2个权重值，合并之后为 5,256 ：第一层是iou_token，下面的事mask token
        output_tokens = torch.cat([self.iou_token.weight, self.mask_tokens.weight], dim=0)
        # unsqueeze(0)，在张量最外层添加一个维度。
        # expand(sparse_prompt_embeddings.size(0), -1, -1) 将维度扩大到 稀疏矩阵的 BS。
        output_tokens = output_tokens.unsqueeze(0).expand(sparse_prompt_embeddings.size(0), -1, -1)  # 保证BS上是对其的
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)  # 得到与稀疏矩阵合并的tokens，在1维度上进行合并 - N_Tokens维度

        # Expand per-image data in batch direction to be per-mask
        # tokens 测试用例中 1 - 8 - 256
        src = torch.repeat_interleave(image_embeddings, tokens.shape[0], dim=0)
        src = src + dense_prompt_embeddings  # src混合：稠密层+image编码
        # image_pe 我们得看看是怎么来的。
        pos_src = torch.repeat_interleave(image_pe, tokens.shape[0], dim=0)  # 位置编码
        b, c, h, w = src.shape

        # Run the transformer - 输入为 图像编码（已经与mask融合）、位置编码、稀疏提示 ，输出则是经过双路注意力机制的稀疏提示tokens，和图像与masks融合的tokens
        hs, src = self.transformer(src, pos_src, tokens)
        # 1维度上 0 层的信息是iou token？
        iou_token_out = hs[:, 0, :]
        # 下面的token都是mask_token
        mask_tokens_out = hs[:, 1: (1 + self.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        src = src.transpose(1, 2).view(b, c, h, w)  # 1 4096 256 -- 1 256 4096 -- 1 256 64 64
        # x4 上采样 1 256 64 64 -- 1 32 256 256
        upscaled_embedding = self.output_upscaling(src)  # 经过上采样
        hyper_in_list: List[torch.Tensor] = []
        # 4个token，分别是 1 ， 32 维度的
        # 4层分别计算4个mask token，单独计算并且保存到列表中
        for i in range(self.num_mask_tokens):
            # hypernetworks 超网络，只一种网络结构专门生成其他网络权重的，
            # 在这里我们的超网络就是用来生成下面 mask图形的权重
            #
            hyper_in_list.append(self.output_hypernetworks_mlps[i](mask_tokens_out[:, i, :]))
        # 拼接成一个新的张量，dim的意义在于，在张量的那个维度进行拼接
        # 代码中 我们的张量为1,32维度，如果dim=1则表示，生成一个张量的dim维度上进行拼接，生成1,4,32的张量。
        ######  dim=0 -- 4 1 32 || dim=2 -- 1 32 4
        hyper_in = torch.stack(hyper_in_list, dim=1)
        b, c, h, w = upscaled_embedding.shape
        # hyper_in -- 之前网络计算出来的超参数 1,4,32
        # upscaled_embedding -- 我们通过上采样得到的1,32,256,256 的张量 -- view(b,c,h*w)之后 1,32,65536
        # hyper_in@upscaled_embedding.view(b, c, h * w) -- 1,4,32 @ 1,32,65536 = 1,4,65536 【4行32列 乘以 65536列32行】 合并行向量，计算高维信息。
        # 最后：重新调整维度信息 1 4 256 256，H x W
        masks = (hyper_in @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        # Generate mask quality predictions
        # iou_token_out -- 1,256
        # 返回4个mask的评分。
        iou_pred = self.iou_prediction_head(iou_token_out)

        return masks, iou_pred


# Lightly adapted from
# https://github.com/facebookresearch/MaskFormer/blob/main/mask_former/modeling/transformer/transformer_predictor.py # noqa
class MLP(nn.Module):
    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            sigmoid_output: bool = False,
    ) -> None:
        """
        MLP
        """
        super().__init__()
        self.num_layers = num_layers  # 层数
        # 隐藏层个数 带着维度的 数组
        h = [hidden_dim] * (num_layers - 1)
        # 这是一个线性层组合，3层 zip 是一个包装类 将2个可以迭代的对象一一组合
        # input dim - hidden
        # h - h
        # h - dim
        self.layers = nn.ModuleList(
            # n - 输入维度 隐层 隐层
            # k 隐层，隐层，输出维度
            nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])
        )
        self.sigmoid_output = sigmoid_output

    def forward(self, x):
        # 前向传播，最后一层使用sigmoid
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        if self.sigmoid_output:
            x = F.sigmoid(x)
        return x
