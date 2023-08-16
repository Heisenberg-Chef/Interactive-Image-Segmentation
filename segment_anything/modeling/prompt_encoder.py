# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
from torch import nn

from typing import Any, Optional, Tuple, Type

from .common import LayerNorm2d
# from common import LayerNorm2d


# 提示编码器
#         prompt_encoder=PromptEncoder(
#             embed_dim=prompt_embed_dim,  # 256
#             image_embedding_size=(image_embedding_size, image_embedding_size), # (64,64)
#             input_image_size=(image_size, image_size), # (1024,1024)
#             mask_in_chans=16,
#         ),
class PromptEncoder(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            image_embedding_size: Tuple[int, int],
            input_image_size: Tuple[int, int],
            mask_in_chans: int,
            activation: Type[nn.Module] = nn.GELU,
    ) -> None:
        """
        Encodes prompts for input to SAM's mask decoder.

        Arguments:
          embed_dim (int): The prompts' embedding dimension
          image_embedding_size (tuple(int, int)): The spatial size of the
            image embedding, as (H, W).
          input_image_size (int): The padded size of the image as input
            to the image encoder, as (H, W).
          mask_in_chans (int): The number of hidden channels used for
            encoding input masks.
          activation (nn.Module): The activation to use when encoding
            input masks.
        """
        super().__init__()
        self.embed_dim = embed_dim  # 256
        self.input_image_size = input_image_size  # 1024 1024
        self.image_embedding_size = image_embedding_size  # 64 64
        self.pe_layer = PositionEmbeddingRandom(embed_dim // 2)  # param : 128
        # embedding层的数量，4个，点 + 正负样本 + box信息 + box信息
        # 在这里值得一提的是为什么是 1 ，256
        # 在NLP中embedding是将稀疏矩阵的文字转化为稠密矩阵的
        # 1：嵌入维度为我们输入的提示坐标数据为 只能是tensor([0])
        # 256 表示我们代表那个位置编码信息
        self.num_point_embeddings: int = 4  # pos/neg point + 2 box corners
        point_embeddings = [nn.Embedding(1, embed_dim) for i in range(self.num_point_embeddings)]
        self.point_embeddings = nn.ModuleList(point_embeddings)  # embedding(1,256) 一共4个
        self.not_a_point_embed = nn.Embedding(1, embed_dim)  # embedding(1,256)
        # 掩码embedding也作为提示。 256 256
        self.mask_input_size = (4 * image_embedding_size[0], 4 * image_embedding_size[1])
        # 掩码下采样 - mask_in_chans 其实是隐藏的通道数量 我们这里取 16
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, mask_in_chans // 4, kernel_size=2, stride=2),  # 尺寸缩小了 2倍 ，通道变成了 4
            LayerNorm2d(mask_in_chans // 4),  # 2维归一化， 图片均值1 方差1
            activation(),  # GELU
            nn.Conv2d(mask_in_chans // 4, mask_in_chans, kernel_size=2, stride=2),  # 尺寸缩小了2倍，通道变成了 16
            LayerNorm2d(mask_in_chans),  # 归一化
            activation(),  # GELU
            nn.Conv2d(mask_in_chans, embed_dim, kernel_size=1),  # 256 维度 64 64
        )
        self.no_mask_embed = nn.Embedding(1, embed_dim)

    def get_dense_pe(self) -> torch.Tensor:
        """
        Returns the positional encoding used to encode point prompts,
        applied to a dense set of points the shape of the image encoding.

        Returns:
          torch.Tensor: Positional encoding with shape
            1x(embed_dim)x(embedding_h)x(embedding_w)
        """
        return self.pe_layer(self.image_embedding_size).unsqueeze(0)

    def _embed_points(
            self,
            points: torch.Tensor,
            labels: torch.Tensor,
            pad: bool,
    ) -> torch.Tensor:

        """
        嵌入点坐标提示：
        Embeds point prompts.
        """
        points = points + 0.5  # Shift to center of pixel # 这里所说的移动到 像素中心，也就是都曲中 在像素点的中心作为提示点 trick？
        if pad:  # 是否填充？-- 取决于是否有boxes提示的存在，如果没有boxes 这里需要填充 0 我猜测是给boxes占位置。
            padding_point = torch.zeros((points.shape[0], 1, 2), device=points.device)  # B 1 2 -- 全部都是 0
            padding_label = -torch.ones((labels.shape[0], 1), device=labels.device)  # B 1  -- 全部都是 -1
            points = torch.cat([points, padding_point], dim=1)  # 在 1 维度上进行填充
            labels = torch.cat([labels, padding_label], dim=1)  # 在 1 维度上进行填充
        # （B,N,256）
        point_embedding = self.pe_layer.forward_with_coords(points, self.input_image_size)
        # 这段操作，是根据我们判别式返回的维度，在对应尺寸维度上进行赋值。
        point_embedding[labels == -1] = 0.0
        point_embedding[labels == -1] += self.not_a_point_embed.weight
        point_embedding[labels == 0] += self.point_embeddings[0].weight # 嵌入embeddings的权重
        point_embedding[labels == 1] += self.point_embeddings[1].weight # 嵌入embeddings的权重
        return point_embedding

    def _embed_boxes(self, boxes: torch.Tensor) -> torch.Tensor:
        """Embeds box prompts."""
        boxes = boxes + 0.5  # Shift to center of pixel
        coords = boxes.reshape(-1, 2, 2)
        corner_embedding = self.pe_layer.forward_with_coords(coords, self.input_image_size)
        corner_embedding[:, 0, :] += self.point_embeddings[2].weight
        corner_embedding[:, 1, :] += self.point_embeddings[3].weight
        return corner_embedding

    def _embed_masks(self, masks: torch.Tensor) -> torch.Tensor:
        """Embeds mask inputs."""
        mask_embedding = self.mask_downscaling(masks)
        return mask_embedding

    def _get_batch_size(
            self,
            points: Optional[Tuple[torch.Tensor, torch.Tensor]],
            boxes: Optional[torch.Tensor],
            masks: Optional[torch.Tensor],
    ) -> int:
        """
        Gets the batch size of the output given the batch size of the input prompts.
        """
        if points is not None:
            # tuple 的第一个元素 -- tensor数据 0 号维度。
            return points[0].shape[0]
        elif boxes is not None:
            return boxes.shape[0]
        elif masks is not None:
            return masks.shape[0]
        else:
            return 1

    def _get_device(self) -> torch.device:
        return self.point_embeddings[0].weight.device

    def forward(
            self,
            points: Optional[Tuple[torch.Tensor, torch.Tensor]],
            boxes: Optional[torch.Tensor],
            masks: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Embeds different types of prompts, returning both sparse and dense
        embeddings.

        Arguments: # 参数中都可以为None，点坐标prompt ((x,y),(pos/neg))
          points (tuple(torch.Tensor, torch.Tensor) or none): point coordinates
            and labels to embed.
          boxes (torch.Tensor or none): boxes to embed
          masks (torch.Tensor or none): masks to embed

        Returns:
            # 返回：稀疏嵌入提示 Batch 输入提示的个数 256（默认维度）
          torch.Tensor: sparse embeddings for the points and boxes, with shape
            BxNx(embed_dim), where N is determined by the number of input points
            and boxes.
            # 返回：稠密嵌入masks，也就是之前推理过的掩码数据 Batch 256 64 64 ！！ TODO
          torch.Tensor: dense embeddings for the masks, in the shape
            Bx(embed_dim)x(embed_H)x(embed_W)
        """
        bs = self._get_batch_size(points, boxes, masks)
        # 稀疏矩阵提示
        # torch.empty 是用来创建一个空的张量 ，只有维度，内存不清零：里面的垃圾值是张量的内存中保存的数。
        # dim 1 为 0 ，这是一个空矩阵，有维度信息但是没有数据，需要将来进行拼接使用
        sparse_embeddings = torch.empty((bs, 0, self.embed_dim), device=self._get_device())
        # 分解points
        if points is not None:
            coords, labels = points  # 坐标信息与正负样本提示 coords：[[500, 375],...] labels：[1,...]
            # 点提示：（B,N,256）
            point_embeddings = self._embed_points(coords, labels, pad=(boxes is None))
            # 形成稀疏矩阵 （B,N,256）
            sparse_embeddings = torch.cat([sparse_embeddings, point_embeddings], dim=1)
        if boxes is not None:
            box_embeddings = self._embed_boxes(boxes)
            sparse_embeddings = torch.cat([sparse_embeddings, box_embeddings], dim=1)

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
        else:
            # 如果没有生成token 那么就生成一个垃圾噪音
            dense_embeddings = self.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
                bs, -1, self.image_embedding_size[0], self.image_embedding_size[1]
            )
        # sparse （B,N,256） dense (B,256,64,64)
        return sparse_embeddings, dense_embeddings


class PositionEmbeddingRandom(nn.Module):
    """
    Positional encoding using random spatial frequencies.
    随机空间概率
    如果不使用提示的时候可以使用随机噪声代替提示信息
    """

    def __init__(self, num_pos_feats: int = 64, scale: Optional[float] = None) -> None:
        super().__init__()
        if scale is None or scale <= 0.0:
            scale = 1.0
        # 注册一个缓冲区，缓冲区是一种可以被模型持有并与模型一起存储的张量，但是不会被优化器更新
        # 这样做的目的是为模型创建一些随机性
        self.register_buffer(
            "positional_encoding_gaussian_matrix",  # 名字为 位置编码高斯随机矩阵
            scale * torch.randn((2, num_pos_feats)),  # randn 均值为0 标准差为 scale (2,128) 维度
        )

    def _pe_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """Positionally encode points that are normalized to [0,1]."""
        # assuming coords are in [0, 1]^2 square and have d_1 x ... x d_n x 2 shape
        # 将标准化到 0 1 的编码，转为-1 1
        # 然后通过buffer随机参数进行线性变换
        # 最后结果映射到0 2*PI，生成正弦与余弦编码
        coords = 2 * coords - 1 # 1,2,2 （中心化）
        # coords 维度是 (1,2,128)
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2 * np.pi * coords
        # outputs d_1 x ... x d_n x C shape
        # --- THIS IS A TRICK.
        # 在最后一个维度进行拼接，形成一个(1,2,256)的encoding 矩阵
        # 该矩阵表示了通过随机高斯矩阵与正弦余弦位置编码，将位置坐标与随机信息融合，提升模型感知能力。
        return torch.cat([torch.sin(coords), torch.cos(coords)], dim=-1)

    def forward(self, size: Tuple[int, int]) -> torch.Tensor:
        """Generate positional encoding for a grid of the specified size."""
        h, w = size
        device: Any = self.positional_encoding_gaussian_matrix.device
        grid = torch.ones((h, w), device=device, dtype=torch.float32)
        y_embed = grid.cumsum(dim=0) - 0.5
        x_embed = grid.cumsum(dim=1) - 0.5
        y_embed = y_embed / h
        x_embed = x_embed / w

        pe = self._pe_encoding(torch.stack([x_embed, y_embed], dim=-1))
        return pe.permute(2, 0, 1)  # C x H x W

    def forward_with_coords(
            self, coords_input: torch.Tensor, image_size: Tuple[int, int]
    ) -> torch.Tensor:
        """Positionally encode points that are not normalized to [0,1]."""
        # 通过除以图像的宽高，标准化到 0 - 1 之间
        # 0 是输入的原始坐标
        # 1 是填充的padding 或者 输入的box
        coords = coords_input.clone()
        coords[:, :, 0] = coords[:, :, 0] / image_size[1]
        coords[:, :, 1] = coords[:, :, 1] / image_size[0]
        return self._pe_encoding(coords.to(torch.float))  # B x N x C

