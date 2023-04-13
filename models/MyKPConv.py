import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.kernel_points import load_kernels
from utils.util import knn_in_ball


def gather(neighbors_idx, support):
    B, N, n_neighbors = neighbors_idx.shape
    C = support.shape[-1]
    idx = neighbors_idx.unsqueeze(-1).expand(-1, -1, -1, C).reshape(B, N * n_neighbors, C)
    return torch.gather(support, dim=1, index=idx).view(B, N, n_neighbors, C)


class KPConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, radius,
                 sigma, dimension=3, inf=1e6, use_edge=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.radius = radius
        self.sigma = sigma
        self.dimension = dimension
        self.inf = inf
        self.use_edge = use_edge

        in_channels = in_channels * 2 if use_edge else in_channels

        # Initialize weights
        weights = torch.zeros(size=(kernel_size, in_channels, out_channels))
        self.weights = nn.Parameter(weights)

        # Reset parameters
        self.reset_parameters()

        # Initialize kernel points
        kernel_points = self.initialize_kernel_points()  # (N, 3)
        self.register_buffer("kernel_points", kernel_points)

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))

    def initialize_kernel_points(self):
        """Initialize the kernel point positions in a sphere."""
        kernel_points = load_kernels(self.radius, self.kernel_size, dimension=self.dimension, fixed="center")
        return torch.from_numpy(kernel_points).float()

    def forward(self, q_points, s_points, q_feats, s_feats, neighbors_idx):
        """KPConv forward.
       Args:
           s_feats (Tensor): (B, M, C_in)
           q_feats (Tensor): (B, N, C_in)
           q_points (Tensor): (B, N, 3)
           s_points (Tensor): (B, M, 3)
           neighbors_idx (LongTensor): (B, N, n_neighbors) n_neighbors <= 160
       Returns:
           q_feats (Tensor): (B, N, C_out)
        """

        # get neighbors xyz (B, N, n_neighbors, 3)
        padded_s_points = torch.cat([s_points, torch.zeros_like(s_points[:, :1, :]) + self.inf], 1)  # (B, N, 3) -> (B, N+1, 3)
        neighbors_xyz = gather(neighbors_idx, padded_s_points)  # (B, N, n_neighbors, 3)
        # center the neighbors
        neighbors_xyz -= q_points.unsqueeze(2)

        # get kernel point influences
        sq_distances = torch.sum(torch.square(neighbors_xyz.unsqueeze(3) - self.kernel_points), dim=-1)  # (B, N, n_neighbors, k)
        neighbors_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.sigma, min=0.0).transpose(-2, -1)  # (B, N, k, n_neighbors)
        # print(neighbor_weights)

        # apply neighbors weights
        padded_s_feats = torch.cat([s_feats, torch.zeros_like(s_feats[:, :1, :])], dim=1)  # (B, N, C) -> (B, N+1, C)
        neighbors_feats = gather(neighbors_idx, padded_s_feats)  # (B, N, n_neighbors, C)

        # edge feats
        if self.use_edge:
            center_feats = q_feats.unsqueeze(2).expand(-1, -1, neighbors_idx.shape[-1], -1)  # (B, N, n_neighbors, C)
            neighbors_feats = torch.concat([neighbors_feats - center_feats, center_feats], dim=-1)  # (B, N, n_neighbors, C)

        weighted_feats = torch.matmul(neighbors_weights, neighbors_feats)  # (B, N, k, n_neighbors) x (B, N, n_neighbors, C_in) -> (B, N, k, C_in)
        output_feats = torch.einsum("bnkc,kcd->bnd", weighted_feats, self.weights)  # (B, N, k, C_in) x (K, C_in, C_out) -> (B, N, C_out)

        # density normalization
        neighbors_feats_sum = torch.sum(neighbors_feats, dim=-1)  # (B, N, n_neighbors)
        neighbors_num = torch.sum(torch.gt(neighbors_feats_sum, 0.0), dim=-1)  # (B, N)
        neighbors_num = torch.max(neighbors_num, torch.ones_like(neighbors_num))  # (B, N)
        output_feats = output_feats / neighbors_num.unsqueeze(-1)

        return output_feats

    def extra_repr(self) -> str:
        param_strings = [
            f"kernel_size={self.kernel_size}",
            f"in_channels={self.in_channels}",
            f"out_channels={self.out_channels}",
            f"radius={self.radius:g}",
            f"sigma={self.sigma:g}",
        ]
        if self.dimension != 3:
            param_strings.append(f"dimension={self.dimension}")
        format_string = ", ".join(param_strings)
        return format_string


class KPConvBlock(nn.Module):
    def __init__(self, in_feat, layer_dims, kernel_size, radius,
                 sigma, dimension=3):
        super(KPConvBlock, self).__init__()
        self.layer_dims = layer_dims

        self.layers = nn.ModuleList()
        self.norms_and_act = nn.ModuleList()
        self.act = nn.LeakyReLU(0.2, inplace=True)

        for i in range(len(self.layer_dims)):
            # use_edge = i % 2 == 0
            use_edge = False
            in_dim = in_feat if i == 0 else self.layer_dims[i - 1]
            out_dim = self.layer_dims[i]

            self.layers.append(KPConv(in_dim, out_dim, kernel_size, radius, sigma, dimension=dimension, use_edge=use_edge))
            self.norms_and_act.append(nn.Sequential(nn.LayerNorm(out_dim), nn.LeakyReLU(0.2)))

    def forward(self, q_points, s_points, q_feats, s_feats, neighbors_idx, affine_transformation=None, support_sample=None):
        for i in range(len(self.layer_dims)):
            residual = q_feats
            q_feats = self.layers[i](q_points, s_points, q_feats, s_feats, neighbors_idx)

            if affine_transformation is not None and i == 0:
                # q_feats = affine_transformation[..., :self.layer_dims[0]] * q_feats + affine_transformation[..., self.layer_dims[0]:]

                q_feats = q_feats.unsqueeze(2)  # (B, N, 1, C)
                support_sample = support_sample.unsqueeze(1)  # (1, 1, M, C)
                affine_transformation = affine_transformation.unsqueeze(1)  # (1, 1, M, C)

                cos_similarity = torch.cosine_similarity(q_feats, support_sample, dim=-1)
                affine_weights = torch.softmax(cos_similarity * 10, dim=-1)  # (B, N, M)

                q_feats = affine_transformation[..., :self.layer_dims[0]] * q_feats + affine_transformation[..., self.layer_dims[0]:]  # (B, N, M, C)
                q_feats = torch.einsum('bnm,bnmc->bnc', affine_weights, q_feats)

            if i % 2 != 0 and q_feats.shape[-1] == residual.shape[-1]:
                q_feats += residual

            q_feats = self.norms_and_act[i](q_feats)
            s_feats = q_feats

        return q_feats


class MlpBlock(nn.Module):
    def __init__(self, in_feat, layer_dims):
        super(MlpBlock, self).__init__()
        self.layer_dims = layer_dims

        layers = []
        for i in range(len(layer_dims)):
            in_dim = in_feat if i == 0 else layer_dims[i - 1]
            out_dim = layer_dims[i]
            layers.append(nn.Linear(in_dim, out_dim, bias=False))
            layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.LeakyReLU(0.2))

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class AnchorBackbone(nn.Module):
    def __init__(self, conv_widths, mlp_widths, first_dim, voxel_size=0.03, kernel_size=15, radius=15.0, sigma=8.0, neighbors_limit=80):
        super(AnchorBackbone, self).__init__()

        self.conv_widths = conv_widths
        self.mlp_widths = mlp_widths
        self.first_dim = first_dim

        self.voxel_size = voxel_size
        self.first_radius = radius * self.voxel_size
        self.first_sigma = sigma * self.voxel_size
        self.neighbors_limit = neighbors_limit
        self.kernel_size = kernel_size

        self.kpconv_list = nn.ModuleList()
        for i in range(len(self.conv_widths)):
            if i == 0:
                in_feat = first_dim
            else:
                in_feat = conv_widths[i - 1][-1]
            self.kpconv_list.append(KPConvBlock(in_feat, conv_widths[i], self.kernel_size, self.first_radius, self.first_sigma))

        in_dim = 0
        for width in conv_widths:
            in_dim += width[-1]
        self.mlp = MlpBlock(in_dim, mlp_widths)

    def forward(self, x, affine_transformation=None, support_samples=None):
        """
        Args:
            x (Tensor): (B, 9, N)

        Returns:
            out (Tensor): (B, N, C)
        """
        xyz = x[..., :3]

        neighbors_idx = knn_in_ball(xyz, xyz, self.first_radius, neighbors_limit=self.neighbors_limit)

        kpconv_outputs = []
        for i in range(len(self.conv_widths)):
            if affine_transformation is not None:
                affine = affine_transformation[i]
                support_sample = support_samples[i]
            else:
                affine = affine_transformation
                support_sample = support_samples

            x = self.kpconv_list[i](xyz, xyz, x, x, neighbors_idx, affine_transformation=affine, support_sample=support_sample)
            kpconv_outputs.append(x)

        x = torch.cat(kpconv_outputs, dim=-1)
        x = self.mlp(x)

        return kpconv_outputs, x
