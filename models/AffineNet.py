import torch
import torch.nn as nn


class AffineNet(nn.Module):
    def __init__(self, in_dims, out_dims):
        super(AffineNet, self).__init__()
        self.out_dims = out_dims

        self.net = nn.Sequential(nn.Linear(in_dims, in_dims * 3, bias=False),
                                 nn.LayerNorm(in_dims * 3),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Dropout(0.2),
                                 nn.Linear(in_dims * 3, in_dims * 6, bias=False),
                                 nn.LayerNorm(in_dims * 6),
                                 nn.LeakyReLU(0.2, inplace=True),
                                 nn.Dropout(0.2),
                                 nn.Linear(in_dims * 6, out_dims))

        # self.net = nn.Sequential(nn.Linear(in_dims, in_dims // 2, bias=False),
        #                          nn.LayerNorm(in_dims // 2),
        #                          nn.LeakyReLU(0.2, inplace=True),
        #                          nn.Linear(in_dims // 2, out_dims))

        # self.mlp1 = nn.Sequential(nn.Linear(in_dims, in_dims * 3),
        #                           nn.LayerNorm(in_dims * 3),
        #                           nn.LeakyReLU(0.2, inplace=True))
        #
        # self.mlp2 = nn.Sequential(nn.Linear(in_dims * 3, out_dims))

    def forward(self, x):
        x = self.net(x)
        # x = self.mlp1(x)
        # x = torch.max(x, dim=2, keepdim=True)[0]
        # x = self.mlp2(x)
        return x
