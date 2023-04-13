import torch


def knn_in_ball(new_xyz, xyz, radius, neighbors_limit=40):
    # inner = 2 * torch.matmul(new_xyz, xyz.transpose(1, 2))  # (B, N, M)
    # xx = torch.sum(new_xyz ** 2, dim=-1, keepdim=True)  # (B, N, 1)
    # yy = torch.sum(xyz ** 2, dim=-1, keepdim=True)  # (B, M, 1)
    # diff = -xx + inner - yy.transpose(1, 2)  # (B, N, M)

    diff = torch.sum(torch.square(new_xyz.unsqueeze(2) - xyz.unsqueeze(1)), dim=-1)
    # max_num = min(torch.max(torch.sum(diff < radius ** 2, dim=-1).view(-1)).item(), neighbors_limit)
    # print(torch.mean(torch.sum(diff < radius ** 2, dim=-1, dtype=torch.float32).view(-1)))
    max_num = min(neighbors_limit, xyz.shape[1])
    # print(max_num)
    diff *= -1

    knn_distances, knn_idx = diff.topk(k=max_num, dim=-1)
    knn_distances *= -1

    idx = torch.ones_like(knn_idx) * xyz.shape[1]
    mask = knn_distances < radius ** 2
    idx[mask] = knn_idx[mask]

    return torch.unique(idx, dim=-1)


# def fps(src: torch.Tensor, k=None, random_start=False):
#     device = src.device
#     N, C = src.shape
#     npoint = k
#
#     centroids = torch.zeros(npoint, dtype=torch.long).to(device)
#     distance = torch.ones(N, dtype=torch.long).to(device) * 1e10
#     if random_start:
#         farthest = torch.randint(0, N, (1, ), dtype=torch.long).to(device)
#     else:
#         barycenter = torch.sum(src, 0)
#         barycenter = barycenter / N
#         barycenter = barycenter.view(1, C)
#         dist = torch.sum(torch.pow(src - barycenter, 2), -1)
#         farthest = torch.max(dist, 0)[1]
#
#     for i in range(npoint):
#         centroids[i] = farthest
#         centroid = src[farthest, :].view(1, C)
#         dist = torch.sum((src - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = torch.max(distance, -1)[1]
#
#     return centroids


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret


def batch_fps(src: torch.Tensor, k=None, random_start=False, mask=None):
    device = src.device
    B, N, C = src.shape
    npoint = k

    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N, dtype=torch.long).to(device) * 1e10
    if random_start:
        farthest = torch.randint(0, N, (B, 1), dtype=torch.long).to(device)
    else:
        barycenter = torch.sum(src, 1, keepdim=True) / N
        dist = torch.sum(torch.square(src - barycenter), -1)  # (B, N)
        farthest = torch.max(dist, 1)[1]  # (B, )

    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = torch.gather(src, dim=1, index=farthest[:, None, None].expand(-1, -1, C))
        dist = torch.sum(torch.square(src - centroid), -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, 1)[1]

    return centroids
