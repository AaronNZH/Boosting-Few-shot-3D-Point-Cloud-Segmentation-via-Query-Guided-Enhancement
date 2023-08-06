"""
Adapted from https://github.com/Na-Z/attMPTI/tree/main/models/mpti.py
Author: Zhenhua Ning, 2022
"""

import numpy as np
import faiss

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import fps

from models.dgcnn import DGCNN
from models.attention import MultiHeadAttention, SelfAttention


class BaseLearner(nn.Module):
    """The class for inner loop."""

    def __init__(self, in_channels, params):
        super(BaseLearner, self).__init__()

        self.num_convs = len(params)
        self.convs = nn.ModuleList()

        for i in range(self.num_convs):
            if i == 0:
                in_dim = in_channels
            else:
                in_dim = params[i - 1]
            self.convs.append(nn.Sequential(
                nn.Conv1d(in_dim, params[i], 1),
                nn.BatchNorm1d(params[i])))

    def forward(self, x):
        for i in range(self.num_convs):
            x = self.convs[i](x)
            if i != self.num_convs - 1:
                x = F.relu(x)
        return x


class MultiPrototypeTransductiveInference(nn.Module):
    def __init__(self, args):
        super(MultiPrototypeTransductiveInference, self).__init__()
        # self.gpu_id = args.gpu_id
        self.n_way = args.n_way
        self.k_shot = args.k_shot
        self.in_channels = args.pc_in_dim
        self.n_points = args.pc_npts
        self.n_subprototypes = args.n_subprototypes
        self.teacher_n_subprototypes = args.n_teacher_subprototypes
        self.k_connect = args.k_connect
        self.sigma = args.mpti_sigma

        self.n_classes = self.n_way + 1

        self.encoder = DGCNN(args.edgeconv_widths, args.dgcnn_mlp_widths, args.pc_in_dim, k=args.dgcnn_k)
        self.base_learner = BaseLearner(args.dgcnn_mlp_widths[-1], args.base_widths)
        self.att_learner = SelfAttention(args.dgcnn_mlp_widths[-1], args.output_dim)

        self.feat_dim = args.edgeconv_widths[0][-1] + args.output_dim + args.base_widths[-1]

        self.cross_align = MultiHeadAttention(in_channel=self.feat_dim, out_channel=self.feat_dim, n_heads=3,
                                              att_dropout=args.dropout_ratio, use_proj=True)

        self.cross_bg_proj = nn.Sequential(nn.Linear(self.feat_dim, self.feat_dim // 2),
                                           nn.ReLU(inplace=True),
                                           nn.Dropout(0.2),
                                           nn.Linear(self.feat_dim // 2, self.feat_dim),
                                           nn.Sigmoid())

    def forward(self, support_x, support_y, query_x, query_y, use_hr=False, use_bpa=False):
        """
        Args:
            support_x: support point clouds with shape (n_way, k_shot, in_channels, num_points)
            support_y: support masks (foreground) with shape (n_way, k_shot, num_points)
            query_x: query point clouds with shape (n_queries, in_channels, num_points)
            query_y: query labels with shape (n_queries, num_points), each point \in {0,..., n_way}
        Return:
            query_pred: query point clouds predicted similarity, shape: (n_queries, n_way+1, num_points)
        """
        support_x = support_x.view(self.n_way * self.k_shot, self.in_channels, self.n_points)
        support_feat = self.getFeatures(support_x)
        support_feat = support_feat.view(self.n_way, self.k_shot, self.feat_dim, self.n_points)
        query_feat = self.getFeatures(query_x)  # (n_queries, feat_dim, num_points)

        # teacher
        if use_hr:
            query_feat_for_teacher = query_feat.unsqueeze(1)
            query_fg_mask = query_y.unsqueeze(1)
            query_bg_mask = torch.logical_not(query_fg_mask)
            query_fg_proto, query_fg_labels = self.getForegroundPrototypes(query_feat_for_teacher, query_fg_mask,
                                                                           k=self.teacher_n_subprototypes)
            query_bg_proto, query_bg_labels = self.getBackgroundPrototypes(query_feat_for_teacher, query_bg_mask,
                                                                           k=self.teacher_n_subprototypes)

            if query_bg_proto is not None and query_bg_labels is not None:
                prototypes = torch.cat((query_bg_proto, query_fg_proto), dim=0)  # (*, feat_dim)
                prototype_labels = torch.cat((query_bg_labels, query_fg_labels), dim=0)  # (*,n_classes)
            else:
                prototypes = query_fg_proto
                prototype_labels = query_fg_labels

            query_feat_for_teacher = query_feat_for_teacher.squeeze(1).transpose(1, 2).contiguous().view(-1,
                                                                                                         self.feat_dim)
            teacher_pred = self.get_pred(prototypes, prototype_labels, query_feat_for_teacher)

        query_feat = query_feat.transpose(1, 2).contiguous().view(-1, self.feat_dim)  # (n_queries*num_points, feat_dim)
        fg_mask = support_y
        bg_mask = torch.logical_not(support_y)

        fg_prototypes, fg_labels = self.getForegroundPrototypes(support_feat, fg_mask, k=self.n_subprototypes)
        bg_prototypes, bg_labels = self.getBackgroundPrototypes(support_feat, bg_mask, k=self.n_subprototypes)

        if bg_prototypes is not None and bg_labels is not None:
            if use_bpa:
                bg_prototypes = bg_prototypes.unsqueeze(0)
                fg_prototypes = fg_prototypes.unsqueeze(0)
                query_feat = query_feat.unsqueeze(0)

                cross_bg_prototypes1 = self.cross_align([bg_prototypes, query_feat, query_feat])
                cross_bg_prototypes2 = self.cross_align([bg_prototypes, fg_prototypes, fg_prototypes])
                cross_bg_prototypes3 = cross_bg_prototypes1 * (1 - self.cross_bg_proj(cross_bg_prototypes2))
                # cross_bg_prototypes3 = self.additional_proj(cross_bg_prototypes3)
                bg_prototypes = self.cross_align([bg_prototypes, cross_bg_prototypes3, cross_bg_prototypes3])

                bg_prototypes = bg_prototypes.squeeze(0)
                fg_prototypes = fg_prototypes.squeeze(0)
                query_feat = query_feat.squeeze(0)

            prototypes = torch.cat((bg_prototypes, fg_prototypes), dim=0)  # (*, feat_dim)
            prototype_labels = torch.cat((bg_labels, fg_labels), dim=0)  # (*, n_classes)
        else:
            prototypes = fg_prototypes
            prototype_labels = fg_labels

        query_pred = self.get_pred(prototypes, prototype_labels, query_feat)
        loss = self.computeCrossEntropyLoss(query_pred, query_y)

        if use_hr:
            teacher_pred_loss = self.computeCrossEntropyLoss(teacher_pred, query_y)
            distill_loss = self.get_distill_loss(query_pred, teacher_pred, query_y, tau=1)
            # We observed that training without student CE loss is more stable
            loss = teacher_pred_loss + distill_loss

        return query_pred, loss

    def getFeatures(self, x):
        """
        Forward the input data to network and generate features
        :param x: input data with shape (B, C_in, L)
        :return: features with shape (B, C_out, L)
        """
        feat_level1, feat_level2 = self.encoder(x)
        feat_level3 = self.base_learner(feat_level2)
        att_feat = self.att_learner(feat_level2)
        return torch.cat((feat_level1, att_feat, feat_level3), dim=1)

    def getMutiplePrototypes(self, feat, k):
        """
        Extract multiple prototypes by points separation and assembly

        Args:
            feat: input point features, shape:(n_points, feat_dim)
        Return:
            prototypes: output prototypes, shape: (n_prototypes, feat_dim)
        """
        # sample k seeds as initial centers with Farthest Point Sampling (FPS)
        n = feat.shape[0]
        assert n > 0
        ratio = k / n
        if ratio < 1:
            fps_index = fps(feat, None, ratio=ratio, random_start=False).unique()
            num_prototypes = len(fps_index)
            farthest_seeds = feat[fps_index]

            # compute the point-to-seed distance
            distances = F.pairwise_distance(feat[:, None, :], farthest_seeds[None, :, :],
                                            p=2)  # (n_points, n_prototypes)

            # hard assignment for each point
            assignments = torch.argmin(distances, dim=1)  # (n_points,)

            # aggregating each cluster to form prototype
            prototypes = torch.zeros((num_prototypes, self.feat_dim)).cuda()
            for i in range(num_prototypes):
                selected = torch.nonzero(assignments == i).squeeze(1)
                selected = feat[selected, :]
                prototypes[i] = selected.mean(0)
            return prototypes
        else:
            return feat

    def getForegroundPrototypes(self, feats, masks, k=100):
        """
        Extract foreground prototypes for each class via clustering point features within that class

        Args:
            feats: input support features, shape: (n_way, k_shot, feat_dim, num_points)
            masks: foreground binary masks, shape: (n_way, k_shot, num_points)
        Return:
            prototypes: foreground prototypes, shape: (n_way*k, feat_dim)
            labels: foreground prototype labels (one-hot), shape: (n_way*k, n_way+1)
        """
        prototypes = []
        labels = []
        for i in range(self.n_way):
            # extract point features belonging to current foreground class
            feat = feats[i, ...].transpose(1, 2).contiguous().view(-1, self.feat_dim)  # (k_shot*num_points, feat_dim)
            index = torch.nonzero(torch.eq(masks[i, ...].view(-1), i + 1)).squeeze(1)  # (k_shot*num_points,)
            feat = feat[index]
            class_prototypes = self.getMutiplePrototypes(feat, k)
            prototypes.append(class_prototypes)

            # construct label matrix
            class_labels = torch.zeros(class_prototypes.shape[0], self.n_classes)
            class_labels[:, i + 1] = 1
            labels.append(class_labels)

        prototypes = torch.cat(prototypes, dim=0)
        labels = torch.cat(labels, dim=0)

        return prototypes, labels

    def getBackgroundPrototypes(self, feats, masks, k=100):
        """
        Extract background prototypes via clustering point features within background class

        Args:
            feats: input support features, shape: (n_way, k_shot, feat_dim, num_points)
            masks: background binary masks, shape: (n_way, k_shot, num_points)
        Return:
            prototypes: background prototypes, shape: (k, feat_dim)
            labels: background prototype labels (one-hot), shape: (k, n_way+1)
        """
        feats = feats.transpose(2, 3).contiguous().view(-1, self.feat_dim)
        index = torch.nonzero(masks.reshape(-1)).squeeze(1)
        feat = feats[index]
        # in case this support set does not contain background points..
        if feat.shape[0] != 0:
            prototypes = self.getMutiplePrototypes(feat, k)

            labels = torch.zeros(prototypes.shape[0], self.n_classes)
            labels[:, 0] = 1

            return prototypes, labels
        else:
            return None, None

    def calculateLocalConstrainedAffinity(self, node_feat, k=200, method='gaussian'):
        """
        Calculate the Affinity matrix of the nearest neighbor graph constructed by prototypes and query points,
        It is a efficient way when the number of nodes in the graph is too large.

        Args:
            node_feat: input node features
                  shape: (num_nodes, feat_dim)
            k: the number of nearest neighbors for each node to compute the similarity
            method: 'cosine' or 'gaussian', different similarity function
        Return:
            A: Affinity matrix with zero diagonal, shape: (num_nodes, num_nodes)
        """
        # kNN search for the graph
        num_nodes = node_feat.shape[0]
        X = node_feat.detach().cpu().numpy()
        # build the index with cpu version
        index = faiss.IndexFlatL2(self.feat_dim)
        index.add(X)
        _, I = index.search(X, k + 1)
        I = torch.from_numpy(I[:, 1:]).cuda()  # (num_nodes, k)

        # create the affinity matrix
        knn_idx = I.unsqueeze(2).expand(-1, -1, self.feat_dim).contiguous().view(-1, self.feat_dim)
        knn_feat = torch.gather(node_feat, dim=0, index=knn_idx).contiguous().view(num_nodes, k, self.feat_dim)

        if method == 'cosine':
            knn_similarity = F.cosine_similarity(node_feat[:, None, :], knn_feat, dim=2)
        elif method == 'gaussian':
            dist = F.pairwise_distance(node_feat[:, None, :], knn_feat, p=2)
            knn_similarity = torch.exp(-0.5 * (dist / self.sigma) ** 2)
        else:
            raise NotImplementedError('Error! Distance computation method (%s) is unknown!' % method)

        A = torch.zeros(num_nodes, num_nodes, dtype=torch.float).cuda()
        A = A.scatter_(1, I, knn_similarity)
        A = A + A.transpose(0, 1)

        identity_matrix = torch.eye(num_nodes, requires_grad=False).cuda()
        A = A * (1 - identity_matrix)
        return A

    def label_propagate(self, A, Y, alpha=0.99):
        """ Label Propagation, refer to "Learning with Local and Global Consistency" NeurIPs 2003
        Args:
            A: Affinity matrix with zero diagonal, shape: (num_nodes, num_nodes)
            Y: initial label matrix, shape: (num_nodes, n_way+1)
            alpha: a parameter to control the amount of propagated info.
        Return:
            Z: label predictions, shape: (num_nodes, n_way+1)
        """
        # compute symmetrically normalized matrix S
        eps = np.finfo(float).eps
        D = A.sum(1)  # (num_nodes,)
        D_sqrt_inv = torch.sqrt(1.0 / (D + eps))
        D_sqrt_inv = torch.diag_embed(D_sqrt_inv).cuda()
        S = D_sqrt_inv @ A @ D_sqrt_inv

        # close form solution
        Z = torch.inverse(torch.eye(S.shape[0]).cuda() - alpha * S + eps) @ Y
        return Z

    def get_pred(self, prototypes, prototype_labels, query_feat):
        num_prototypes = prototypes.shape[0]

        # construct label matrix Y, with Y_ij = 1 if x_i is from the support set and labeled as y_i = j, otherwise Y_ij = 0.
        num_nodes = num_prototypes + query_feat.shape[0]  # number of node of partial observed graph
        Y = torch.zeros(num_nodes, self.n_classes).cuda()
        Y[:num_prototypes] = prototype_labels

        # construct feat matrix F
        node_feat = torch.cat((prototypes, query_feat), dim=0)  # (num_nodes, feat_dim)

        # label propagation
        A = self.calculateLocalConstrainedAffinity(node_feat, k=self.k_connect)
        Z = self.label_propagate(A, Y)  # (num_nodes, n_way+1)

        query_pred = Z[num_prototypes:, :]  # (n_queries*num_points, n_way+1)
        query_pred = query_pred.view(-1, self.n_points, self.n_classes).transpose(1,
                                                                                  2)  # (n_queries, n_way+1, num_points)

        return query_pred

    def get_distill_loss(self, student_pred, teacher_pred, gt, tau=1):
        """
        Args:
            teacher_pred: [b, c, n]
            student_pred: [b, c, n]
            gt: [b, n]
        """
        loss = 0.
        student_pred = torch.log_softmax(student_pred * tau, dim=1)
        teacher_pred = teacher_pred.detach()
        teacher_pred = torch.softmax(teacher_pred * tau, dim=1)
        gt = gt.unsqueeze(1)  # [b, 1, n]
        cross_entropy = -1 * torch.mul(teacher_pred, student_pred)  # [b, c, n]

        for i in range(self.n_classes):
            mask = torch.eq(gt, i)
            cur_loss = (cross_entropy * mask).sum() / (mask.sum() + 1e-5)
            loss = loss + cur_loss

        return loss / self.n_classes

    def computeCrossEntropyLoss(self, query_logits, query_labels):
        """ Calculate the CrossEntropy Loss for query set
        """
        return F.cross_entropy(query_logits, query_labels)
