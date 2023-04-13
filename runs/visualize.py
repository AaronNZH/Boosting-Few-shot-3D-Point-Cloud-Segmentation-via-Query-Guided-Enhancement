import collections
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
import open3d as o3d
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloaders.loader import sample_K_pointclouds
from dataloaders.loader import MyTestDataset, batch_test_task_collate, MyDataset
from models.mpti import MultiPrototypeTransductiveInference
from utils.checkpoint_util import load_model_checkpoint
from utils.cuda_util import cast_cuda

class2type = {0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'beam', 4: 'column', 5: 'window', 6: 'door', 7: 'table',
                  8: 'chair', 9: 'sofa', 10: 'bookcase', 11: 'board', 12: 'clutter'}
type2class = {class2type[t]: t for t in class2type}


def get_pcd(xyz, label=None, rgb=None, barycenter=None, attn_weight=None):
    if rgb is None:
        rgb = [0.0, 0.0, 0.0]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.paint_uniform_color(rgb)
    color = np.asarray(pcd.colors)

    if label is not None:
        color *= label

    if attn_weight is not None:
        color *= attn_weight[..., None]

    if barycenter is not None:
        color[barycenter] = [0., 1., 0.]

    return pcd


def random_visualize(args):
    TEST_DATASET = MyTestDataset(args.data_path, args.dataset, cvfold=args.cvfold,
                                 num_episode_per_comb=args.n_episode_test,
                                 n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,
                                 num_point=args.pc_npts, pc_attribs=args.pc_attribs, mode='test')
    TRAIN_DATASET = MyDataset(args.data_path, args.dataset, cvfold=args.cvfold, num_episode=args.n_iters,
                              n_way=args.n_way, k_shot=args.k_shot, n_queries=args.n_queries,
                              phase=args.phase, mode='train',
                              num_point=args.pc_npts, pc_attribs=args.pc_attribs,
                              pc_augm=args.pc_augm, pc_augm_config=None)
    TEST_LOADER = DataLoader(TEST_DATASET, batch_size=1, shuffle=True, collate_fn=batch_test_task_collate)
    TRAIN_LOADER = DataLoader(TRAIN_DATASET, batch_size=1, shuffle=True, collate_fn=batch_test_task_collate)
    class2type = {0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'beam', 4: 'column', 5: 'window', 6: 'door', 7: 'table',
                  8: 'chair', 9: 'sofa', 10: 'bookcase', 11: 'board', 12: 'clutter'}

    model = MultiPrototypeTransductiveInference(args)
    model = load_model_checkpoint(model, args.model_checkpoint_path, mode='test')
    data1, sampled_class1 = next(iter(TEST_LOADER))
    data2, sampled_class2 = next(iter(TEST_LOADER))

    if torch.cuda.is_available():
        data1 = cast_cuda(data1)
        data2 = cast_cuda(data2)
        model.cuda()

    # [support_x1, support_y1, query_x1, query_y1] = data1
    # [support_x2, support_y2, query_x2, query_y2] = data2

    support_x, support_y, query_x, query_y = zip(data1, data2)

    support_x = torch.concat(support_x, dim=0)
    support_y = torch.concat(support_y, dim=0)
    query_x = torch.concat(query_x, dim=0)
    query_y = torch.concat(query_y, dim=0)

    print(f"############ query class is {class2type[sampled_class1[0]], class2type[sampled_class2[0]]} ############")

    model.eval()
    with torch.no_grad():
        logits, loss, attn_weight = model(support_x.clone(), support_y.clone(), query_x.clone(), query_y.clone())
        query_pred = F.softmax(logits[0], dim=1).argmax(dim=1).cpu().reshape(-1)

    query_x = query_x[0].transpose(1, 2)[..., :3].cpu().reshape(-1, 3)
    query_y = query_y[0].cpu().reshape(-1)
    tp_points_idx = torch.nonzero(torch.logical_and((query_pred == 1), torch.eq(query_pred, query_y))).squeeze(-1)

    # get the barycenter of ture positive points
    src = query_x[tp_points_idx]
    barycenter = torch.sum(src, dim=0, keepdim=True) / src.shape[0]
    dist = torch.sum(torch.pow(src - barycenter, 2), -1)
    barycenter_idx = tp_points_idx[torch.min(dist, dim=0)[1]]

    # [N, N]
    attn_weight = attn_weight.mean(dim=0)[0][barycenter_idx]
    attn_weight = attn_weight - attn_weight.min()
    attn_weight = (attn_weight / attn_weight.max()).cpu().numpy()

    query_x = query_x.numpy()
    query_pred = query_pred.view(-1, 1).numpy()
    query_y = query_y.view(-1, 1).numpy()

    visualize_scene = [get_pcd(query_x, query_pred, rgb=[0., 0., 1.], barycenter=barycenter_idx)]

    query_x[..., 0] += 3
    visualize_scene.append(get_pcd(query_x, query_y, [0., 0.6, 0.]))

    query_x[..., 0] += 3
    visualize_scene.append(get_pcd(query_x, rgb=[1., 0., 0.], attn_weight=attn_weight))

    # support_x[..., 0] += 3
    # visualize_scene.append(get_pcd(support_x, support_y, rgb=[0., 0.6, 0.]))

    # support_x[..., 0] += 3
    # visualize_scene.append(get_pcd(support_x, rgb=[1., 0., 0.], attn_weight=cross_weight_0))

    o3d.visualization.draw_geometries(visualize_scene)


def visualize_Area_1(args):
    data_path = './datasets/S3DIS/blocks_bs1_s1'
    class2scans_file = os.path.join(data_path, 'class2scans.pkl')
    if os.path.exists(class2scans_file):
        with open(class2scans_file, 'rb') as f:
            class2scans = pickle.load(f)
    else:
        print("Error, not class2scans.pkl file")
        exit()

    sampled_type = 'bookcase'
    sampled_class = type2class[sampled_type]
    all_scannames = class2scans[sampled_class].copy()
    support_scan = np.random.choice(all_scannames, 1)
    support_x, support_y = sample_K_pointclouds(data_path, 2048, 'xyzrgbXYZ', False, None, support_scan,
                                                sampled_class, [sampled_class], is_support=True)
    support_x, support_y = torch.from_numpy(support_x.astype(np.float32)).transpose(1, 2).unsqueeze(0), \
                           torch.from_numpy(support_y.astype(np.int32)).unsqueeze(0)

    Area_1_scans = []
    for file_name in os.listdir(os.path.join(data_path, 'data')):
        if 'Area_1_conferenceRoom_1_' in file_name:
            Area_1_scans.append(file_name[:-4])
    print(f"There are {len(Area_1_scans)} scans.")
    total_query_x, total_query_y = sample_K_pointclouds(data_path, 2048, 'xyzrgbXYZ', False, None,
                                                        Area_1_scans,
                                                        sampled_class, [sampled_class], is_support=False)
    total_query_x, total_query_y = torch.from_numpy(total_query_x.astype(np.float32)).transpose(1, 2).unsqueeze(1), \
                                   torch.from_numpy(total_query_y.astype(np.int64)).unsqueeze(1)

    model = MultiPrototypeTransductiveInference(args)
    model = load_model_checkpoint(model, args.model_checkpoint_path, mode='test')

    if torch.cuda.is_available():
        total_query_x = total_query_x.cuda()
        total_query_y = total_query_y.cuda()
        support_x = support_x.cuda()
        support_y = support_y.cuda()
        model.cuda()

    visualize_scene = []
    total_positive = 0
    model.eval()
    for i in tqdm(range(total_query_x.shape[0]), desc='Visualizing'):
        with torch.no_grad():
            query_x, query_y = total_query_x[i], total_query_y[i]
            query_logits, loss = model(support_x.clone(), support_y.clone(), query_x.clone(), query_y.clone())
            query_pred = F.softmax(query_logits, dim=1).argmax(dim=1).reshape(-1, 1).cpu().numpy()

        total_positive += query_pred[query_pred > 0].shape[0]
        query_x = query_x.transpose(1, 2)[..., :3].reshape(-1, 3).cpu().numpy()
        query_y = query_y.reshape(-1, 1).cpu().numpy()

        visualize_scene.append(get_pcd(query_x, query_pred, rgb=[1., 0., 0.]))
        query_x[..., 0] += 3
        visualize_scene.append(get_pcd(query_x, query_y, rgb=[0., 0., 1.]))

    print(f"Total number of points of {sampled_type} is: ", total_positive)
    o3d.visualization.draw_geometries(visualize_scene)


def get_data(sampled_classes, random_sample=True, file_name=None):
    data_path = './datasets/S3DIS/blocks_bs1_s1'
    class2scans_file = os.path.join(data_path, 'class2scans.pkl')
    if os.path.exists(class2scans_file):
        with open(class2scans_file, 'rb') as f:
            class2scans = pickle.load(f)
    else:
        print("Error, not class2scans.pkl file")
        exit()

    for i in range(len(sampled_classes)):
        sampled_classes[i] = type2class[sampled_classes[i]]

    support_ptclouds = []
    support_masks = []
    query_ptclouds = []
    query_labels = []

    support_ptclouds2 = []
    support_masks2 = []
    query_ptclouds2 = []
    query_labels2 = []

    black_list = []  # to store the sampled scan names, in order to prevent sampling one scan several times...
    for sampled_class in sampled_classes:
        all_scannames = class2scans[sampled_class].copy()

        if len(black_list) != 0:
            all_scannames = [x for x in all_scannames if x not in black_list]
        selected_scannames = np.random.choice(all_scannames, 2, replace=False)
        black_list.extend(selected_scannames)
        query_scannames = selected_scannames[:1]
        support_scannames = selected_scannames[1:]

        if file_name:
            query_scannames = [file_name]

        query_ptclouds_one_way, query_labels_one_way = sample_K_pointclouds(data_path, 2048, 'xyzrgbXYZ', False, None,
                                                                            query_scannames,
                                                                            sampled_class, sampled_classes,
                                                                            is_support=False)

        support_ptclouds_one_way, support_masks_one_way = sample_K_pointclouds(data_path, 2048, 'xyzrgbXYZ', False,
                                                                               None, support_scannames,
                                                                               sampled_class, sampled_classes,
                                                                               is_support=True)

        query_ptclouds.append(query_ptclouds_one_way)
        query_labels.append(query_labels_one_way)
        support_ptclouds.append(support_ptclouds_one_way)
        support_masks.append(support_masks_one_way)

    support_ptclouds = np.stack(support_ptclouds, axis=0)
    support_masks = np.stack(support_masks, axis=0)
    query_ptclouds = np.concatenate(query_ptclouds, axis=0)
    query_labels = np.concatenate(query_labels, axis=0)

    support_ptclouds = torch.from_numpy(support_ptclouds.astype(np.float32)).transpose(1, 2).unsqueeze(0)
    support_masks = torch.from_numpy(support_masks.astype(np.int32)).unsqueeze(0)
    query_ptclouds = torch.from_numpy(query_ptclouds.astype(np.float32)).transpose(1, 2)
    query_labels = torch.from_numpy(query_labels.astype(np.int64))

    return support_ptclouds, support_masks, query_ptclouds, query_labels
    #
    # support_type = class_name
    # support_class = type2class[support_type]
    # query_type = class_name
    # query_class = type2class[query_type]
    #
    # all_scannames = class2scans[support_class].copy()
    # selected_scan = np.random.choice(all_scannames, 2, replace=False)
    # support_scan = selected_scan[:1]
    # query_scan = selected_scan[1:]
    #
    # support_x, support_y = sample_K_pointclouds(data_path, 2048, 'xyzrgbXYZ', False, None, support_scan,
    #                                             support_class, [support_class], is_support=True)
    # support_x, support_y = torch.from_numpy(support_x.astype(np.float32)).transpose(1, 2).unsqueeze(0), \
    #                        torch.from_numpy(support_y.astype(np.int32)).unsqueeze(0)
    #
    # query_x, query_y = sample_K_pointclouds(data_path, 2048, 'xyzrgbXYZ', False, None, query_scan,
    #                                         query_class, [query_class], is_support=False)
    # query_x, query_y = torch.from_numpy(query_x.astype(np.float32)).transpose(1, 2), \
    #                    torch.from_numpy(query_y.astype(np.int64))
    #
    # return support_x, support_y, query_x, query_y


def certain_class_visualize(args):
    # ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'column']
    # support_x1, support_y1, query_x1, query_y1 = get_data(['chair'])
    # support_x2, support_y2, query_x2, query_y2 = get_data(['ceiling'])
    #
    # support_x = torch.concat([support_x1, support_x2], dim=0)
    # support_y = torch.concat([support_y1, support_y2], dim=0)
    # query_x = torch.concat([query_x1, query_x2], dim=0)
    # query_y = torch.concat([query_y1, query_y2], dim=0)

    file_name = 'Area_1_office_5_block_10'

    support_x, support_y, query_x, query_y = get_data(['chair'], random_sample=False, file_name=file_name)
    # support_x2, support_y2, query_x2, query_y2 = get_data(['chair'], random_sample=True, file_name=file_name)
    # support_x3, support_y3, query_x3, query_y3 = get_data(['window'], random_sample=False, file_name='Area_1_conferenceRoom_2_block_4')

    model = MultiPrototypeTransductiveInference(args)
    model = load_model_checkpoint(model, args.model_checkpoint_path, mode='test')

    if torch.cuda.is_available():

        support_x = support_x.cuda()
        support_y = support_y.cuda()
        query_x = query_x.cuda()
        query_y = query_y.cuda()

        # support_x2 = support_x2.cuda()
        # support_y2 = support_y2.cuda()
        # query_x2 = query_x2.cuda()
        # query_y2 = query_y2.cuda()

        # support_x3 = support_x3.cuda()
        # support_y3 = support_y3.cuda()
        # query_x3 = query_x3.cuda()
        # query_y3 = query_y3.cuda()

        model.cuda()

    model.eval()
    # with torch.no_grad():
    #     logits1, loss1 = model(support_x1, support_y1, query_x1, query_y1)
    #     query_pred1 = logits1[:, 1, :].reshape(-1, 1).cpu().numpy()
    #     query_pred1[query_pred1 < 0.5] = 0

    # query_x1 = query_x1.transpose(1, 2)[..., :3].reshape(-1, 3).cpu().numpy()
    # query_y1 = query_y1.reshape(-1, 1).cpu().numpy()
    #
    # support_x1 = support_x1.transpose(2, 3)[..., :3].reshape(-1, 3).cpu().numpy()
    # support_y1 = support_y1.reshape(-1, 1).cpu().numpy()

    with torch.no_grad():
        logits, loss = model(support_x, support_y, query_x, query_y)
        query_pred = torch.argmax(logits[:1], dim=1).reshape(-1, 1).cpu().numpy()

        # logits2, loss = model(support_x2, support_y2, query_x2, query_y2)
        # query_pred2 = torch.argmax(logits2[:1], dim=1).reshape(-1, 1).cpu().numpy()
        #
        # logits3, loss = model(support_x3, support_y3, query_x3, query_y3)
        # query_pred3 = torch.argmax(logits3[:1], dim=1).reshape(-1, 1).cpu().numpy()

    # query_x = query_x[:1].transpose(1, 2)[..., :3].reshape(-1, 3).cpu().numpy()
    # query_x2 = query_x2[:1].transpose(1, 2)[..., :3].reshape(-1, 3).cpu().numpy()
    # query_x3 = query_x3[:1].transpose(1, 2)[..., :3].reshape(-1, 3).cpu().numpy()
    # query_y = query_y[:1].reshape(-1, 1).cpu().numpy()

    # visualize_scene = [tmp("./datasets/S3DIS/blocks_bs1_s1/data", file_name, None, 0)]

    support_x = support_x[:1, 0].transpose(1, 2)[..., :3].reshape(-1, 3).cpu().numpy()
    visualize_scene = [get_pcd(support_x)]

    support_x[:, 1] -= 3
    visualize_scene.append(get_pcd(support_x, query_pred, rgb=[1., 0., 0.]))

    # query_x2[:, 1] -= 6
    # visualize_scene.append(get_pcd(query_x2, query_pred2, rgb=[1., 0., 0.]))
    #
    # query_x3[:, 1] += 12
    # visualize_scene.append(get_pcd(query_x3, query_pred3, rgb=[1., 0., 0.]))

    # query_x[..., 0] += 3
    # visualize_scene.append(get_pcd(query_x, query_pred1, rgb=[0., 0., 1.]))
    #
    # query_x[..., 0] += 3
    # visualize_scene.append(get_pcd(query_x, query_pred2, rgb=[0., 0., 1.]))

    o3d.visualization.draw_geometries(visualize_scene)

    # TEST_DATASET = MyTestDataset("./datasets/S3DIS/blocks_bs1_s1", "s3dis", cvfold=0,
    #                              num_episode_per_comb=100,
    #                              n_way=1, k_shot=1, n_queries=1,
    #                              num_point=4096, pc_attribs=None, mode='test')
    #
    # support_x, support_y, query_x, query_y = TEST_DATASET[253][:4]
    #
    # support_x = torch.from_numpy(support_x).transpose(2, 3)
    # support_y = torch.from_numpy(support_y)
    # query_x = torch.from_numpy(query_x).transpose(1, 2)
    # query_y = torch.from_numpy(query_y.astype(np.int64))
    #
    # model = MultiPrototypeTransductiveInference(args)
    # model = load_model_checkpoint(model, args.model_checkpoint_path, mode='test')
    #
    # if torch.cuda.is_available():
    #     support_x = support_x.cuda()
    #     support_y = support_y.cuda()
    #     query_x = query_x.cuda()
    #     query_y = query_y.cuda()
    #
    #     model.cuda()
    #
    # model.eval()
    # with torch.no_grad():
    #     logits, loss = model(support_x, support_y, query_x, query_y)
    #     query_pred = torch.argmax(logits[:1], dim=1).reshape(-1, 1).cpu().numpy()
    #
    # query_x = query_x[:1].transpose(1, 2)[..., :3].reshape(-1, 3).cpu().numpy()
    # query_y = query_y[:1].reshape(-1, 1).cpu().numpy()
    #
    # visualize_scene = [get_pcd(query_x, query_pred, rgb=[0., 0., 1.])]
    #
    # query_x[:, 0] += 3
    # visualize_scene.append(get_pcd(query_x, query_y, rgb=[1., 0., 0.]))
    #
    # o3d.visualization.draw_geometries(visualize_scene)


def tmp(data_path, file_name, sample_class=None, sample_type=0):
    data_path = data_path
    file_name = file_name
    n_points = 2048

    if sample_class is not None:
        sample_class = type2class[sample_class]

    data = np.load(os.path.join(data_path, file_name + ".npy"))
    N = data.shape[0]
    xyz = data[:, :3]
    rgb = data[:, 3:6] / 255.

    if sample_class is None or sample_type == 0:
        sample_idx = np.arange(N)
    elif sample_type == 1:
        valid_points_idx = np.nonzero(data[:, 6] == sample_class)[0]
        valid_num = int(len(valid_points_idx) / float(N) * n_points)
        valid_sample_idx = np.random.choice(valid_points_idx, size=valid_num)
        other_sample_idx = np.random.choice(np.arange(N), size=(n_points - valid_num), replace=(N < n_points))
        sample_idx = np.concatenate([valid_sample_idx, other_sample_idx], axis=0)
    else:
        sample_idx = np.random.choice(np.arange(N), size=n_points, replace=(N < n_points))

    sample = data[sample_idx]
    sample_xyz = xyz[sample_idx]
    sample_rgb = rgb[sample_idx]

    print(f"the number of fg points with sample type {sample_type} is {len(np.nonzero(sample[:, 6] == sample_class)[0])}")

    if sample_type != 0:
        sample_rgb[sample[:, 6] == sample_class] = [1., 0., 0.]

    counter = [class2type[c] for c in sample[:, 6]]
    counter = collections.Counter(counter)
    print(counter)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sample_xyz)
    pcd.paint_uniform_color([1., 1., 1.])
    color = np.asarray(pcd.colors)
    color *= sample_rgb

    return pcd


if __name__ == '__main__':
    class2type = {0: 'ceiling', 1: 'floor', 2: 'wall', 3: 'beam', 4: 'column', 5: 'window', 6: 'door', 7: 'table',
                  8: 'chair', 9: 'sofa', 10: 'bookcase', 11: 'board', 12: 'clutter'}
    type2class = {class2type[t]: t for t in class2type}

    fold_0 = ['beam', 'board', 'bookcase', 'ceiling', 'chair', 'column']
    fold_1 = ['door', 'floor', 'sofa', 'table', 'wall', 'window']

    view = []
    data_path = "../datasets/S3DIS/blocks_bs1_s1/data"
    file_name = 'Area_1_conferenceRoom_2_block_4'
    sample_class = 'window'

    pcd = tmp(data_path, file_name, sample_class, 0)
    print("total number of point in this block is: ", len(pcd.points))
    view.append(pcd)

    pcd = tmp(data_path, file_name, sample_class, 1)
    np.asarray(pcd.points)[:, 0] += 3
    view.append(pcd)

    pcd = tmp(data_path, file_name, sample_class, 2)
    np.asarray(pcd.points)[:, 0] += 6
    view.append(pcd)

    o3d.visualization.draw_geometries(view)
