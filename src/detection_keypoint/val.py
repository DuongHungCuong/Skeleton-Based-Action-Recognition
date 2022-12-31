import argparse
import json
import os, os.path as osp
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).absolute()
sys.path.append(FILE.parents[0].as_posix())  # add kapao/ to path

from utils.augmentations import letterbox
from utils.general import check_dataset, check_file, check_img_size, \
    non_max_suppression_kp, scale_coords, set_logging, colorstr
import cv2

PAD_COLOR = (114 / 255, 114 / 255, 114 / 255)


def run_nms(data, model_out):
    if data['iou_thres'] == data['iou_thres_kp'] and data['conf_thres_kp'] >= data['conf_thres']:
        # Combined NMS saves ~0.2 ms / image
        dets = non_max_suppression_kp(model_out, data['conf_thres'], data['iou_thres'], num_coords=data['num_coords'])
        person_dets = [d[d[:, 5] == 0] for d in dets]
        kp_dets = [d[d[:, 4] >= data['conf_thres_kp']] for d in dets]
        kp_dets = [d[d[:, 5] > 0] for d in kp_dets]
    else:
        person_dets = non_max_suppression_kp(model_out, data['conf_thres'], data['iou_thres'],
                                             classes=[0],
                                             num_coords=data['num_coords'])

        kp_dets = non_max_suppression_kp(model_out, data['conf_thres_kp'], data['iou_thres_kp'],
                                         classes=list(range(1, 1 + len(data['kp_flip']))),
                                         num_coords=data['num_coords'])
    return person_dets, kp_dets


def post_process_batch(data, imgs, paths, shapes, person_dets, kp_dets,
                       two_stage=False, pad=0, device='cpu', model=None, origins=None):

    batch_bboxes, batch_poses, batch_scores, batch_ids = [], [], [], []
    n_fused = np.zeros(data['num_coords'] // 2)

    if origins is None:  # used only for two-stage inference so set to 0 if None
        origins = [np.array([0, 0, 0]) for _ in range(len(person_dets))]

    # process each image in batch
    for si, (pd, kpd, origin) in enumerate(zip(person_dets, kp_dets, origins)):
        nd = pd.shape[0]
        nkp = kpd.shape[0]

        if nd:
            path, shape = Path(paths[si]) if len(paths) else '', shapes[si][0]
            img_id = int(osp.splitext(osp.split(path)[-1])[0]) if path else si

            # TWO-STAGE INFERENCE (EXPERIMENTAL)
            if two_stage:
                gs = max(int(model.stride.max()), 32)  # grid size (max stride)
                crops, origins, crop_shapes = [], [], []

                for bbox in pd[:, :4].cpu().numpy():
                    x1, y1, x2, y2 = map(int, map(round, bbox))
                    x1, x2 = max(x1, 0), min(x2, data['imgsz'])
                    y1, y2 = max(y1, 0), min(y2, data['imgsz'])
                    h0, w0 = y2 - y1, x2 - x1
                    crop_shapes.append([(h0, w0)])
                    crop = np.transpose(imgs[si][:, y1:y2, x1:x2].cpu().numpy(), (1, 2, 0))
                    crop = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=PAD_COLOR)  # add padding
                    h0 += 2 * pad
                    w0 += 2 * pad
                    origins = [np.array([x1 - pad, y1 - pad, 0])]
                    crop_pre = letterbox(crop, data['imgsz'], color=PAD_COLOR, stride=gs, auto=False)[0]
                    crop_input = torch.Tensor(np.transpose(np.expand_dims(crop_pre, axis=0), (0, 3, 1, 2))).to(device)

                    out = model(crop_input, augment=True, kp_flip=data['kp_flip'], scales=data['scales'], flips=data['flips'])[0]
                    person_dets, kp_dets = run_nms(data, out)
                    _, poses, scores, img_ids, _ = post_process_batch(
                        data, crop_input, paths, [[(h0, w0)]], person_dets, kp_dets, device=device, origins=origins)

                    # map back to original image
                    if len(poses):
                        poses = np.stack(poses, axis=0)
                        poses = poses[:, :, :2].reshape(poses.shape[0], -1)
                        poses = scale_coords(imgs[si].shape[1:], poses, shape)
                        poses = poses.reshape(poses.shape[0], data['num_coords'] // 2, 2)
                        poses = np.concatenate((poses, np.zeros((poses.shape[0], data['num_coords'] // 2, 1))), axis=-1)
                    poses = [p for p in poses]  # convert back to list

            # SINGLE-STAGE INFERENCE
            else:
                scores = pd[:, 4].cpu().numpy()  # person detection score
                bboxes = scale_coords(imgs[si].shape[1:], pd[:, :4], shape).round().cpu().numpy()
                poses = scale_coords(imgs[si].shape[1:], pd[:, -data['num_coords']:], shape).cpu().numpy()
                poses = poses.reshape((nd, -data['num_coords'], 2))
                poses = np.concatenate((poses, np.zeros((nd, poses.shape[1], 1))), axis=-1)

                if data['use_kp_dets'] and nkp:
                    mask = scores > data['conf_thres_kp_person']
                    poses_mask = poses[mask]

                    if len(poses_mask):
                        kpd2 = kpd.clone()
                        kpd[:, :4] = scale_coords(imgs[si].shape[1:], kpd2[:, :4], shape)
                        kpd = kpd[:, :6].cpu()

                        for x1, y1, x2, y2, conf, cls in kpd:
                            x, y = np.mean((x1, x2)), np.mean((y1, y2))
                            pose_kps = poses_mask[:, int(cls - 1)]
                            dist = np.linalg.norm(pose_kps[:, :2] - np.array([[x, y]]), axis=-1)
                            kp_match = np.argmin(dist)
                            if conf > pose_kps[kp_match, 2] and dist[kp_match] < data['overwrite_tol']:
                                pose_kps[kp_match] = [x, y, conf]
                                if data['count_fused']:
                                    n_fused[int(cls - 1)] += 1
                        poses[mask] = poses_mask

                poses = [p + origin for p in poses]

            batch_bboxes.extend(bboxes)
            batch_poses.extend(poses)
            batch_scores.extend(scores)
            batch_ids.extend([img_id] * len(scores))

    return batch_bboxes, batch_poses, batch_scores, batch_ids, n_fused


