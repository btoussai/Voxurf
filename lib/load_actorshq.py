import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import imageio
import xml.etree.cElementTree as ET
import cv2
import multiprocessing

def read_img(mask_path, rgb_path, reso_level):
    mask = imageio.imread(mask_path)
    rgb = imageio.imread(rgb_path)
    if reso_level > 1:
        mask = cv2.resize(mask, (0,0), fx=1/reso_level, fy=1/reso_level, interpolation=cv2.INTER_AREA)
        mask = mask[...,None]
        rgb = cv2.resize(rgb, (0,0), fx=1/reso_level, fy=1/reso_level, interpolation=cv2.INTER_AREA)

    rgb[mask[:,:,0] < 128] = 0
    return rgb, mask

def load_actorshq_data(basedir, normalize=True, reso_level=2, mask=True, white_bg=True):

    scalemat = np.eye(4)
    xyz_min_fine = np.array([0, -0.7, -0.2])
    xyz_max_fine = np.array([1.5, 0.6, 2.1])

    scalemat = np.eye(4)
    # scalemat[:3, 3] = -(xyz_max_fine - xyz_min_fine) / 2
    # scalemat = np.linalg.inv(scalemat)

    NCAMS = 160
    rgb_paths = [basedir + f"/rgbs1x/{i+1:03d}.png" for i in range(NCAMS)]
    mask_paths = [basedir + f"/masks1x/{i+1:03d}.png" for i in range(NCAMS)]

    root = ET.parse(basedir + "/calib.xml").getroot()
    matrices = []
    for c in root.findall("Camera"):
        K = np.array([float(f) for f in c.find("K").text.split(' ')]).reshape((3,3))
        R = np.array([float(f) for f in c.find("R").text.split(' ')]).reshape((3,3))
        T = np.array([float(f) for f in c.find("T").text.split(' ')]).reshape((3,1))
        pose = np.eye(4)
        pose[:3, :3] = R.T # From camera to world rotation
        pose[:3, 3] = (-R.T @ T)[:, 0] # Camera center in world pose
        pose = pose @ scalemat
        matrices.append((K, pose))


    read_img(mask_paths[0], rgb_paths[0], reso_level)
    all_imgs = []
    all_masks = []
    with multiprocessing.Pool(processes=48) as pool:
        masks_rgbs = pool.starmap(read_img, zip(mask_paths, rgb_paths, [reso_level]*NCAMS))
        all_imgs, all_masks = zip(*masks_rgbs)

    

    all_intrinsics = []
    all_poses = []
    for i in range(NCAMS):
        print(f"Loading image {i}...")
        all_intrinsics.append(matrices[i][0])
        all_poses.append(matrices[i][1])
        # if len(mask_paths) > 0:
        #     mask_ = (cv2.imread(mask_paths[i]) / 255.).astype(np.float32)
        #     if mask_.ndim == 3:
        #         all_masks.append(mask_[...,:3])
        #     else:
        #         all_masks.append(mask_[...,None])
        # all_imgs.append((cv2.imread(rgb_paths[i]) / 255.).astype(np.float32))
    imgs = np.stack(all_imgs, 0).astype(np.float32) / 255.
    poses = np.stack(all_poses, 0)
    H, W = imgs[0].shape[:2]
    focal = all_intrinsics[0][0,0]
    all_intrinsics = np.array(all_intrinsics)
    print("Date original shape: ", H, W)
    masks = np.stack(all_masks, 0).astype(np.float32) / 255.
    # if mask:
    #     assert len(mask_paths) > 0
    #     bg = 1. if white_bg else 0.
    #     imgs = imgs * masks + bg * (1 - masks)
    if reso_level > 1:
        H, W = int(H / reso_level), int(W / reso_level)
        # imgs =  F.interpolate(torch.from_numpy(imgs).permute(0,3,1,2), size=(H, W)).permute(0,2,3,1).numpy()
        # if masks is not None:
        #     masks =  F.interpolate(torch.from_numpy(masks).permute(0,3,1,2), size=(H, W)).permute(0,2,3,1).numpy()
        all_intrinsics[:, :2, :] /= reso_level
        focal /= reso_level

    i_train = np.arange(len(imgs))
    i_test = i_train
    i_val = i_test

    i_split = [np.array(i_train), np.array(i_val), np.array(i_test)]

    render_poses = poses[i_split[-1]]
    return imgs, poses, render_poses, [H, W, focal], all_intrinsics, i_split, scalemat, masks
