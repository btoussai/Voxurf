import torch
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import os
from scipy.spatial.transform import Rotation as Rot
from scipy.spatial.transform import Slerp
import imageio
import xml.etree.cElementTree as ET

def load_mvmannequins_data(basedir, normalize=True, reso_level=2, mask=True, white_bg=True):

    scalemat = np.eye(4)
    # xyz_min_fine = np.array([0, -0.7, -0.2])
    # xyz_max_fine = np.array([1.5, 0.6, 2.1])

    # scalemat = np.eye(4)
    # scalemat[:3, 3] = -(xyz_max_fine - xyz_min_fine) / 2
    # scalemat = np.linalg.inv(scalemat)

    NCAMS = 68
    rgb_paths = [basedir + f"/ImagesUndistorted/cam-{i+1}.png" for i in range(NCAMS)]
    mask_paths = [basedir + f"/Masks/cam-{i+1}.png" for i in range(NCAMS)]

    root = ET.parse(basedir + "/calibration_undistorted.xml").getroot()
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

    all_intrinsics = []
    all_poses = []
    all_imgs = []
    all_masks = []
    for i in range(NCAMS):
        print(f"Loading image {i}...")
        all_intrinsics.append(matrices[i][0])
        all_poses.append(matrices[i][1])
        if len(mask_paths) > 0:
            mask_ = (imageio.imread(mask_paths[i]) / 255.).astype(np.float32)
            if mask_.ndim == 3:
                all_masks.append(mask_[...,:3])
            else:
                all_masks.append(mask_[...,None])
        all_imgs.append((imageio.imread(rgb_paths[i]) / 255.).astype(np.float32))
    imgs = np.stack(all_imgs, 0)
    poses = np.stack(all_poses, 0)
    H, W = imgs[0].shape[:2]
    focal = all_intrinsics[0][0,0]
    all_intrinsics = np.array(all_intrinsics)
    print("Date original shape: ", H, W)
    masks = np.stack(all_masks, 0)
    if mask:
        assert len(mask_paths) > 0
        bg = 1. if white_bg else 0.
        imgs = imgs * masks + bg * (1 - masks)
    if reso_level > 1:
        H, W = int(H / reso_level), int(W / reso_level)
        imgs =  F.interpolate(torch.from_numpy(imgs).permute(0,3,1,2), size=(H, W)).permute(0,2,3,1).numpy()
        if masks is not None:
            masks =  F.interpolate(torch.from_numpy(masks).permute(0,3,1,2), size=(H, W)).permute(0,2,3,1).numpy()
        all_intrinsics[:, :2, :] /= reso_level
        focal /= reso_level

    i_train = np.arange(len(imgs))
    i_test = i_train
    i_val = i_test

    i_split = [np.array(i_train), np.array(i_val), np.array(i_test)]

    render_poses = poses[i_split[-1]]
    return imgs, poses, render_poses, [H, W, focal], all_intrinsics, i_split, scalemat, masks
