#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# from scene.cameras import Camera
import os
import torch
from PIL import Image
from torch import nn
import numpy as np
from copy import deepcopy
# from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal, focal2fov, getProjectionMatrix, getProjectionMatrix2, getWorld2View2, getWorld2View
from utils.image_utils import PIL2Torch
from utils.dataset_utils import readCamerasFromTransforms, readColmapCameras, readColmapSceneInfo
from utils.general_utils import PILtoTorch

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, image_path, image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()
        self.FoVx = FoVx
        self.FoVy = FoVy

        self.R = R
        self.T = T

        self.image_path = image_path
        self.image_name = image_name
        
        self.trans = trans
        self.scale = scale

        self.zfar = 100.0
        self.znear = 0.01

        self.uid = uid
        self.device = data_device
        self.colmap_id = colmap_id

        self.original_image = image.clamp(0.0, 1.0).to(self.device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

    @property
    def w2c(self):
        transform = torch.eye(4).to(self.device)
        transform[:3, :3] = torch.from_numpy(self.R.T)
        transform[:3, 3] = torch.from_numpy(self.T)
        return transform
    
    @property
    def c2w(self):
        transform = torch.eye(4).to(self.device)
        transform[:3, :3] = torch.from_numpy(self.R.T)
        transform[:3, 3] = torch.from_numpy(self.T)
        transform = torch.linalg.inv(transform)
        return transform
    
    @property
    def t(self):
        T = np.eye(4)
        T[:3, :3] = self.R.T
        T[:3, 3] = self.T
        T = np.linalg.inv(T)
        return T[:3, 3]

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if False:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    # WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, image_path=cam_info.image_path,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)

def  cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def load_cameras(data_dir, white_background):
    if os.path.exists(os.path.join(data_dir, "transforms_train.json")):
        cam_infos = readCamerasFromTransforms(data_dir, "transforms_test.json", white_background)
    elif os.path.exists(os.path.join(data_dir, "sparse")):
        # cam_infos = readColmapCameras()
        cam_infos = readColmapSceneInfo(data_dir, "images", True).test_cameras
    cameraList = cameraList_from_camInfos(cam_infos)
    cameras = {}
    for camera in cameraList:
        cameras[camera.image_name] = camera
    return cameras

rot_psi = lambda phi: np.array([
        [1, 0, 0, 0],
        [0, np.cos(phi), -np.sin(phi), 0],
        [0, np.sin(phi), np.cos(phi), 0],
        [0, 0, 0, 1]])

rot_theta = lambda th: np.array([
        [np.cos(th), 0, -np.sin(th), 0],
        [0, 1, 0, 0],
        [np.sin(th), 0, np.cos(th), 0],
        [0, 0, 0, 1]])

rot_phi = lambda psi: np.array([
        [np.cos(psi), -np.sin(psi), 0, 0],
        [np.sin(psi), np.cos(psi), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])

def trans_t(tx, ty, tz):
    T = np.array([
        [1, 0, 0, tx],
        [0, 1, 0, ty],
        [0, 0, 1, tz],
        [0, 0, 0, 1]
    ])
    return T

def initial_view(target_view, delta_rot, delta_trans):

    target_pose_c2w = target_view.c2w.cpu().numpy()

    start_pose_c2w = trans_t(delta_trans[0],delta_trans[1],delta_trans[2]) @ rot_phi(delta_rot[0]/180.*np.pi) @ rot_theta(delta_rot[1]/180.*np.pi) @ rot_psi(delta_rot[2]/180.*np.pi)  @ target_pose_c2w
    start_pose_w2c = np.linalg.inv(start_pose_c2w)

    new_view = deepcopy(target_view)
    new_view.R = start_pose_w2c[:3, :3].T
    new_view.T = start_pose_w2c[:3, 3]
    return new_view

def sampling_view(curr_view, delta_rot):
    curr_pose_c2w = curr_view.c2w.cpu().numpy()

    sample_pose_c2w = rot_psi(-delta_rot[0]/180.*np.pi) @ rot_theta(-delta_rot[1]/180.*np.pi) @ rot_phi(-delta_rot[2]/180.*np.pi) @ curr_pose_c2w
    sample_pose_w2c = np.linalg.inv(sample_pose_c2w)

    new_view = deepcopy(curr_view)
    new_view.R = sample_pose_w2c[:3, :3].T
    new_view.T = sample_pose_w2c[:3, 3]
    return new_view
