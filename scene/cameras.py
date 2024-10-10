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

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, se3_to_SE3

# Define a class named Camera_Pose. The code is based on the camera_transf class in iNeRF. You can refer to iNeRF at https://github.com/salykovaa/inerf.
class Camera_Pose(nn.Module):
    def __init__(self, camera):
        super(Camera_Pose, self).__init__()

        self.FoVx = camera.FoVx
        self.FoVy = camera.FoVy

        self.image_width = camera.image_width
        self.image_height = camera.image_height

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = camera.trans
        self.scale = camera.scale
        self.cov_offset = 0
    
        self.w = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).to(camera.device))
        self.v = nn.Parameter(torch.normal(0., 1e-6, size=(3,)).to(camera.device))
        
        self.forward(camera.w2c)
    
    def forward(self, start_pose_w2c):
        deltaT=se3_to_SE3(self.w,self.v)
        self.pose_w2c = torch.matmul(deltaT, start_pose_w2c.inverse()).inverse()
        self.update()
    
    def current_campose_c2w(self):
        return self.pose_w2c.inverse().clone().cpu().detach().numpy()

    def update(self):
        self.world_view_transform = self.pose_w2c.transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]