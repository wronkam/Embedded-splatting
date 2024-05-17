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
from typing import Dict, Tuple

import torch
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange, tqdm

from utils.general_utils import inverse_sigmoid, get_expon_lr_func, build_rotation
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.utils import GFFT1D, PositionalEncoding, plt_scatter, stats
from chamferdist.chamfer import ChamferDistance


def get_if_not_none(dictionary, key, default, map=None):
    if key in dictionary:
        if map is not None:
            return map(dictionary[key])
        else:
            return dictionary[key]
    else:
        return default


def get_net(input_size, output_size, layers, width, activation, bn):
    """
    [B,C x I] -> [B,O]
    """
    activation = {'relu': nn.ReLU(), 'tanh': nn.Tanh(), 'sigmoid': nn.Sigmoid(), 'identity': nn.Identity()}[activation]
    layers_list = []
    sizes = [width for _ in range(layers - 1)]
    for idx, (ins, outs) in enumerate(zip([input_size] + sizes, sizes + [output_size])):
        layers_list.append(nn.Linear(ins, outs))
        if idx + 1 != layers:  # no activation and bs on last layer
            if bn:
                layers_list.append(nn.BatchNorm1d(outs))
            layers_list.append(activation)
    return nn.Sequential(*layers_list)


class FFNet(nn.Module):
    def __init__(self, ff_args):
        super(FFNet, self).__init__()
        self.embed = None
        self.embed_size = None
        self.input_size = get_if_not_none(ff_args, 'input_size', 3, int)
        self.output_size = get_if_not_none(ff_args, 'output_size', 3, int)
        if self.input_size != self.output_size:
            pass
            # Now it works, sorta...
            # raise NotImplementedError('Issues with adding new points')
        self.residual = get_if_not_none(ff_args, 'residual', 0, int)
        self.layers = get_if_not_none(ff_args, 'layers', 4, int)
        self.width = get_if_not_none(ff_args, 'width', 256, int)
        self.activation = get_if_not_none(ff_args, 'activation', 'relu', None)
        self.type = get_if_not_none(ff_args, 'type', 'none', None)
        self.bn = get_if_not_none(ff_args, 'bn', False, bool)
        self.normalize = get_if_not_none(ff_args, 'norm', False, bool)


        self.initial = get_if_not_none(ff_args, 'init', 0, int)
        self.rand_color = get_if_not_none(ff_args, 'rand_color', False, bool)
        print('rand_color', self.rand_color)
        self.plots = get_if_not_none(ff_args, 'plots', False, bool)

        #  embedders [B,S] -> [B,M,S], flatten to [B,M x S], then net [B,M x S] -> [B,O]
        if self.type == 'net':  # just net
            self.embed = nn.Identity()
            self.embed_size = self.input_size
            self.net = get_net(self.input_size, self.output_size, self.layers, self.width, self.activation, self.bn)
        elif self.type == 'fft':  # fft:
            self.embed = lambda x: torch.fft.rfftn(x, s=self.input_size, norm='ortho').float() # no im part
            self.embed_size = self.input_size // 2 + 1
            self.net = get_net(self.embed_size, self.output_size, self.layers, self.width, self.activation, self.bn)
        elif self.type == 'gff':  # gaussian fourier features
            self.embed_size = get_if_not_none(ff_args, 'embed_size', 16, int)
            learnable = get_if_not_none(ff_args, 'learnable', False, bool)
            self.embed = GFFT1D(self.embed_size, learnable=learnable)
            self.net = get_net(self.input_size * self.embed_size, self.output_size,
                               self.layers, self.width, self.activation, self.bn)
        elif self.type in ['positional','pe']:  # positional embedding
            self.embed_size = get_if_not_none(ff_args, 'embed_size', 16, int)
            learnable = get_if_not_none(ff_args, 'learnable', False, bool)
            self.embed = PositionalEncoding(self.embed_size, learnable=learnable)
            self.net = get_net(self.input_size * self.embed_size, self.output_size,
                               self.layers, self.width, self.activation, self.bn)
        elif self.type == 'none':  # nothing
            self.embed = nn.Identity()
            self.net = nn.Identity()
        else:
            raise NotImplementedError("Unkwnown ff_type {}".format(self.type))
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.net = self.net.to(self.device)
        self.scale = torch.nn.Parameter(torch.ones(1,self.output_size)).to(self.device)
        self.transl = torch.nn.Parameter(torch.zeros(1,self.output_size)).to(self.device)
        if self.type not in ['none','fft']:
            self.embed = self.embed.to(self.device)
    def forward(self, x_in: torch.Tensor) -> torch.Tensor:
        x = self.embed(x_in)
        x = x.reshape(x.shape[0], -1)
        if self.normalize:
            x = x/torch.linalg.norm(x,ord=2,dim=-1,keepdim=True)
        x = self.net(x)
        if self.residual >= 0:
            x = nn.functional.tanh(x)
        if self.residual == 1:
            x = x + x_in
        if self.residual >= 0:
            x = x * self.scale + self.transl
        if self.residual == 2:
            x = x + x_in
        return x


def match_points(target:torch.Tensor, ff_net:FFNet) -> torch.Tensor:
    points_num = target.shape[0]
    points = nn.Parameter((torch.rand(points_num,ff_net.input_size).to(target.device)),requires_grad=True)

    with torch.no_grad():
        t_center, t_scale = target.mean(0), (target - target.mean(0)).std(0)
        output = ff_net(points)
        s_center, s_scale = output.mean(0), (output-output.mean(0)).std(0)
        scale = 1.1*t_scale/s_scale
        center = t_center - (output*scale).mean(0)
    ff_net.transl = nn.Parameter(center.unsqueeze(0), requires_grad=True)
    ff_net.scale = nn.Parameter(scale.unsqueeze(0), requires_grad=True)
    with torch.no_grad():
        outputs = ff_net(points)
        print(f"initial: points({outputs.mean(0)},{outputs.std(0)}) target({target.mean(0)},{target.std(0)})")

    assert len(points.shape) == 2
    assert len(target.shape) == 2

    l = [
        {'params': [points], 'lr':  0.00025, "name": "points","weight_decay":0.3},
        {'params': ff_net.parameters(), 'lr': 0.0001, "name": "ff_net","weight_decay":0.3}
    ]
    optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)

    chamfer_dist_criterion = ChamferDistance()
    bar = trange(ff_net.initial)
    losses = []
    for _ in bar:
        optimizer.zero_grad()

        # Forward pass
        outputs = ff_net(points)
        loss = chamfer_dist_criterion(source_cloud=outputs.unsqueeze(0),
                                      target_cloud=target.unsqueeze(0),
                                      bidirectional=True)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        bar.set_postfix({'loss': loss.item()})
        losses.append(torch.log(loss).item())

    with torch.no_grad():
        outputs = ff_net(points)
        print(f"matched: points({outputs.mean(0)},{outputs.std(0)}) target({target.mean(0)},{target.std(0)})")
    if ff_net.plots:
        plt.plot(list(range(len(losses))),losses)
        plt.show()
    return points.detach()


def distil_colors(pcd, old_pcd,colors):
    print("Hello1")
    assert pcd.shape[0] == old_pcd.shape[0] == colors.shape[0]
    print("Hello2")
    new_colors = torch.zeros_like(colors).to(pcd.device)
    max_dist = pcd.std(0).mean()/25
    for idx,point in enumerate(tqdm(pcd,desc="Color distillation")):
        dist = ((old_pcd - point) ** 2).sum(-1)
        # indices
        closest = torch.topk(-dist, k=3)[1]
        dists = dist[closest]
        loc_max_dist = torch.maximum(max_dist, torch.min(dists))
        closest = closest[ dists <= loc_max_dist]
        new_colors[idx] = torch.mean(colors[closest],dim=0)
    print('new_colors')
    stats(new_colors)
    return new_colors
class GaussianModel:

    def setup_functions(self):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize


    def __init__(self, sh_degree : int, ff_args: Dict[str, str] = None):
        self.active_sh_degree = 0
        self.max_sh_degree = sh_degree
        self._xyz = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self.max_radii2D = torch.empty(0)
        self.xyz_gradient_accum = torch.empty(0)
        self.denom = torch.empty(0)
        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions()
        self.ff = False if ff_args is None else True
        if self.ff:
            print("Using FF with args: {}".format(ff_args))
            self.ff_net = FFNet(ff_args)

    def capture(self):
        return (
            self.active_sh_degree,
            self._xyz,
            self._features_dc,
            self._features_rest,
            self._scaling,
            self._rotation,
            self._opacity,
            self.max_radii2D,
            self.xyz_gradient_accum,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
            None if not self.ff else self.ff_net.state_dict()
        )

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._xyz,
        self._features_dc,
        self._features_rest,
        self._scaling,
        self._rotation,
        self._opacity,
        self.max_radii2D,
        xyz_gradient_accum,
        denom,
        opt_dict,
        self.spatial_lr_scale,
        ff_dict) = model_args
        self.training_setup(training_args)
        self.xyz_gradient_accum = xyz_gradient_accum
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)
        if ff_dict is not None:
            self.ff_net.load_state_dict(ff_dict)

    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_xyz(self):
        if self.ff:
            return self.ff_net(self._xyz)
        else:
            return self._xyz

    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)

    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        colors = torch.tensor(np.asarray(pcd.colors)).float().cuda()
        print('pcd.colors',stats(pcd.colors))
        print('colors',stats(colors))

        if self.ff:
            old_fpc = fused_point_cloud.clone()
            fused_point_cloud = match_points(fused_point_cloud,self.ff_net)
            if self.ff_net.plots:
                with torch.no_grad():
                    fig = plt.figure()
                    ax = fig.add_subplot(projection='3d')
                    plt_scatter(torch.tensor(np.asarray(pcd.points)).float(), ax, 'orange', alpha=0.05, sample=2000)
                    plt_scatter(self.ff_net(fused_point_cloud), ax, 'g', alpha=0.05, sample=2000)
                    plt.show()
                    plt.close()
            if self.ff_net.rand_color:
                colors = torch.randn_like(colors).to(colors.device)
            else:
                with torch.no_grad():
                    print('colors',stats(colors))
                    print('colors.c',stats(colors.clone()))
                    colors = distil_colors(self.ff_net(fused_point_cloud),old_fpc,colors.clone())
                    print('final')
                    stats(colors)


        fused_color = RGB2SH(colors)
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = inverse_sigmoid(0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda"))

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense
        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")

        l = [
            {'params': [self._xyz], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "xyz"},
            {'params': [self._features_dc], 'lr': training_args.feature_lr, "name": "f_dc"},
            {'params': [self._features_rest], 'lr': training_args.feature_lr / 20.0, "name": "f_rest"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"}
        ]
        if self.ff:
            l.append({'params': self.ff_net.parameters(), 'lr': training_args.ff_lr, "name": "ff_net"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.xyz_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init * self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final * self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "xyz":
                lr = self.xyz_scheduler_args(iteration)
                param_group['lr'] = lr
                return lr

    def construct_list_of_attributes(self):
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # All channels except the 3 DC
        for i in range(self._features_dc.shape[1] * self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1] * self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        xyz = self.get_xyz.detach().cpu().numpy()
        normals = np.zeros_like(xyz)
        f_dc = self._features_dc.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        f_rest = self._features_rest.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scale = self._scaling.detach().cpu().numpy()
        rotation = self._rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(xyz.shape[0], dtype=dtype_full)
        attributes = np.concatenate((xyz, normals, f_dc, f_rest, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def reset_opacity(self):
        opacities_new = inverse_sigmoid(torch.min(self.get_opacity, torch.ones_like(self.get_opacity) * 0.01))
        optimizable_tensors = self.replace_tensor_to_optimizer(opacities_new, "opacity")
        self._opacity = optimizable_tensors["opacity"]

    def load_ply(self, path):
        plydata = PlyData.read(path)

        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])), axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key=lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names) == 3 * (self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape (P,F*SH_coeffs) to (P, F, SH_coeffs except DC)
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] == 'ff_net':
                continue # FFnet
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]

        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group['name'] == 'ff_net':
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:

                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation):
        d = {"xyz": new_xyz,
             "f_dc": new_features_dc,
             "f_rest": new_features_rest,
             "opacity": new_opacities,
             "scaling": new_scaling,
             "rotation": new_rotation}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]
        self._features_dc = optimizable_tensors["f_dc"]
        self._features_rest = optimizable_tensors["f_rest"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device="cuda")
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device="cuda")

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device="cuda")
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means =torch.zeros((stds.size(0), self._xyz.shape[-1]),device="cuda")
        if self.ff:
            # if ff, than scaling has nothing to do with actual new_xyz
            samples = torch.normal(mean=means, std=torch.zeros_like(means).to(means.device)+1e-3)
        else:
            samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N, 1, 1)
        if self.ff:
            # if ff, than rots have nothing to do with new_xyz
            new_xyz = samples + self._xyz[selected_pts_mask].repeat(N, 1)
        else:
            new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self._xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1)
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1)
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1)

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device="cuda", dtype=bool)))
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values <= self.percent_dense*scene_extent)

        new_xyz = self._xyz[selected_pts_mask]
        new_features_dc = self._features_dc[selected_pts_mask]
        new_features_rest = self._features_rest[selected_pts_mask]
        new_opacities = self._opacity[selected_pts_mask]
        new_scaling = self._scaling[selected_pts_mask]
        new_rotation = self._rotation[selected_pts_mask]

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)

    def densify_and_prune(self, max_grad, min_opacity, extent, max_screen_size):
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0

        self.densify_and_clone(grads, max_grad, extent)
        self.densify_and_split(grads, max_grad, extent)

        prune_mask = (self.get_opacity < min_opacity).squeeze()
        if max_screen_size:
            big_points_vs = self.max_radii2D > max_screen_size
            big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
            prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        self.prune_points(prune_mask)

        torch.cuda.empty_cache()

    def add_densification_stats(self, viewspace_point_tensor, update_filter):
        self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1