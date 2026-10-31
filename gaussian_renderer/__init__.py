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
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh


def _make_rasterizer(viewpoint_camera, pc, bg_color, scaling_modifier, pipe):
    """Create the standard 3DGS rasterizer for a camera.

    Keeping this small setup helper shared by the PSCD auxiliary passes ensures
    that evidence is accumulated with exactly the same projected Gaussian
    footprints and alpha compositing order as the colour renderer.
    """
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )
    return GaussianRasterizer(raster_settings=raster_settings)


def _covariance_inputs(pc, pipe, scaling_modifier, detach=False):
    """Return the covariance representation expected by the rasterizer."""
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
        if detach:
            cov3D_precomp = cov3D_precomp.detach()
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation
        if detach:
            scales = scales.detach()
            rotations = rotations.detach()
    return scales, rotations, cov3D_precomp

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    rasterizer = _make_rasterizer(viewpoint_camera, pc, bg_color, scaling_modifier, pipe)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales, rotations, cov3D_precomp = _covariance_inputs(pc, pipe, scaling_modifier)

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}


@torch.no_grad()
def render_depth(viewpoint_camera, pc : GaussianModel, pipe, scaling_modifier=1.0):
    """Render alpha-composited camera-space depth for PSCD-GS.

    The original CUDA extension does not expose a depth buffer.  Rendering
    camera-space depth and a unit colour with its ordinary alpha compositor is
    equivalent to an expected-depth / accumulated-alpha auxiliary pass, while
    retaining the stock extension and model format.
    """
    device = pc.get_xyz.device
    background = torch.zeros(3, dtype=pc.get_xyz.dtype, device=device)
    rasterizer = _make_rasterizer(viewpoint_camera, pc, background, scaling_modifier, pipe)
    scales, rotations, cov3D_precomp = _covariance_inputs(pc, pipe, scaling_modifier, detach=True)
    means3D = pc.get_xyz.detach()
    opacities = pc.get_opacity.detach()

    ones = torch.ones((means3D.shape[0], 1), dtype=means3D.dtype, device=device)
    homogeneous = torch.cat((means3D, ones), dim=1)
    camera_points = homogeneous @ viewpoint_camera.world_view_transform
    camera_depth = camera_points[:, 2].clamp_min(1e-6)
    depth_colours = camera_depth[:, None].expand(-1, 3).contiguous()
    alpha_colours = torch.ones_like(depth_colours)

    depth_image, _ = rasterizer(
        means3D=means3D, means2D=torch.zeros_like(means3D),
        shs=None, colors_precomp=depth_colours, opacities=opacities,
        scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp)
    alpha_image, _ = rasterizer(
        means3D=means3D, means2D=torch.zeros_like(means3D),
        shs=None, colors_precomp=alpha_colours, opacities=opacities,
        scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp)

    alpha = alpha_image[0].clamp(0.0, 1.0)
    valid = alpha > 1e-6
    depth = torch.where(valid, depth_image[0] / alpha.clamp_min(1e-6), torch.zeros_like(alpha))
    return {"depth": depth, "alpha": alpha, "valid": valid}


def accumulate_contribution_statistics(viewpoint_camera, pc : GaussianModel, pipe, evidence_maps):
    """Accumulate ``sum_u w_i(u) X_c(u)`` for three image-space fields.

    Let the RGB colour of each Gaussian be a differentiable dummy variable.
    The derivative of ``sum(render_c * X_c)`` with respect to Gaussian ``i``'s
    colour is precisely its front-to-back alpha-compositing contribution
    weighted by ``X_c``.  This turns the installed differentiable rasterizer
    into the contribution-weighted aggregation pass used by PSCD-GS without a
    custom CUDA ABI or a dense Gaussian-by-pixel tensor.
    """
    if evidence_maps.ndim != 3 or evidence_maps.shape[0] != 3:
        raise ValueError("evidence_maps must have shape [3, H, W]")
    expected_shape = (int(viewpoint_camera.image_height), int(viewpoint_camera.image_width))
    if tuple(evidence_maps.shape[-2:]) != expected_shape:
        raise ValueError("evidence map resolution must match the training camera")

    device = pc.get_xyz.device
    maps = evidence_maps.to(device=device, dtype=pc.get_xyz.dtype).detach()
    background = torch.zeros(3, dtype=pc.get_xyz.dtype, device=device)
    rasterizer = _make_rasterizer(viewpoint_camera, pc, background, 1.0, pipe)
    scales, rotations, cov3D_precomp = _covariance_inputs(pc, pipe, 1.0, detach=True)
    means3D = pc.get_xyz.detach()
    opacities = pc.get_opacity.detach()

    # This helper may be called from the no-grad section of the training loop.
    # Enable gradients only for the dummy colours, never for scene parameters.
    with torch.enable_grad():
        dummy_colours = torch.zeros((means3D.shape[0], 3), dtype=means3D.dtype,
                                    device=device, requires_grad=True)
        rendered, _ = rasterizer(
            means3D=means3D, means2D=torch.zeros_like(means3D),
            shs=None, colors_precomp=dummy_colours, opacities=opacities,
            scales=scales, rotations=rotations, cov3D_precomp=cov3D_precomp)
        objective = (rendered * maps).sum()
        statistics = torch.autograd.grad(objective, dummy_colours, create_graph=False,
                                         retain_graph=False, only_inputs=True)[0]
    return statistics.detach()
