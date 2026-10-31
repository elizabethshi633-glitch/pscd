#
# Copyright (C) 2026
# PSCD-GS extension for the original 3D Gaussian Splatting implementation.
#
"""Role-aware perception--structure collaborative densification.

This module implements the density-control part of PSCD-GS while deliberately
leaving the original Gaussian representation, renderer, reconstruction loss and
PLY layout intact.  It is only constructed when ``--pscd`` is passed to
``train.py``.

The stock Graphdeco rasterizer predates an exposed contribution buffer.  Rather
than changing its binary interface, :func:`accumulate_contribution_statistics`
obtains the same contribution-weighted sums from the derivative with respect to
dummy Gaussian colours.  It is exact for the current front-to-back alpha
compositing rule, although it is not as fast as a fused CUDA implementation.
"""

from __future__ import division

import math

import torch
import torch.nn.functional as F

from gaussian_renderer import accumulate_contribution_statistics, render_depth
from utils.general_utils import build_scaling_rotation
from utils.graphics_utils import fov2focal


EPS = 1e-8


def _gradient_2d(image):
    """Return horizontal and vertical Sobel derivatives of ``[1,C,H,W]``."""
    channels = image.shape[1]
    kernel_x = image.new_tensor(((1.0, 0.0, -1.0),
                                 (2.0, 0.0, -2.0),
                                 (1.0, 0.0, -1.0))).view(1, 1, 3, 3) / 8.0
    kernel_y = kernel_x.transpose(-1, -2)
    kernel_x = kernel_x.expand(channels, 1, 3, 3)
    kernel_y = kernel_y.expand(channels, 1, 3, 3)
    return (F.conv2d(image, kernel_x, padding=1, groups=channels),
            F.conv2d(image, kernel_y, padding=1, groups=channels))


def _robust_normalize(values, q_low, q_high, mask=None):
    """Quantile clip and normalize a non-negative image response to [0, 1]."""
    samples = values[mask] if mask is not None else values.reshape(-1)
    if samples.numel() == 0:
        return torch.zeros_like(values), values.new_tensor(0.0), values.new_tensor(1.0)
    lower = torch.quantile(samples, q_low)
    upper = torch.quantile(samples, q_high)
    span = upper - lower
    if not torch.isfinite(span) or span <= EPS:
        return torch.zeros_like(values), lower, upper
    return ((values - lower) / span).clamp(0.0, 1.0), lower, upper


def _normalise_tensor_axis(tensor):
    """Normalize compact 2D symmetric tensors to unit trace.

    The scalar response is accumulated separately.  Unit-trace tensors preserve
    the evidence axis and make their trace a convenient contribution-weighted
    directional support estimate.
    """
    trace = tensor[0] + tensor[2]
    return tensor / trace.clamp_min(EPS).unsqueeze(0)


def _compact_outer_2d(x, y):
    return torch.stack((x * x, x * y, y * y), dim=0)


def _camera_points(camera, xyz):
    ones = torch.ones((xyz.shape[0], 1), dtype=xyz.dtype, device=xyz.device)
    return torch.cat((xyz, ones), dim=1) @ camera.world_view_transform


class PSCDGuidance(object):
    """Perceptual/structural evidence cache and PSCD density-control policy."""

    def __init__(self, args, cameras):
        if not cameras:
            raise ValueError("PSCD-GS requires at least one training camera")
        self.args = args
        self.cameras = cameras
        self.device = cameras[0].original_image.device
        self.fixed_cache = {}
        self.neighbours = self._build_neighbour_graph(cameras)
        self.calibration = {"P": None, "Q": None}
        self._precompute_fixed_evidence()

    # ------------------------------------------------------------------
    # One-time evidence construction
    # ------------------------------------------------------------------
    def _build_neighbour_graph(self, cameras):
        """Use shared COLMAP tracks, with camera-centre distance as fallback."""
        centres = torch.stack([camera.camera_center.detach().cpu() for camera in cameras])
        point_sets = []
        for camera in cameras:
            ids = getattr(camera, "point3D_ids", None)
            if ids is None:
                point_sets.append(None)
            else:
                point_sets.append(set(int(point_id) for point_id in ids if point_id >= 0))

        graph = {}
        neighbour_count = min(int(self.args.pscd_neighbors), max(0, len(cameras) - 1))
        for index in range(len(cameras)):
            scores = []
            for other in range(len(cameras)):
                if other == index:
                    continue
                if point_sets[index] is not None and point_sets[other] is not None:
                    # Larger overlap is preferable.  A distance tie break keeps
                    # this deterministic when two views share no tracks.
                    overlap = len(point_sets[index].intersection(point_sets[other]))
                    distance = float(torch.norm(centres[index] - centres[other]))
                    scores.append((-overlap, distance, other))
                else:
                    distance = float(torch.norm(centres[index] - centres[other]))
                    scores.append((0, distance, other))
            scores.sort()
            graph[index] = [item[-1] for item in scores[:neighbour_count]]
        return graph

    def _vgg_features(self):
        """Load the frozen ImageNet VGG-16 response backbone used by PSCD-GS."""
        try:
            from torchvision.models import VGG16_Weights, vgg16
            # torchvision loads an already cached weight file without network
            # access.  If it is absent, its error below explains how to proceed.
            model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except Exception as error:
            raise RuntimeError(
                "PSCD-GS requires cached ImageNet VGG-16 weights for the fixed "
                "perceptual branch. Install/cache torchvision's VGG16 weights, or "
                "run with --pscd_perceptual_backbone sobel for the offline "
                "engineering fallback. Original error: {}".format(error))
        return model.to(self.device).eval().requires_grad_(False)

    def _perceptual_response_vgg(self, image, backbone):
        """Multi-level LPIPS-style VGG feature variation and axis tensor."""
        target_size = image.shape[-2:]
        # The manuscript uses RGB scaled to [-1, 1].  The channel normalization
        # after each VGG block controls different feature-scale magnitudes.
        activation = image * 2.0 - 1.0
        selected_layers = {3, 8, 15}
        response = None
        axis_tensor = None
        with torch.no_grad():
            for layer_index, layer in enumerate(backbone):
                activation = layer(activation)
                if layer_index not in selected_layers:
                    continue
                activation = activation / torch.sqrt(
                    (activation * activation).mean(dim=1, keepdim=True) + EPS)
                grad_x, grad_y = _gradient_2d(activation)
                magnitude = torch.sqrt((grad_x.square() + grad_y.square()).sum(dim=1) + EPS)[0]
                tensor = _compact_outer_2d(grad_x[0], grad_y[0]).sum(dim=1)
                tensor = _normalise_tensor_axis(tensor)
                magnitude = F.interpolate(magnitude[None, None], size=target_size,
                                          mode="bilinear", align_corners=False)[0, 0]
                tensor = F.interpolate(tensor[None], size=target_size, mode="bilinear",
                                       align_corners=False)[0]
                response = magnitude if response is None else response + magnitude
                axis_tensor = tensor if axis_tensor is None else axis_tensor + tensor
        response = response / 3.0
        axis_tensor = _normalise_tensor_axis(axis_tensor)
        return _robust_normalize(response, self.args.pscd_q_low, self.args.pscd_q_high)[0], axis_tensor

    def _perceptual_response_sobel(self, image):
        """Offline fallback retaining the same multi-scale evidence interface."""
        target_size = image.shape[-2:]
        response = None
        axis_tensor = None
        current = image
        for _ in range(3):
            grad_x, grad_y = _gradient_2d(current)
            magnitude = torch.sqrt((grad_x.square() + grad_y.square()).sum(dim=1) + EPS)[0]
            tensor = _compact_outer_2d(grad_x[0], grad_y[0]).sum(dim=1)
            tensor = _normalise_tensor_axis(tensor)
            magnitude = F.interpolate(magnitude[None, None], size=target_size,
                                      mode="bilinear", align_corners=False)[0, 0]
            tensor = F.interpolate(tensor[None], size=target_size, mode="bilinear",
                                   align_corners=False)[0]
            response = magnitude if response is None else response + magnitude
            axis_tensor = tensor if axis_tensor is None else axis_tensor + tensor
            if min(current.shape[-2:]) < 4:
                break
            current = F.avg_pool2d(current, kernel_size=2, stride=2)
        return _robust_normalize(response, self.args.pscd_q_low, self.args.pscd_q_high)[0], _normalise_tensor_axis(axis_tensor)

    def _precompute_fixed_evidence(self):
        backbone = None
        if self.args.pscd_perceptual_backbone.lower() == "vgg16":
            print("[PSCD] Precomputing frozen VGG-16 perceptual evidence.")
            backbone = self._vgg_features()
        elif self.args.pscd_perceptual_backbone.lower() != "sobel":
            raise ValueError("pscd_perceptual_backbone must be 'vgg16' or 'sobel'")
        else:
            print("[PSCD] Using Sobel perceptual fallback; this is not the paper's VGG-16 branch.")

        for index, camera in enumerate(self.cameras):
            image = camera.original_image[None]
            if backbone is None:
                perceptual, perceptual_tensor = self._perceptual_response_sobel(image)
            else:
                perceptual, perceptual_tensor = self._perceptual_response_vgg(image, backbone)

            luminance = (0.2126 * image[:, 0:1] + 0.7152 * image[:, 1:2] +
                         0.0722 * image[:, 2:3])
            edge_x, edge_y = _gradient_2d(luminance)
            edge_magnitude = torch.sqrt(edge_x.square() + edge_y.square())[0, 0]
            edge, edge_low, edge_high = _robust_normalize(
                edge_magnitude, self.args.pscd_q_low, self.args.pscd_q_high)
            edge_gradient = torch.stack((edge_x[0, 0], edge_y[0, 0]), dim=0)
            if edge_high - edge_low > EPS:
                edge_gradient = edge_gradient / (edge_high - edge_low)
            else:
                edge_gradient.zero_()

            # Evidence is visited one camera at a time during every event, so
            # keep the immutable cache in CPU half precision rather than hold a
            # scene-wide image pyramid in VRAM.
            self.fixed_cache[index] = {
                "A": perceptual.detach().cpu().half(),
                "RA": perceptual_tensor.detach().cpu().half(),
                "E": edge.detach().cpu().half(),
                "gE": edge_gradient.detach().cpu().half(),
            }
        del backbone
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Dynamic structure branch
    # ------------------------------------------------------------------
    def _reliability_map(self, index, depth, structure, depths):
        """Cross-view reprojection reliability from Eq. (3)."""
        camera = self.cameras[index]
        height, width = depth.shape
        yy, xx = torch.meshgrid(torch.arange(height, device=self.device, dtype=depth.dtype),
                                torch.arange(width, device=self.device, dtype=depth.dtype))
        fx = fov2focal(camera.FoVx, width)
        fy = fov2focal(camera.FoVy, height)
        x_camera = (xx - width * 0.5) * depth / fx
        y_camera = (yy - height * 0.5) * depth / fy
        points = torch.stack((x_camera, y_camera, depth, torch.ones_like(depth)), dim=-1).view(-1, 4)
        world_points = points @ torch.inverse(camera.world_view_transform)

        accumulated = torch.zeros_like(depth)
        count = torch.zeros_like(depth)
        for neighbour_index in self.neighbours[index]:
            neighbour = self.cameras[neighbour_index]
            neighbour_depth = depths[neighbour_index].to(self.device, dtype=depth.dtype)
            neighbour_structure = structure[neighbour_index]["H"].to(self.device, dtype=depth.dtype)
            projected = world_points @ neighbour.world_view_transform
            z_projected = projected[:, 2].view(height, width)
            neighbour_width = neighbour_depth.shape[1]
            neighbour_height = neighbour_depth.shape[0]
            u = (fov2focal(neighbour.FoVx, neighbour_width) * projected[:, 0] /
                 projected[:, 2].clamp_min(EPS) + neighbour_width * 0.5).view(height, width)
            v = (fov2focal(neighbour.FoVy, neighbour_height) * projected[:, 1] /
                 projected[:, 2].clamp_min(EPS) + neighbour_height * 0.5).view(height, width)
            grid = torch.stack((2.0 * u / max(neighbour_width - 1, 1) - 1.0,
                                2.0 * v / max(neighbour_height - 1, 1) - 1.0), dim=-1)[None]
            sampled_depth = F.grid_sample(neighbour_depth[None, None], grid, mode="bilinear",
                                          padding_mode="zeros", align_corners=True)[0, 0]
            pooled_structure = F.avg_pool2d(neighbour_structure[None, None], 3, stride=1, padding=1)
            sampled_structure = F.grid_sample(pooled_structure, grid, mode="bilinear",
                                              padding_mode="zeros", align_corners=True)[0, 0]
            valid = ((depth > EPS) & (z_projected > EPS) & (sampled_depth > EPS) &
                     (u >= 0) & (u < neighbour_width) & (v >= 0) & (v < neighbour_height))
            relative_depth = (sampled_depth - z_projected).abs() / torch.maximum(
                torch.maximum(sampled_depth, z_projected), torch.full_like(depth, EPS))
            valid = valid & (relative_depth < self.args.pscd_depth_threshold)
            agreement = torch.exp(-relative_depth / self.args.pscd_depth_sigma)
            agreement = agreement * torch.exp(-(structure[index]["H"].to(self.device, dtype=depth.dtype) -
                                                 sampled_structure).abs() / self.args.pscd_structure_sigma)
            accumulated += torch.where(valid, agreement, torch.zeros_like(agreement))
            count += valid.to(depth.dtype)
        return torch.where(count > 0, accumulated / count.clamp_min(1.0), torch.zeros_like(depth))

    def _build_structure_cache(self, gaussians, pipe):
        """Refresh depth, structural response and cross-view reliability."""
        print("[PSCD] Refreshing depth and multi-view structural evidence.")
        depths = {}
        for index, camera in enumerate(self.cameras):
            depths[index] = render_depth(camera, gaussians, pipe)["depth"].detach().cpu().half()

        # H is constructed before M and is the single-view response used in the
        # reliability comparison.  Storing it on CPU keeps multi-view scenes
        # within the original 3DGS VRAM budget.
        structure = {}
        for index, camera in enumerate(self.cameras):
            fixed = self.fixed_cache[index]
            depth = depths[index].to(self.device, dtype=gaussians.get_xyz.dtype)
            log_depth = torch.log(depth.clamp_min(EPS))[None, None]
            depth_x, depth_y = _gradient_2d(log_depth)
            depth_magnitude = torch.sqrt(depth_x.square() + depth_y.square())[0, 0]
            valid = depth > EPS
            depth_response, depth_low, depth_high = _robust_normalize(
                depth_magnitude, self.args.pscd_q_low, self.args.pscd_q_high, valid)
            depth_gradient = torch.stack((depth_x[0, 0], depth_y[0, 0]), dim=0)
            if depth_high - depth_low > EPS:
                depth_gradient = depth_gradient / (depth_high - depth_low)
            else:
                depth_gradient.zero_()
            edge = fixed["E"].to(self.device, dtype=depth.dtype)
            single_view = self.args.pscd_edge_weight * edge + self.args.pscd_depth_weight * depth_response
            structure[index] = {"H": single_view.detach().cpu().half(),
                                "gD": depth_gradient.detach().cpu().half()}

        event_cache = {}
        for index in range(len(self.cameras)):
            fixed = self.fixed_cache[index]
            depth = depths[index].to(self.device, dtype=gaussians.get_xyz.dtype)
            reliability = self._reliability_map(index, depth, structure, depths)
            trust = self.args.pscd_min_reliability + (1.0 - self.args.pscd_min_reliability) * reliability
            single_view = structure[index]["H"].to(self.device, dtype=depth.dtype)
            structural, _, _ = _robust_normalize(
                single_view * trust, self.args.pscd_q_low, self.args.pscd_q_high)
            edge_gradient = fixed["gE"].to(self.device, dtype=depth.dtype)
            depth_gradient = structure[index]["gD"].to(self.device, dtype=depth.dtype)
            structural_tensor = trust.unsqueeze(0) * (
                self.args.pscd_edge_weight * _compact_outer_2d(edge_gradient[0], edge_gradient[1]) +
                self.args.pscd_depth_weight * _compact_outer_2d(depth_gradient[0], depth_gradient[1]))
            structural_tensor = _normalise_tensor_axis(structural_tensor)
            event_cache[index] = {
                "B": structural.detach().cpu().half(),
                "RB": structural_tensor.detach().cpu().half(),
                "M": reliability.detach().cpu().half(),
            }
        return event_cache

    # ------------------------------------------------------------------
    # Contribution aggregation, calibration and role construction
    # ------------------------------------------------------------------
    def _update_calibration(self, name, raw):
        low = torch.quantile(raw, self.args.pscd_q_low)
        high = torch.quantile(raw, self.args.pscd_q_high)
        old = self.calibration[name]
        if old is not None:
            old_low, old_high = old
            if high - low <= EPS:
                low, high = old_low, old_high
            else:
                beta = self.args.pscd_calibration_ema
                low = beta * old_low + (1.0 - beta) * low
                high = beta * old_high + (1.0 - beta) * high
        if high - low <= EPS:
            self.calibration[name] = (low.detach(), high.detach())
            return torch.zeros_like(raw)
        self.calibration[name] = (low.detach(), high.detach())
        return ((raw - low) / (high - low + 1e-6)).clamp(0.0, 1.0)

    def _aggregate_scores(self, gaussians, pipe, event_cache):
        count = gaussians.get_xyz.shape[0]
        dtype = gaussians.get_xyz.dtype
        support = torch.zeros(count, device=self.device, dtype=dtype)
        sum_p = torch.zeros_like(support)
        sum_p2 = torch.zeros_like(support)
        sum_q = torch.zeros_like(support)
        sum_q2 = torch.zeros_like(support)
        for index, camera in enumerate(self.cameras):
            fixed = self.fixed_cache[index]
            dynamic = event_cache[index]
            perceptual = fixed["A"].to(self.device, dtype=dtype)
            structural = dynamic["B"].to(self.device, dtype=dtype)
            fields = torch.stack((torch.ones_like(perceptual), perceptual, structural), dim=0)
            statistics = accumulate_contribution_statistics(camera, gaussians, pipe, fields).clamp_min(0.0)
            contribution = statistics[:, 0]
            visibility = (contribution / self.args.pscd_contribution_threshold).clamp(0.0, 1.0)
            p_value = statistics[:, 1] / contribution.clamp_min(EPS)
            q_value = statistics[:, 2] / contribution.clamp_min(EPS)
            support += visibility
            sum_p += visibility * p_value
            sum_p2 += visibility * p_value.square()
            sum_q += visibility * q_value
            sum_q2 += visibility * q_value.square()

        denominator = support.clamp_min(EPS)
        mean_p = sum_p / denominator
        mean_q = sum_q / denominator
        variance_p = (sum_p2 / denominator - mean_p.square()).clamp_min(0.0)
        variance_q = (sum_q2 / denominator - mean_q.square()).clamp_min(0.0)
        support_weight = (support / self.args.pscd_visibility_views).clamp(0.0, 1.0)
        raw_p = support_weight * torch.exp(-variance_p / (self.args.pscd_evidence_sigma ** 2 + EPS)) * mean_p
        raw_q = support_weight * torch.exp(-variance_q / (self.args.pscd_evidence_sigma ** 2 + EPS)) * mean_q
        return self._update_calibration("P", raw_p), self._update_calibration("Q", raw_q)

    def _roles_and_candidates(self, gaussians, perception, structure, grad_threshold, extent):
        grad = gaussians.xyz_gradient_accum.squeeze(-1) / gaussians.denom.squeeze(-1).clamp_min(1.0)
        grad = torch.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0)
        gradient_score = grad / (grad + grad_threshold + EPS)
        slope = self.args.pscd_role_slope
        threshold = self.args.pscd_role_threshold
        member_p = torch.sigmoid(slope * (perception - threshold))
        member_q = torch.sigmoid(slope * (structure - threshold))
        co_dominant = member_p * member_q
        p_dominant = member_p * (1.0 - member_q)
        q_dominant = (1.0 - member_p) * member_q
        evidence = (p_dominant * perception + q_dominant * structure +
                    co_dominant * 0.5 * (perception + structure))
        priority = gradient_score + self.args.pscd_evidence_bonus * (1.0 - gradient_score) * evidence

        gaussian_count = gaussians.get_xyz.shape[0]
        budget = max(0, min(int(self.args.pscd_max_candidates),
                            int(self.args.pscd_growth_rate * gaussian_count),
                            int(self.args.pscd_max_gaussians) - gaussian_count))
        if budget == 0:
            return None, None, None, None, None
        scheduled_for_pruning = gaussians.get_opacity.squeeze(-1) < 0.005
        eligible = (priority >= self.args.pscd_priority_threshold) & ~scheduled_for_pruning
        eligible_count = int(eligible.sum().item())
        if eligible_count == 0:
            return None, None, None, None, None
        candidate_count = min(budget, eligible_count)
        candidate_priority = priority.masked_fill(~eligible, -float("inf"))
        indices = torch.topk(candidate_priority, k=candidate_count, sorted=False).indices
        structural_share = structure[indices] / (perception[indices] + structure[indices] + EPS)
        split_tendency = (q_dominant[indices] + co_dominant[indices] *
                          (self.args.pscd_codominant_split +
                           (1.0 - self.args.pscd_codominant_split) * structural_share))
        scale_safeguard = gaussians.get_scaling[indices].amax(dim=1) > gaussians.percent_dense * extent
        split_mask = scale_safeguard | (split_tendency >= self.args.pscd_split_threshold)
        return indices, split_mask, member_p, member_q, priority

    # ------------------------------------------------------------------
    # Directional lifting for selected split candidates
    # ------------------------------------------------------------------
    def _lift_axis(self, camera, gaussians, indices, compact_tensor, contribution,
                   reliability_sum, is_structural):
        """Lift a compact image-axis tensor into covariance-constrained 3D axes."""
        values = compact_tensor[indices].clamp_min(0.0)
        support = (values[:, 0] + values[:, 2]).clamp_min(EPS)
        matrices = torch.stack((torch.stack((values[:, 0], values[:, 1]), dim=-1),
                                torch.stack((values[:, 1], values[:, 2]), dim=-1)), dim=-2) / support[:, None, None]
        eigenvalues, eigenvectors = torch.linalg.eigh(matrices)
        image_axis = eigenvectors[:, :, 1]
        anisotropy = (eigenvalues[:, 1] - eigenvalues[:, 0]) / eigenvalues[:, 1].clamp_min(EPS)

        xyz = gaussians.get_xyz[indices]
        camera_xyz = _camera_points(camera, xyz)[:, :3]
        width = camera.image_width
        height = camera.image_height
        fx = fov2focal(camera.FoVx, width)
        fy = fov2focal(camera.FoVy, height)
        z = camera_xyz[:, 2].clamp_min(EPS)
        jacobian = torch.zeros((indices.numel(), 2, 3), device=self.device, dtype=xyz.dtype)
        jacobian[:, 0, 0] = fx / z
        jacobian[:, 0, 2] = -fx * camera_xyz[:, 0] / z.square()
        jacobian[:, 1, 1] = fy / z
        jacobian[:, 1, 2] = -fy * camera_xyz[:, 1] / z.square()
        factors = build_scaling_rotation(gaussians.get_scaling[indices], gaussians.get_rotation[indices])
        projected = torch.bmm(jacobian, factors)
        normal_matrix = torch.bmm(projected.transpose(1, 2), projected)
        normal_matrix = normal_matrix + self.args.pscd_jacobian_ridge * torch.eye(
            3, device=self.device, dtype=xyz.dtype)[None]
        lifted = torch.linalg.solve(normal_matrix,
                                    torch.bmm(projected.transpose(1, 2), image_axis.unsqueeze(-1))).squeeze(-1)
        direction = F.normalize(torch.bmm(factors, lifted.unsqueeze(-1)).squeeze(-1), dim=1, eps=EPS)
        fitted = torch.bmm(projected, lifted.unsqueeze(-1)).squeeze(-1)
        fit = torch.exp(-(fitted - image_axis).square().sum(dim=1) /
                        (self.args.pscd_fit_sigma ** 2 + EPS))
        visibility = (contribution[indices] / self.args.pscd_contribution_threshold).clamp(0.0, 1.0)
        response_support = (support / self.args.pscd_direction_support).clamp(0.0, 1.0)
        if is_structural:
            reliability = (reliability_sum[indices] / contribution[indices].clamp_min(EPS)).clamp(0.0, 1.0)
        else:
            reliability = torch.ones_like(visibility)
        confidence = anisotropy.clamp_min(0.0) * visibility * response_support * fit * reliability
        return direction, confidence

    def _estimate_split_directions(self, gaussians, pipe, event_cache, indices, member_p, member_q):
        selected_count = indices.numel()
        dtype = gaussians.get_xyz.dtype
        fused_p = torch.zeros((selected_count, 3, 3), device=self.device, dtype=dtype)
        fused_q = torch.zeros_like(fused_p)
        support_p = torch.zeros(selected_count, device=self.device, dtype=dtype)
        support_q = torch.zeros_like(support_p)
        for index, camera in enumerate(self.cameras):
            fixed = self.fixed_cache[index]
            dynamic = event_cache[index]
            perceptual = fixed["A"].to(self.device, dtype=dtype)
            p_tensor = fixed["RA"].to(self.device, dtype=dtype)
            structural = dynamic["B"].to(self.device, dtype=dtype)
            q_tensor = dynamic["RB"].to(self.device, dtype=dtype)
            reliability = dynamic["M"].to(self.device, dtype=dtype)

            common = accumulate_contribution_statistics(
                camera, gaussians, pipe,
                torch.stack((torch.ones_like(perceptual), reliability, torch.zeros_like(perceptual)), dim=0)).clamp_min(0.0)
            p_statistics = accumulate_contribution_statistics(camera, gaussians, pipe,
                perceptual.unsqueeze(0) * p_tensor).clamp_min(0.0)
            q_statistics = accumulate_contribution_statistics(camera, gaussians, pipe,
                structural.unsqueeze(0) * q_tensor).clamp_min(0.0)
            p_direction, p_confidence = self._lift_axis(
                camera, gaussians, indices, p_statistics, common[:, 0], common[:, 1], False)
            q_direction, q_confidence = self._lift_axis(
                camera, gaussians, indices, q_statistics, common[:, 0], common[:, 1], True)
            fused_p += p_confidence[:, None, None] * torch.bmm(p_direction.unsqueeze(-1), p_direction.unsqueeze(1))
            fused_q += q_confidence[:, None, None] * torch.bmm(q_direction.unsqueeze(-1), q_direction.unsqueeze(1))
            support_p += p_confidence
            support_q += q_confidence

        def finalise_branch(tensor, support):
            matrix = tensor / support.clamp_min(EPS)[:, None, None]
            values, vectors = torch.linalg.eigh(matrix)
            direction = vectors[:, :, 2]
            confidence = ((values[:, 2] - values[:, 1]) / values[:, 2].clamp_min(EPS) *
                          (support / self.args.pscd_direction_views).clamp(0.0, 1.0))
            return direction, confidence.clamp_min(0.0)

        p_direction, p_confidence = finalise_branch(fused_p, support_p)
        q_direction, q_confidence = finalise_branch(fused_q, support_q)
        p_confident = p_confidence * (p_confidence >= self.args.pscd_direction_threshold).to(dtype)
        q_confident = q_confidence * (q_confidence >= self.args.pscd_direction_threshold).to(dtype)
        role_tensor = (member_p[indices, None, None] * p_confident[:, None, None] *
                       torch.bmm(p_direction.unsqueeze(-1), p_direction.unsqueeze(1)) +
                       member_q[indices, None, None] * q_confident[:, None, None] *
                       torch.bmm(q_direction.unsqueeze(-1), q_direction.unsqueeze(1)))
        values, vectors = torch.linalg.eigh(role_tensor)
        direction = vectors[:, :, 2]
        confidence = ((values[:, 2] - values[:, 1]) / values[:, 2].clamp_min(EPS) *
                      (member_p[indices] * p_confident + member_q[indices] * q_confident).clamp(0.0, 1.0))
        valid = confidence >= self.args.pscd_direction_threshold
        return direction, valid

    # ------------------------------------------------------------------
    # Public training-loop entry point and checkpoint state
    # ------------------------------------------------------------------
    @torch.no_grad()
    def densify_and_prune(self, gaussians, pipe, grad_threshold, extent, max_screen_size):
        event_cache = self._build_structure_cache(gaussians, pipe)
        try:
            perception, structure = self._aggregate_scores(gaussians, pipe, event_cache)
            indices, split_mask, member_p, member_q, priority = self._roles_and_candidates(
                gaussians, perception, structure, grad_threshold, extent)
            if indices is None:
                gaussians.densify_and_prune_role_aware(
                    torch.empty(0, dtype=torch.long, device=self.device),
                    torch.empty(0, dtype=torch.long, device=self.device), None, None,
                    0.005, extent, max_screen_size, self.args.pscd_split_displacement,
                    self.args.pscd_covariance_anisotropy)
                print("[PSCD] No primitive passed the role-aware event budget.")
                return

            clone_indices = indices[~split_mask]
            split_indices = indices[split_mask]
            direction = None
            direction_valid = None
            if split_indices.numel() > 0:
                direction, direction_valid = self._estimate_split_directions(
                    gaussians, pipe, event_cache, split_indices, member_p, member_q)
            gaussians.densify_and_prune_role_aware(
                clone_indices, split_indices, direction, direction_valid, 0.005, extent,
                max_screen_size, self.args.pscd_split_displacement,
                self.args.pscd_covariance_anisotropy)
            print("[PSCD] candidates={} clone={} split={} mean(P)={:.3f} mean(Q)={:.3f}".format(
                indices.numel(), clone_indices.numel(), split_indices.numel(),
                perception.mean().item(), structure.mean().item()))
        finally:
            # Structural maps are only valid for this synchronized Gaussian
            # snapshot. Release the CPU event cache before optimization resumes.
            event_cache.clear()

    def state_dict(self):
        state = {}
        for name, value in self.calibration.items():
            state[name] = None if value is None else (float(value[0]), float(value[1]))
        return state

    def load_state_dict(self, state):
        if not state:
            return
        for name in self.calibration:
            value = state.get(name)
            if value is not None:
                self.calibration[name] = (torch.tensor(value[0], device=self.device),
                                          torch.tensor(value[1], device=self.device))
