import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import TINY_NUMBER, HUGE_NUMBER
from collections import OrderedDict
import mcubes
from neus_network import SphericalHarmonicsEmbedder, DummyEmbedder, Embedder, MLPNet, SDFNetwork, RenderingNetwork, SingleVarianceNetwork
import os
import logging

logger = logging.getLogger(__package__)


######################################################################################
# wrapper to simplify the use of NeusRenderer
######################################################################################

# def extract_fields(bound_min, bound_max, resolution, query_func):
#     N = 64
#     X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
#     Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
#     Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

#     u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
#     with torch.no_grad():
#         for xi, xs in enumerate(X):
#             for yi, ys in enumerate(Y):
#                 for zi, zs in enumerate(Z):
#                     xx, yy, zz = torch.meshgrid(xs, ys, zs)
#                     pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1)
#                     val = query_func(pts).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
#                     u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = val
#     return u


# def extract_geometry(bound_min, bound_max, resolution, threshold, query_func):
#     print('threshold: {}'.format(threshold))
#     u = extract_fields(bound_min, bound_max, resolution, query_func)
#     vertices, triangles = mcubes.marching_cubes(u, threshold)
#     b_max_np = bound_max.detach().cpu().numpy()
#     b_min_np = bound_min.detach().cpu().numpy()

#     vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
#     return vertices, triangles


def sample_pdf(bins, weights, n_samples, det=False):
    # This implementation is from NeRF
    # Get pdf
    weights = weights + 1e-5  # prevent nans
    pdf = weights / torch.sum(weights, -1, keepdim=True)
    cdf = torch.cumsum(pdf, -1)
    cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], -1)
    # Take uniform samples
    if det:
        u = torch.linspace(0. + 0.5 / n_samples, 1. - 0.5 / n_samples, steps=n_samples)
        u = u.expand(list(cdf.shape[:-1]) + [n_samples])
    else:
        u = torch.rand(list(cdf.shape[:-1]) + [n_samples])

    # Invert CDF
    u = u.contiguous().cuda()
    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.max(torch.zeros_like(inds - 1), inds - 1)
    above = torch.min((cdf.shape[-1] - 1) * torch.ones_like(inds), inds)
    inds_g = torch.stack([below, above], -1)  # (batch, N_samples, 2)

    matched_shape = [inds_g.shape[0], inds_g.shape[1], cdf.shape[-1]]
    cdf_g = torch.gather(cdf.unsqueeze(1).expand(matched_shape), 2, inds_g)
    bins_g = torch.gather(bins.unsqueeze(1).expand(matched_shape), 2, inds_g)

    denom = (cdf_g[..., 1] - cdf_g[..., 0])
    denom = torch.where(denom < 1e-5, torch.ones_like(denom), denom)
    t = (u - cdf_g[..., 0]) / denom
    samples = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])

    return samples

class NeusRenderer(nn.Module):
    def __init__(self, args):
        super().__init__()
        # foreground
        if args.use_pe:
            self.fg_embedder_position = Embedder(input_dim=3,
                                                max_freq_log2=args.max_freq_log2 - 1,
                                                N_freqs=args.max_freq_log2,
                                                N_anneal=args.N_anneal,
                                                N_anneal_min_freq=args.N_anneal_min_freq,
                                                use_annealing=args.use_annealing)
            
            self.fg_embedder_viewdir = SphericalHarmonicsEmbedder(degree=4)  # Adjust degree as needed
            
            # self.fg_embedder_viewdir = Embedder(input_dim=3,
            #                                     max_freq_log2=args.max_freq_log2_viewdirs - 1,
            #                                     N_freqs=args.max_freq_log2_viewdirs,
            #                                     # include_input=True,
            #                                     N_anneal=args.N_anneal,
            #                                     N_anneal_min_freq=args.N_anneal_min_freq_viewdirs,
            #                                     use_annealing=args.use_annealing)
            # self.fg_embedder_viewdir_color_net = Embedder(input_dim=9,
            #                                     max_freq_log2=args.max_freq_log2_viewdirs - 1,
            #                                     N_freqs=args.max_freq_log2_viewdirs,
            #                                     include_input=False,
            #                                     N_anneal=args.N_anneal,
            #                                     N_anneal_min_freq=args.N_anneal_min_freq_viewdirs,
            #                                     use_annealing=args.use_annealing)
            
        else:
            self.fg_embedder_position = DummyEmbedder(input_dim=3)
            self.fg_embedder_viewdir = DummyEmbedder(input_dim=3)
            # self.fg_embedder_viewdir_color_net = DummyEmbedder(input_dim=9)

        # Print dimensions for debugging
        # print(f"\nNetwork dimensions:")
        # print(f"View direction embedding output dim: {self.fg_embedder_viewdir.out_dim}")
        # print(f"Position embedding output dim: {self.fg_embedder_position.out_dim}")

        self.nerf = MLPNet(D=args.netdepth,
                                    W=args.netwidth,
                                    input_ch=self.fg_embedder_position.out_dim,
                                    input_ch_viewdirs=self.fg_embedder_viewdir.out_dim,
                                    use_viewdirs=args.color_mode != "no_view_dir",
                                    act=args.activation,
                                    garf_sigma=args.garf_sigma,
                                    crop_y=(args.crop_y_min, args.crop_y_max),
                                    crop_r=args.crop_r,
                                    init_gain=args.init_gain)

        self.color_network = RenderingNetwork(d_feature_color=args.color_d_feature,
                                                mode_color=args.color_mode,
                                                d_out_color=args.color_output_dim,
                                                d_hidden_color=args.color_hidden_dim,
                                                n_layers_color=args.color_n_layers,
                                                d_in_color=self.fg_embedder_viewdir.out_dim,
                                                weight_norm=args.color_weight_norm,
                                                squeeze_out=args.color_squeeze_out)

        self.sdf_network = SDFNetwork(D=args.netdepth,
                                        W=args.netwidth,
                                        d_out=args.sdf_d_out,
                                        input_ch=self.fg_embedder_position.out_dim,
                                        bias=args.sdf_bias,
                                        scale=args.sdf_scale,
                                        geometric_init=args.sdf_geometric_init,
                                        weight_norm=args.sdf_weight_norm,
                                        inside_outside=args.sdf_inside_outside)

        self.deviation_network = SingleVarianceNetwork(init_val=args.dev_init_val)

        # # background; bg_pt is (x, y, z, 1/r)
        # self.bg_embedder_position = Embedder(input_dim=4,
        #                                      max_freq_log2=args.max_freq_log2 - 1,
        #                                      N_freqs=args.max_freq_log2,
        #                                      N_anneal=args.N_anneal,
        #                                      N_anneal_min_freq=args.N_anneal_min_freq,
        #                                      use_annealing=args.use_annealing)
        # self.bg_embedder_viewdir = Embedder(input_dim=3,
        #                                     max_freq_log2=args.max_freq_log2_viewdirs - 1,
        #                                     N_freqs=args.max_freq_log2_viewdirs,
        #                                     N_anneal=args.N_anneal,
        #                                     N_anneal_min_freq=args.N_anneal_min_freq_viewdirs,
        #                                     use_annealing=args.use_annealing)
        # self.bg_net = MLPNet(D=args.netdepth, W=args.netwidth,
        #                      input_ch=self.bg_embedder_position.out_dim,
        #                      input_ch_viewdirs=self.bg_embedder_viewdir.out_dim,
        #                      use_viewdirs=args.use_viewdirs,
        #                      act=args.activation)

        self.with_bg = args.with_bg
        self.with_ldist = args.use_reg

        self.bg_color = args.bg_color
        self.N_anneal = args.N_anneal
        self.n_samples = args.n_samples
        self.n_importance = args.n_importance
        self.n_outside = args.n_outside
        self.up_sample_steps = args.up_sample_steps
        self.perturb = args.perturb

        self.crop_r = args.crop_r
        self.crop_y = (args.crop_y_min, args.crop_y_max)

        # self.optimizer  = torch.optim.Adam(net.parameters(), lr=args.lrate)

    # def render_core_outside(self, rays_o, rays_d, z_vals, sample_dist, iteration, background_rgb=None):
    #     """
    #     Render background  ////////////// NOT USED NOW   /////////////////
    #     """
    #     batch_size, n_samples = z_vals.shape

    #     # Section length

    #     """ Calculate the distances between consecutive depth values and append the constant sample_dist for the last segment. """
    #     dists = z_vals[..., 1:] - z_vals[..., :-1]

    #     dists = torch.cat([dists, torch.Tensor([sample_dist]).expand(dists[..., :1].shape).cuda()], -1)

    #     """ Find the 3D points in space corresponding to the middle of each depth segment. """
    #     mid_z_vals = z_vals + dists * 0.5
    #     # Section midpoints
    #     pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # batch_size, n_samples, 3

    #     """ For points inside a sphere (possibly of radius 1.0), normalize their positions relative to the sphere's center.
    #     This is a common technique in NeRF to ensure that points are represented in a normalized space, especially when
    #     dealing with spherical datasets or scenes. """
    #     dis_to_center = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).clip(1.0, 1e10)
    #     pts = torch.cat([pts / dis_to_center, 1.0 / dis_to_center], dim=-1)       # batch_size, n_samples, 4

    #     dirs = rays_d[:, None, :].expand(batch_size, n_samples, 3)

    #     pts = pts.reshape(-1, 3 + int(self.n_outside > 0))
    #     dirs = dirs.reshape(-1, 3)

    #     """ For each midpoint, query the NeRF model to obtain density and color values. """
    #     density, sampled_color = self.nerf(pts, dirs, iteration,
    #                                        embedder_position=self.fg_embedder_position,
    #                                        embedder_viewdir=self.fg_embedder_viewdir)
    #     sampled_color = torch.sigmoid(sampled_color)

    #     """ Convert the densities into alpha (transparency) values using the equation of volumetric rendering.
    #     This essentially calculates how much light is absorbed or scattered at each sampled depth. """
    #     alpha = 1.0 - torch.exp(-F.softplus(density.reshape(batch_size, n_samples)) * dists)
    #     alpha = alpha.reshape(batch_size, n_samples)

    #     """ These weights are used for alpha compositing - combining colors from all sampled points along the ray.
    #     The equation ensures that areas with high density contribute more to the final color. """
    #     ones_tensor = torch.ones([batch_size, 1]).cuda()
    #     weights = alpha * torch.cumprod(torch.cat([ones_tensor, 1. - alpha + 1e-7], -1), -1)[:, :-1]
    #     sampled_color = sampled_color.reshape(batch_size, n_samples, 3)

    #     """ Using the computed weights, accumulate (integrate) the colors along each ray. """
    #     color = (weights[:, :, None] * sampled_color).sum(dim=1)

    #     """ If a background color is provided, blend it with the accumulated color based on the transparency of the scene. """
    #     if background_rgb is not None:
    #         color = color + background_rgb * (1.0 - weights.sum(dim=-1, keepdim=True))

    #     return {
    #         'color': color,
    #         'sampled_color': sampled_color,
    #         'alpha': alpha,
    #         'weights': weights,
    #     }

    def up_sample(self, rays_o, rays_d, z_vals, sdf, n_importance, inv_s):
        """
        Up sampling give a fixed inv_s
        """
        batch_size, n_samples = z_vals.shape

        """ Computes the 3D points in space where the rays intersect the volume at the sampled depths. """
        pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]  # n_rays, n_samples, 3

        """ Determine which of the computed points lie inside a unit sphere. """
        radius = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=False)
        inside_sphere = (radius[:, :-1] < 1.0) | (radius[:, 1:] < 1.0)

        sdf = sdf.reshape(batch_size, n_samples)
        prev_sdf, next_sdf = sdf[:, :-1], sdf[:, 1:]
        prev_z_vals, next_z_vals = z_vals[:, :-1], z_vals[:, 1:]
        mid_sdf = (prev_sdf + next_sdf) * 0.5

        """ Compute the gradient of the sdf with respect to depth (approximated by the
        difference between adjacent sdf values) to get an idea about the shape of the scene.
        The cosine values represent how rapidly the sdf changes. """
        cos_val = (next_sdf - prev_sdf) / (next_z_vals - prev_z_vals + 1e-5)
        cos_val = cos_val.cuda()

        # ----------------------------------------------------------------------------------------------------------
        # Use min value of [ cos, prev_cos ]
        # Though it makes the sampling (not rendering) a little bit biased, this strategy can make the sampling more
        # robust when meeting situations like below:
        #
        # SDF
        # ^
        # |\          -----x----...
        # | \        /
        # |  x      x
        # |---\----/-------------> 0 level
        # |    \  /
        # |     \/
        # |
        # ----------------------------------------------------------------------------------------------------------

        """ A strategy is employed to make the sampling more robust by considering the
        minimum of the current and previous gradient estimates. """

        zero_tensor = torch.zeros([batch_size, 1]).cuda()
        prev_cos_val = torch.cat([zero_tensor, cos_val[:, :-1]], dim=-1)
        cos_val = torch.stack([prev_cos_val, cos_val], dim=-1)
        cos_val, _ = torch.min(cos_val, dim=-1, keepdim=False)
        cos_val = cos_val.clip(-1e3, 0.0) * inside_sphere

        """ Using the estimated gradient (cosine values), the function computes an
        estimated sdf for previous and next depth values. These estimated sdf values are used
        to compute the CDF for each interval between depth samples. The CDF will later be used
        to determine the importance of each interval when sampling new depth values. """
        dist = (next_z_vals - prev_z_vals)
        prev_esti_sdf = mid_sdf - cos_val * dist * 0.5
        next_esti_sdf = mid_sdf + cos_val * dist * 0.5
        prev_cdf = torch.sigmoid(prev_esti_sdf * inv_s)
        next_cdf = torch.sigmoid(next_esti_sdf * inv_s)

        """ The difference between the CDF values at the start and end of each interval is used
        to compute a weight for each interval. These weights will guide where new samples should be placed. """
        alpha = (prev_cdf - next_cdf + 1e-5) / (prev_cdf + 1e-5)
        weights = alpha * torch.cumprod(
            torch.cat([zero_tensor, 1. - alpha + 1e-7], -1), -1)[:, :-1]

        """ Based on the computed weights, new depth values are sampled using importance sampling. This ensures
        that areas with more geometric detail (as inferred from the sdf) get more samples. """
        z_samples = sample_pdf(z_vals, weights, n_importance, det=True).detach()

        """ The function returns the newly sampled depth values (z_samples). """
        return z_samples

    def cat_z_vals(self, rays_o, rays_d, z_vals, new_z_vals, sdf, iteration, last=False):
        batch_size, n_samples = z_vals.shape
        _, n_importance = new_z_vals.shape

        """ Calculate the 3D points in space where the rays intersect the volume at the new sampled depths. """
        pts = rays_o[:, None, :] + rays_d[:, None, :] * new_z_vals[..., :, None]

        """ Combine the old and new depth values, then sort them. """
        z_vals = torch.cat([z_vals, new_z_vals], dim=-1)
        z_vals, index = torch.sort(z_vals, dim=-1)

        """ If the last flag is not set, the function computes the SDF values for the newly sampled depths and
        concatenates them with the existing SDF values. The SDF values are then rearranged based on the sorting
        index to match the order of the sorted z_vals. """
        if not last:
            new_sdf = self.sdf_network.sdf(pts.reshape(-1, 3), iteration, embedder_position=self.fg_embedder_position).reshape(batch_size, n_importance)
            sdf = torch.cat([sdf, new_sdf], dim=-1)
            xx = torch.arange(batch_size)[:, None].expand(batch_size, n_samples + n_importance).reshape(-1)
            index = index.reshape(-1)
            sdf = sdf[(xx, index)].reshape(batch_size, n_samples + n_importance)

        """ The function returns the sorted and concatenated depth values (z_vals) and their corresponding SDF values (sdf). """
        return z_vals, sdf

    def render_core(self,
                    rays_o,
                    rays_d,
                    z_vals,
                    sample_dist,
                    iteration,
                    background_rgb,
                    background_alpha=None,
                    background_sampled_color=None):
        
        """ Extracts the batch size and number of depth samples from the shape of the z_vals. """
        batch_size, n_samples = z_vals.shape

        # Section length
        """ Computes the lengths of depth sections by subtracting adjacent depth values.
        Appends a constant distance sample_dist for the last segment. """
        dists = z_vals[..., 1:] - z_vals[..., :-1]
        dists = torch.cat([dists, (torch.Tensor([sample_dist]).expand(dists[..., :1].shape).cuda())], -1)

        """ Finds the midpoint depth for each section. """
        mid_z_vals = z_vals + dists * 0.5

        # Section midpoints
        """ Calculates the 3D coordinates for the midpoints of each depth segment. """
        pts = rays_o[:, None, :] + rays_d[:, None, :] * mid_z_vals[..., :, None]  # n_rays, n_samples, 3

        """ Expands the ray directions to match the shape of the pts. """
        dirs = rays_d[:, None, :].expand(pts.shape)

        pts = pts.reshape(-1, 3)
        dirs = dirs.reshape(-1, 3)

        """ Queries the sdf_network to get the Signed Distance Field (SDF) values and
        feature vectors for each point. """
        sdf_nn_output = self.sdf_network(pts, iteration, embedder_position=self.fg_embedder_position)
        sdf = sdf_nn_output[:, :1]
        feature_vector = sdf_nn_output[:, 1:]

        """ Computes the gradient (normal) of the SDF for each point. """
        pts.requires_grad_(True)
        gradients = self.sdf_network.gradient(pts, iteration, embedder_position=self.fg_embedder_position).squeeze()

        """ Queries the color_network to obtain RGB colors based on
        the SDF gradients, view direction, and feature vectors. """
        sampled_color = self.color_network(pts, gradients, dirs, feature_vector, iteration, embedder_viewdir=self.fg_embedder_viewdir).reshape(batch_size, n_samples, 3)


        """ Queries the deviation_network to get an inverse deviation parameter. """
        inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)           # Single parameter
        inv_s = inv_s.expand(batch_size * n_samples, 1)

        """ Calculates the cosine of the angle between the gradient (normal) and the view direction. """
        true_cos = (dirs * gradients).sum(-1, keepdim=True)

        # "cos_anneal_ratio" grows from 0 to 1 in the beginning training iterations. The anneal strategy below makes
        # the cos value "not dead" at the beginning training iterations, for better convergence.
        """ Applies annealing to the cosine value, which is a technique
        to adjust the value over time (or iterations) for better convergence. """

        if self.N_anneal == 0:
            cos_anneal_ratio = 1
        else:
            cos_anneal_ratio = np.min([1.0, iteration / self.N_anneal])

        iter_cos = -(F.relu(-true_cos * 0.5 + 0.5) * (1.0 - cos_anneal_ratio) +
                     F.relu(-true_cos) * cos_anneal_ratio)  # always non-positive

        # Estimate signed distances at section points
        """ Estimates the SDF values at the start and end of each
        section based on the gradient and the section length. """
        estimated_next_sdf = sdf + iter_cos * dists.reshape(-1, 1) * 0.5
        estimated_prev_sdf = sdf - iter_cos * dists.reshape(-1, 1) * 0.5

        """ Converts the estimated SDF values to Cumulative Distribution
        Function (CDF) values using the sigmoid function. """
        prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
        next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)

        p = prev_cdf - next_cdf
        c = prev_cdf

        """ Computes the transparency (alpha) value for each section based on the difference in CDF values. """
        alpha = ((p + 1e-5) / (c + 1e-5)).reshape(batch_size, n_samples).clip(0.0, 1.0)

        """ Checks if the sampled points are inside a predefined sphere (possibly of radius 1.0) """
        pts_norm = torch.linalg.norm(pts, ord=2, dim=-1, keepdim=True).reshape(batch_size, n_samples)
        inside_sphere = (pts_norm < 1.0).float().detach()
        relax_inside_sphere = (pts_norm < 1.2).float().detach()

        # Render with background
        """ This block blends the computed colors with provided background information when it's available. """     
        if background_alpha is not None:
            alpha = alpha * inside_sphere + background_alpha[:, :n_samples] * (1.0 - inside_sphere)
            alpha = torch.cat([alpha, background_alpha[:, n_samples:]], dim=-1)
            sampled_color = sampled_color * inside_sphere[:, :, None] +\
                            background_sampled_color[:, :n_samples] * (1.0 - inside_sphere)[:, :, None]
            sampled_color = torch.cat([sampled_color, background_sampled_color[:, n_samples:]], dim=1)
            # sampled_color = sampled_color * mask.unsqueeze(-1).float()    # Not using background yet


        """ Calculates weights for color accumulation using the alpha values and
        then integrates the colors along each ray based on these weights. """
        weights = alpha * torch.cumprod(torch.cat([torch.ones([batch_size, 1]).cuda(), 1. - alpha + 1e-7], -1), -1)[:, :-1]
        weights_sum = weights.sum(dim=-1, keepdim=True)
        color = (sampled_color * weights[:, :, None]).sum(dim=1)

        if background_rgb is not None:    # Fixed background, usually black
            color = color + background_rgb * (1.0 - weights_sum)

        # Eikonal loss
        """ Computes the gradient error which measures how well the computed gradients align with the expected unit normals. """
        gradient_error = (torch.linalg.norm(gradients.reshape(batch_size, n_samples, 3), ord=2,
                                            dim=-1) - 1.0) ** 2
        gradient_error = (relax_inside_sphere * gradient_error).sum() / (relax_inside_sphere.sum() + 1e-5)

        """ Returns a dictionary containing the final rendered color, SDF values, gradients, and other intermediate values. """
        return {
            'color': color,
            'sdf': sdf,
            'dists': dists,
            'gradients': gradients.reshape(batch_size, n_samples, 3),
            's_val': 1.0 / inv_s,
            'mid_z_vals': mid_z_vals,
            'weights': weights,
            'cdf': c.reshape(batch_size, n_samples),
            'gradient_error': gradient_error,
            'inside_sphere': inside_sphere
        }

    def render(self, rays_o, rays_d, near, far, iteration, background_rgb, perturb_overwrite, cos_anneal_ratio=0.0):

        """ Determines the number of rays in the batch. """
        batch_size = len(rays_o)
        """ Calculates the distance between sampled points, assuming the region of interest is a unit sphere. """
        sample_dist = 2.0 / self.n_samples   # Assuming the region of interest is a unit sphere
        """ Generates equally spaced depth samples between near and far boundaries. """
        z_vals = torch.linspace(0.0, 1.0, self.n_samples)
        z_vals = z_vals.cuda()
        z_vals = near + (far - near) * z_vals[None, :]

        """ If the renderer samples points outside the unit sphere (as indicated by self.n_outside > 0),
        this block generates additional depth values outside the unit sphere. """
        z_vals_outside = None
        if self.n_outside > 0:
            z_vals_outside = torch.linspace(1e-3, 1.0 - 1.0 / (self.n_outside + 1.0), self.n_outside)

        n_samples = self.n_samples
        perturb = self.perturb

        """ This block checks if perturbation should be applied to the depth samples and adjusts them accordingly.
        Perturbation is a technique to add randomness to the sampling process, reducing aliasing artifacts. """
        if perturb_overwrite >= 0:
            perturb = perturb_overwrite
        if perturb > 0:
            t_rand = (torch.rand([batch_size, 1]) - 0.5).cuda()
            z_vals = z_vals + t_rand * 2.0 / self.n_samples

            if self.n_outside > 0:
                mids = .5 * (z_vals_outside[..., 1:] + z_vals_outside[..., :-1])
                upper = torch.cat([mids, z_vals_outside[..., -1:]], -1)
                lower = torch.cat([z_vals_outside[..., :1], mids], -1)
                t_rand = torch.rand([batch_size, z_vals_outside.shape[-1]])
                z_vals_outside = lower[None, :] + (upper - lower)[None, :] * t_rand

        if self.n_outside > 0:
            z_vals_outside = z_vals_outside.cuda()
            z_vals_outside = far / torch.flip(z_vals_outside, dims=[-1]) + 1.0 / self.n_samples


        background_alpha = None
        background_sampled_color = None

        # Up sample
        """ If importance sampling (n_importance) is enabled, this block refines the depth sampling by adding more
        samples in regions with high variation. The up-sampling process uses the SDF (Signed Distance Field)
        network to determine regions with significant features. """
        if self.n_importance > 0:
            with torch.no_grad():
                pts = rays_o[:, None, :] + rays_d[:, None, :] * z_vals[..., :, None]
                # fg_sdf_raw = self.fg_sdf_network()
                sdf = self.sdf_network.sdf(pts.reshape(-1, 3), iteration, embedder_position=self.fg_embedder_position).reshape(batch_size, self.n_samples)
                # print(pts.reshape(-1,3).min(dim=0), pts.reshape(-1,3).max(dim=0))

                for i in range(self.up_sample_steps):
                    new_z_vals = self.up_sample(rays_o,
                                                rays_d,
                                                z_vals,
                                                sdf,
                                                self.n_importance // self.up_sample_steps,
                                                64 * 2**i)
                    z_vals, sdf = self.cat_z_vals(rays_o,
                                                  rays_d,
                                                  z_vals,
                                                  new_z_vals,
                                                  sdf,
                                                  iteration,
                                                  last=(i + 1 == self.up_sample_steps))

            n_samples = self.n_samples + self.n_importance

        # Background model
        """ If background rendering is enabled, this block renders the background of the scene using the outside depth samples. """
        if self.n_outside > 0:
            z_vals_feed = torch.cat([z_vals, z_vals_outside], dim=-1)
            z_vals_feed, _ = torch.sort(z_vals_feed, dim=-1)
            ret_outside = self.render_core_outside(rays_o, rays_d, z_vals_feed, sample_dist, iteration)

            background_sampled_color = ret_outside['sampled_color']
            background_alpha = ret_outside['alpha']

        # Render core
        ret_fine = self.render_core(rays_o,
                                    rays_d,
                                    z_vals,
                                    sample_dist,
                                    iteration,
                                    background_rgb=background_rgb,
                                    background_alpha=None,
                                    background_sampled_color=None)

        color_fine = ret_fine['color']
        weights = ret_fine['weights']
        weights_sum = weights.sum(dim=-1, keepdim=True)
        gradients = ret_fine['gradients']
        s_val = ret_fine['s_val'].reshape(batch_size, n_samples).mean(dim=-1, keepdim=True)
        z_vals = ret_fine['mid_z_vals']

        # Calculate weighted depth
        weighted_depth = ((z_vals * weights).sum(dim=-1) / weights_sum.squeeze(-1))
        
        # Compute normals using the provided snippet, adjusted for PyTorch tensor operations
        n_samples = self.n_samples + (self.n_importance if hasattr(self, 'n_importance') else 0)
        normals = gradients * weights[:, :, None]  # Ensure dimensions are correct
        normals = normals.sum(dim=1)
        
        # Optional: Normalize normals to unit vectors if they aren't already
        normals = torch.nn.functional.normalize(normals, p=2, dim=1)

        return {
            'depth': weighted_depth,
            'color_fine': color_fine,
            's_val': s_val,
            'cdf_fine': ret_fine['cdf'],
            'weight_sum': weights_sum,
            'weight_max': torch.max(weights, dim=-1, keepdim=True)[0],
            'gradients': gradients,
            'weights': weights,
            'normals': normals,
            'gradient_error': ret_fine['gradient_error'],
            'inside_sphere': ret_fine['inside_sphere']
        }

    # def extract_geometry(self, bound_min, bound_max, resolution, iteration, threshold=0.0):
    #     return extract_geometry(bound_min,
    #                             bound_max,
    #                             resolution=resolution,
    #                             threshold=threshold,
    #                             query_func=lambda pts: -self.sdf_network.sdf(pts, iteration))


def remap_name(name):
    name = name.replace('.', '-')  # dot is not allowed by pytorch
    if name[-1] == '/':
        name = name[:-1]
    idx = name.rfind('/')
    for i in range(2):
        if idx >= 0:
            idx = name[:idx].rfind('/')
    return name[idx + 1:]

class NeuSNetWrapper(nn.Module):
    def __init__(self, args, optim_autoexpo=False, img_names=None):
        super().__init__()

        self.neus_net = NeusRenderer(args)

        self.optim_autoexpo = optim_autoexpo
        if self.optim_autoexpo:
            assert(img_names is not None)
            logger.info('Optimizing autoexposure!')

            self.img_names = [remap_name(x) for x in img_names]
            logger.info('\n'.join(self.img_names))
            self.autoexpo_params = nn.ParameterDict(OrderedDict([(x, nn.Parameter(torch.Tensor([0.5, 0.]))) for x in self.img_names]))

    def forward(self, ray_o, ray_d, near, far, iteration, background_rgb, perturb_overwrite, img_name=None):
        '''
        :param ray_o, ray_d: [..., 3]
        :param fg_z_max: [...,]
        :param fg_z_vals: [..., N_samples]
        :return
        '''
        # if img_name is not None:
        #     img_name = remap_name(img_name)

        if img_name is not None:
            img_name = remap_name(img_name)

        ret = self.neus_net.render(ray_o, ray_d, near, far, iteration, background_rgb, perturb_overwrite)
        if self.optim_autoexpo and (img_name in self.autoexpo_params):
            autoexpo = self.autoexpo_params[img_name]
            scale = torch.abs(autoexpo[0]) + 0.5  # make sure scale is always positive
            shift = autoexpo[1]
            ret['autoexpo'] = (scale, shift)

        return ret

