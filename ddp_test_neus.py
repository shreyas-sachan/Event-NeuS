#!/usr/bin/env python3
import torch
# import torch.nn as nn
import torch.optim
import torch.distributed
# from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing
import numpy as np
import os

import cv2 
# from collections import OrderedDict
# from ddp_model import NerfNet
import time
from data_loader_split import load_event_data_split
from utils import mse2psnr, colorize_np, to8b
import imageio
# from ddp_train_neus import config_parser, setup_logger, setup, cleanup, render_single_image, create_nerf, near_far_from_sphere
from ddp_train_neus import config_parser, setup_logger, setup, cleanup, create_nerf, near_far_from_sphere
import logging
from nerf_sample_ray_split import CameraManager


logger = logging.getLogger(__package__)


def ddp_test(rank, args):
    ###### set up multi-processing
    setup(rank, args.world_size)
    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()

    ###### decide chunk size according to gpu memory
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1
        args.chunk_size = 1
    else:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 1
        args.chunk_size = 1

    ###### create network and wrap in ddp; each process should do this
    camera_mgr = CameraManager(learnable=False)
    start, models = create_nerf(rank, args, camera_mgr, load_camera_mgr=False)

    # datatype = 'test'
    # render_splits = [x.strip() for x in datatype.strip().split(',')]

    render_splits = [x.strip() for x in args.render_splits.strip().split(',')]

    u = models['net_0']
    neus_net = u.neus_net
    neus_net.fg_embedder_position.use_annealing = False
    neus_net.fg_embedder_viewdir.use_annealing = False

    # start testing
    for split in render_splits:
        out_dir = os.path.join(args.basedir, args.expname,
                               'render_{}_{:06d}'.format(split, start))
        if rank == 0:
            os.makedirs(out_dir, exist_ok=True)

        ###### load data and create ray samplers; each process should do this
        ray_samplers = load_event_data_split(args.datadir, args.scene, split,
                camera_mgr=models['camera_mgr'], max_winsize=1,
                use_ray_jitter=args.use_ray_jitter, is_colored=args.is_colored,
                polarity_offset=args.polarity_offset,
                skip=args.testskip, cycle=args.is_cycled,
                is_rgb_only=args.is_rgb_only)


        for idx in range(len(ray_samplers)):
        # for idx in range(50):
            ray_samplers[idx].update_rays(camera_mgr)
            rays = ray_samplers[idx].get_all()
            
            # print(f"Rays at idx {idx}: ray_o = {rays['ray_o'][:5]}, ray_d = {rays['ray_d'][:5]}")
            
            rays_o_split = torch.split(rays['ray_o'], 1024)
            rays_d_split = torch.split(rays['ray_d'], 1024)
            
            res_color = []
            res_normal = []
            # res_depth = []
            
            for x in range(len(rays_d_split)):
                rays_o = rays_o_split[x].cuda()
                rays_d = rays_d_split[x].cuda()
                near, far = near_far_from_sphere(rays_o, rays_d)
                
                # background_rgb = torch.full([1, 3], 0.5).cuda()
                background_rgb = torch.full([1, 3], 0.5).cuda()

                neus_net_render = neus_net.render(rays_o, rays_d, near, far, start, background_rgb, perturb_overwrite = 0)
                
                # neus_net_render = neus_net.render(rays_o, rays_d, near, far, -1)
            
                # print("------------- Neus_net_render --------------", neus_net_render)
               
                n_samples = neus_net.n_samples + neus_net.n_importance
                normals = neus_net_render['gradients'] * neus_net_render['weights'][:, :n_samples, None]
                # if neus_net_render('inside_sphere'):
                #     normals = normals * neus_net_render['inside_sphere'][..., None]
                normals = normals.sum(dim=1).detach().cpu().numpy()

                # depth = neus_net_render['depth'].detach().cpu().numpy()
                
                res_normal.append(normals)
                res_color.append(neus_net_render['color_fine'].detach().cpu())
                # res_depth.append(depth)
                
                del rays_o
                del rays_d
                del neus_net_render
                
                torch.cuda.empty_cache()
            
            if len(res_color) > 0:
                color_img = torch.cat(res_color).reshape(260, 346, 3).cpu().numpy()
                color_img = (color_img *255).astype(np.uint8)
                color_filename = os.path.join(out_dir, f'test-color_{idx}.png')  # Unique filename for color image
                cv2.imwrite(color_filename, color_img)
            
            if len(res_normal) > 0:
                normal_img = (np.concatenate(res_normal, axis=0).reshape((260, 346, 3)) * 128 + 128).clip(0, 255)
                normal_filename = os.path.join(out_dir, f'test-normal_{idx}.png')  # Unique filename for normal image
                cv2.imwrite(normal_filename, normal_img)

            
            # # print ("---------- DEPTH ------------", res_depth)
            # if len(res_depth) > 0:
            #     depth_img = np.concatenate(res_depth).reshape(260, 346)
            #     # Normalize depth to [0, 1]
            #     depth_img_normalized = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())
                
            #     # Scale to [0, 255] and convert to uint8
            #     depth_img_scaled = (depth_img_normalized * 255).astype(np.uint8)
                
            #     depth_filename = os.path.join(out_dir, f'test-depth_{idx}.png')  # Unique filename for normal image
            #     cv2.imwrite(depth_filename, depth_img_scaled)

            
            # print(" --------------------BS -------------------",Fcolor.max(), Fcolor.min())

            cv2.imwrite('test-color.png', color_img)
            cv2.imwrite('test-normal.png', normal_img)
            # cv2.imwrite('test-depth.png', depth_img)
            

    # clean up for multi-processing
    cleanup()

def test():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())

    args.world_size = 1
    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))

    ddp_test(0, args)

if __name__ == '__main__':
    setup_logger()
    test()



# Try annealing
# Render without using view directions
# check the effect of PE and annealing
# Uniform sampling while rendering the normals 

# try implementing different camera trajectroy (spiral trajectory)
# Render more synthetic and real data results

# Problems: Hand missing, chair background hollow, plane on the top, blurry geometry and normals