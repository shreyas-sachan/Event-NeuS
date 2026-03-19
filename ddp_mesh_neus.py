#!/usr/bin/env python3
import numpy as np
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
# import mcubes
import marching_cubes as mcubes
import logging
import trimesh
from tqdm import tqdm, trange
from ddp_train_neus import config_parser, setup_logger, setup, create_nerf
from nerf_sample_ray_split import CameraManager
import os

logger = logging.getLogger(__package__)

def ddp_mesh_neus(rank, args, resolution=300):
    ###### set up multi-processing
    assert(args.world_size==1)
    setup(rank, args.world_size)
    ###### set up logger
    logger = logging.getLogger(__package__)
    setup_logger()

    ###### decide chunk size according to gpu memory
    if torch.cuda.get_device_properties(rank).total_memory / 1e9 > 14:
        logger.info('setting batch size according to 24G gpu')
        args.N_rand = 1024
        args.chunk_size = 8192
    else:
        logger.info('setting batch size according to 12G gpu')
        args.N_rand = 512
        args.chunk_size = 4096

    ###### create network and wrap in ddp; each process should do this
    camera_mgr = CameraManager(learnable=False)
    start, models = create_nerf(rank, args, camera_mgr, False)

    # center on lk
    ax = np.linspace(-1, 1, num=500, endpoint=True, dtype=np.float32)
    # X, Y, Z = np.meshgrid(ax, ax, ax+0.4)
    X, Y, Z = np.meshgrid(ax, ax, ax)


    # flip yz
    pts = np.stack((X, Y[::-1], Z[::-1]), -1)/4
    pts = pts.reshape((-1, 3))

    pts = torch.tensor(pts).float().to(rank)

    u = models['net_0']
    neus_net = u.neus_net
    neus_net = u.neus_net
    neus_net.fg_embedder_position.use_annealing = False
    neus_net.fg_embedder_viewdir.use_annealing = False
    sdf_net = neus_net.sdf_network
    color_net = neus_net.color_network

    allres = []
    allcolor = []
    with autocast():
        # with torch.no_grad():
            # direction = torch.tensor([0, 0, -1], dtype=torch.float32).to(rank)
            for bid in trange((pts.shape[0]+args.chunk_size-1)//args.chunk_size):
                bstart = bid * args.chunk_size
                bend = bstart + args.chunk_size
                cpts = pts[bstart:bend]
                # cvd = cpts*0 #+direction
                
                # x, y, z = cpts[..., 0], cpts[..., 1], cpts[..., 2]
                # r2 = (x**2+z**2)
                # mask = (r2 <= args.crop_r**2)
                # mask = mask & (y >= args.crop_y_min) & (y <= args.crop_y_min)
                
                # print("----Checkpoints ---------\n ", cpts.min(dim=0).values,cpts.max(dim=0).values)

                out_sdf = sdf_net.sdf(cpts, iteration=start, embedder_position=neus_net.fg_embedder_position).detach().cpu().numpy()
                # mask = mask.reshape(out_sdf.shape).cpu().numpy()
                # out_sdf[~mask] = 100000
                feature_vector = sdf_net.sdf_hidden_appearance(cpts, iteration=start, embedder_position=neus_net.fg_embedder_position)
                gradients = sdf_net.gradient(cpts, iteration=start, embedder_position=neus_net.fg_embedder_position).squeeze()
                
                cvd = -gradients
                
                out_color = color_net(cpts, gradients, cvd, feature_vector, iteration=start, embedder_viewdir=neus_net.fg_embedder_viewdir).detach().cpu().numpy()
                
                # out_color = out_color[..., ::-1]

                allres.append(out_sdf)
                allcolor.append(out_color)
    # print("----------- Vertices --------------", vertices)
    # print("----------- Triangles --------------", triangles)
        
    allres = np.concatenate(allres, 0).reshape(X.shape)

    allcolor = np.concatenate(allcolor,0).reshape(list(X.shape) + [3,])

    # print("----------------- allres DONE -------------", allres)

    # allcolor = np.concatenate(allcolor, 0)
    # allcolor = allcolor.reshape(list(X.shape)+[3,])

    print("allres Details", allres.min(), allres.max(), allres.mean(), np.median(allres), allres.shape)
    print("allcolor Details", allcolor.min(), allcolor.max(), allcolor.mean(), np.median(allcolor), allcolor.shape)

    logger.info('Doing MC')
    # vtx, tri = mcubes.marching_cubes(allres.astype(np.float32), 0)
    vtx, tri = mcubes.marching_cubes_color(allres.astype(np.float32), allcolor.astype(np.float32), 0)
    
    print(vtx, tri)
    
    THR=np.median(allres)
    # THR=30
    print("THR done", THR)
    # vtx, tri = mcubes.marching_cubes(sigma.astype(np.float32), THR)
    print("------------ MC DONE ---------")
    # vtx, tri = mcubes.marching_cubes_color(allres.astype(np.float32), allcolor.astype(np.float32), THR)
    logger.info('Exporting mesh')
    
    output_dir = os.path.join(args.basedir, args.expname, 'mesh')
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{args.expname}.obj")
    mcubes.export_obj(vtx, tri, output_path)

def mesh():
    parser = config_parser()
    args = parser.parse_args()
    logger.info(parser.format_values())


    args.world_size = 1     
    if args.world_size == -1:
        args.world_size = torch.cuda.device_count()
        logger.info('Using # gpus: {}'.format(args.world_size))
    # torch.multiprocessing.spawn(ddp_mesh_neus,
    #                             args=(args,),
    #                             nprocs=args.world_size,
    #                             join=True)
    
    ddp_mesh_neus(0, args)

if __name__ == '__main__':
    setup_logger()
    mesh()
