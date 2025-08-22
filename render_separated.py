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
from scene import Scene, SeparatedScene
import os
from os import makedirs
import copy
from tqdm import tqdm
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, train_test_exp, separate_sh, output, shs_idx):
    if output == "":
        render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    else:
        render_path = os.path.join(output, name, "ours_{}".format(iteration), "renders")
        gts_path = os.path.join(output, name, "ours_{}".format(iteration), "gt")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh, shs_idx=shs_idx)["render"]
        gt = view.original_image[0:3, :, :]

        if args.train_test_exp:
            rendering = rendering[..., rendering.shape[-1] // 2:]
            gt = gt[..., gt.shape[-1] // 2:]

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))

def render_sets(dataset1 : ModelParams, dataset2: ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, separate_sh: bool, output: str, shs_idx: int):
    with torch.no_grad():
        gaussians = GaussianModel(dataset1.sh_degree, two_shs=True)
        scene = SeparatedScene(dataset1, dataset2, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset1.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            cams = scene.getTrainCameras1() if shs_idx==0 else scene.getTrainCameras2()
            render_set(dataset1.model_path, "train", scene.loaded_iter, cams, gaussians, pipeline, background, dataset1.train_test_exp, separate_sh, output, shs_idx)

        if not skip_test:
            cams = scene.getTestCameras1() if shs_idx==0 else scene.getTestCameras2()
            render_set(dataset1.model_path, "test", scene.loaded_iter, cams, gaussians, pipeline, background, dataset1.train_test_exp, separate_sh, output, shs_idx)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--output", default="", type=str, required=False)
    parser.add_argument("--shs_idx", default=0, type=int, required=False)
    parser.add_argument("--source1", type=str, required=True)
    parser.add_argument("--source2", type=str, required=True)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    dataset1 = model.extract(args)
    dataset2 = copy.deepcopy(dataset1)

    dataset1.source_path = os.path.abspath(args.source1)
    dataset1.white_background = False

    dataset2.source_path = os.path.abspath(args.source2)
    dataset2.white_background = False

    render_sets(dataset1, dataset2, args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, SPARSE_ADAM_AVAILABLE, args.output, args.shs_idx)