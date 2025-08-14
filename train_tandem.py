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

import os
import json
import copy
import torch
import torchvision
from random import randint
from utils.loss_utils import l1_loss, ssim
from lpipsPyTorch.modules.lpips import LPIPS
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from scene.cameras import Camera
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

def create_viewpoint_stack(scene: Scene) -> tuple[list[Camera], list[int]]:
    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    return viewpoint_stack, viewpoint_indices

def render_viewpoint(viewpoint_cam: Camera, gaussians: GaussianModel, pipe: PipelineParams, bg, dataset: ModelParams):
    render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
    return render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

def calculate_loss(gt_image, rendered_image, opt: OptimizationParams):
    Ll1 = l1_loss(rendered_image, gt_image)
    if FUSED_SSIM_AVAILABLE:
        ssim_value = fused_ssim(rendered_image.unsqueeze(0), gt_image.unsqueeze(0))
    else:
        ssim_value = ssim(rendered_image, gt_image)

    return (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

def gaussian_optimization(gaussians: GaussianModel, scene: Scene, iteration: int, dataset: ModelParams, opt: OptimizationParams, viewspace_point_tensor, visibility_filter, radii, use_sparse_adam: bool):
    # Densification
    if iteration < opt.densify_until_iter:
        # Keep track of max radii in image-space for pruning
        gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
        gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

        if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
            size_threshold = 20 if iteration > opt.opacity_reset_interval else None
            gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
        
        if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
            gaussians.reset_opacity()

    # Optimizer step
    if iteration < opt.iterations:
        gaussians.exposure_optimizer.step()
        gaussians.exposure_optimizer.zero_grad(set_to_none = True)
        if use_sparse_adam:
            visible = radii > 0
            gaussians.optimizer.step(visible, radii.shape[0])
            gaussians.optimizer.zero_grad(set_to_none = True)
        else:
            gaussians.optimizer.step()
            gaussians.optimizer.zero_grad(set_to_none = True)

def evaluate_performance(results_obj, scene, dataset, pipe, background, lpips_net):
    psnr_test = 0.0
    ssim_test = 0.0
    lpips_test = 0.0

    for viewpoint in scene.getTestCameras():
        image = torch.clamp(render(viewpoint, scene.gaussians, pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp)["render"], 0.0, 1.0)
        gt_image = torch.clamp(viewpoint.original_image.to(dataset.data_device), 0.0, 1.0)

        psnr_test += psnr(image, gt_image).mean().double().item()
        ssim_test += ssim(image, gt_image).mean().double().item()
        lpips_test += lpips_net(image, gt_image).mean().double().item()

    results_obj["psnr"]["value"].append(psnr_test / len(scene.getTestCameras()))
    results_obj["ssim"]["value"].append(ssim_test / len(scene.getTestCameras()))
    results_obj["lpips"]["value"].append(lpips_test / len(scene.getTestCameras()))

def combine_images(img1, img2):
    return torch.clamp((img1 ** 2.2) + (img2 ** 2.2), 0.0, 1.0) ** (1.0 / 2.2)

def training(dataset1: ModelParams, dataset2: ModelParams, opt: OptimizationParams, pipe: PipelineParams, testing_iterations: list[int], saving_iterations: list[int], checkpoint_iterations: list[int], checkpoint: str, debug_from: int) -> None:

    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    _ = prepare_output_and_logger(dataset1)
    _ = prepare_output_and_logger(dataset2)

    # Initialize separate GaussianModels and Scenes
    gaussians1 = GaussianModel(dataset1.sh_degree, opt.optimizer_type)
    scene1 = Scene(dataset1, gaussians1, shuffle=False)
    gaussians1.training_setup(opt)
    
    gaussians2 = GaussianModel(dataset2.sh_degree, opt.optimizer_type)
    scene2 = Scene(dataset2, gaussians2, shuffle=False)
    gaussians2.training_setup(opt)

    # if checkpoint:
    #     (model_params, first_iter) = torch.load(checkpoint)
    #     gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset1.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device=dataset1.data_device)

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack1, viewpoint_indices1 = create_viewpoint_stack(scene1)
    viewpoint_stack2, viewpoint_indices2 = create_viewpoint_stack(scene2)

    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    train_results_obj = {
        "loss": [],
        "points": [],
        "psnr": {
            "iteration": testing_iterations,
            "value": []
        },
        "ssim": {
            "iteration": testing_iterations,
            "value": []
        },
        "lpips": {
            "iteration": testing_iterations,
            "value": []
        }
    }

    train_results_model1 = copy.deepcopy(train_results_obj)
    train_results_model2 = copy.deepcopy(train_results_obj)
    train_results_model_combined = copy.deepcopy(train_results_obj)

    lpips_net = LPIPS(net_type="vgg").to(dataset1.data_device)

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()

        gaussians1.update_learning_rate(iteration)
        gaussians2.update_learning_rate(iteration)

        if iteration % 1000 == 0:
            gaussians1.oneupSHdegree()
            gaussians2.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack1 or not viewpoint_stack2:
            viewpoint_stack1, viewpoint_indices1 = create_viewpoint_stack(scene1)
            viewpoint_stack2, viewpoint_indices2 = create_viewpoint_stack(scene2)

        rand_idx = randint(0, len(viewpoint_indices1) - 1)
        viewpoint_cam1 = viewpoint_stack1.pop(rand_idx)
        viewpoint_cam2 = viewpoint_stack2.pop(rand_idx)
        _ = viewpoint_indices1.pop(rand_idx)
        _ = viewpoint_indices2.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device=dataset1.data_device) if opt.random_background else background

        image1, viewspace_point_tensor1, visibility_filter1, radii1 = render_viewpoint(viewpoint_cam1, gaussians1, pipe, bg, dataset1)
        image2, viewspace_point_tensor2, visibility_filter2, radii2 = render_viewpoint(viewpoint_cam2, gaussians2, pipe, bg, dataset2)

        # Loss
        gt_image1 = viewpoint_cam1.original_image.to(dataset1.data_device)
        gt_image2 = viewpoint_cam2.original_image.to(dataset2.data_device)

        loss1 = calculate_loss(gt_image1, image1, opt)
        loss2 = calculate_loss(gt_image2, image2, opt)

        gt_combined = gt_image1 + gt_image2
        render_combined = image1 + image2

        loss_combined = calculate_loss(gt_combined, render_combined, opt)
        total_loss = loss1 + loss2 + loss_combined

        total_loss.backward()

        with torch.no_grad():
            gaussian_optimization(gaussians1, scene1, iteration, dataset1, opt, viewspace_point_tensor1, visibility_filter1, radii1, use_sparse_adam)
            gaussian_optimization(gaussians2, scene2, iteration, dataset2, opt, viewspace_point_tensor2, visibility_filter2, radii2, use_sparse_adam)

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            # ema_Ll1depth_for_log = 0.4 * Ll1depth + 0.6 * ema_Ll1depth_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            train_results_model1["loss"].append(loss1.item())
            train_results_model2["loss"].append(loss2.item())
            train_results_model_combined["loss"].append(total_loss.item())

            train_results_model1["points"].append(scene1.gaussians.get_xyz.shape[0])
            train_results_model2["points"].append(scene2.gaussians.get_xyz.shape[0])
            train_results_model_combined["points"].append(scene1.gaussians.get_xyz.shape[0] + scene2.gaussians.get_xyz.shape[0])

            # Evaluate test performance
            if (iteration in testing_iterations):
                evaluate_performance(train_results_model1, scene1, dataset1, pipe, background, lpips_net)
                evaluate_performance(train_results_model2, scene2, dataset2, pipe, background, lpips_net)

                # Evaluate combined performance
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0

                for i in range(len(scene1.getTestCameras())):
                    viewpoint1 = scene1.getTestCameras()[i]
                    viewpoint2 = scene2.getTestCameras()[i]

                    image1 = render(viewpoint1, scene1.gaussians, pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset1.train_test_exp)["render"]
                    image2 = render(viewpoint2, scene2.gaussians, pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset2.train_test_exp)["render"]
                    render_combined = combine_images(image1, image2)

                    gt_image1 = viewpoint1.original_image.to(dataset1.data_device)
                    gt_image2 = viewpoint2.original_image.to(dataset2.data_device)
                    gt_combined = combine_images(gt_image1, gt_image2)

                    psnr_test += psnr(render_combined, gt_combined).mean().double().item()
                    ssim_test += ssim(render_combined, gt_combined).mean().double().item()
                    lpips_test += lpips_net(render_combined, gt_combined).mean().double().item()

                train_results_model_combined["psnr"]["value"].append(psnr_test / len(scene1.getTestCameras()))
                train_results_model_combined["ssim"]["value"].append(ssim_test / len(scene1.getTestCameras()))
                train_results_model_combined["lpips"]["value"].append(lpips_test / len(scene1.getTestCameras()))

            # Log and save
            # TODO: skip normal logging for now
            # training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), dataset.train_test_exp)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene1.save(iteration)
                scene2.save(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians1.capture(), iteration), scene1.model_path + "/chkpnt" + str(iteration) + ".pth")
                torch.save((gaussians2.capture(), iteration), scene2.model_path + "/chkpnt" + str(iteration) + ".pth")

    # Save evaluated metrics on test views
    with open(os.path.join(dataset1.model_path, "train_results.json"), "w") as outfile:
        json.dump(train_results_model1, outfile, indent=None)

    with open(os.path.join(dataset2.model_path, "train_results.json"), "w") as outfile:
        json.dump(train_results_model2, outfile, indent=None)

    with open(os.path.join(dataset1.model_path, "train_results_combined.json"), "w") as outfile:
        json.dump(train_results_model_combined, outfile, indent=None)

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if train_test_exp:
                        image = image[..., image.shape[-1] // 2:]
                        gt_image = gt_image[..., gt_image.shape[-1] // 2:]
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=list(range(500, 30_001, 500)))
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--source1", type=str, required=True)
    parser.add_argument("--source2", type=str, required=True)
    parser.add_argument("--model1", type=str, required=True)
    parser.add_argument("--model2", type=str, required=True)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    dataset1 = lp.extract(args)
    dataset2 = copy.deepcopy(dataset1)

    dataset1.source_path = os.path.abspath(args.source1)
    dataset2.source_path = os.path.abspath(args.source2)
    dataset1.model_path = os.path.abspath(args.model1)
    dataset2.model_path = os.path.abspath(args.model2)

    training(dataset1, dataset2, op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
