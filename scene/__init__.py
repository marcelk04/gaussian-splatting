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
import random
import json
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks
from scene.gaussian_model import GaussianModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

def load_scene_info(dataset: ModelParams):
    if os.path.exists(os.path.join(dataset.source_path, "sparse")):
        scene_info = sceneLoadTypeCallbacks["Colmap"](dataset.source_path, dataset.images, dataset.depths, dataset.eval, dataset.train_test_exp)
    elif os.path.exists(os.path.join(dataset.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        scene_info = sceneLoadTypeCallbacks["Blender"](dataset.source_path, dataset.white_background, dataset.depths, dataset.eval)
    else:
        assert False, "Could not recognize scene type!"

    return scene_info

def write_cam_list(scene_info, model_path):
    with open(scene_info.ply_path, 'rb') as src_file, open(os.path.join(model_path, "input.ply") , 'wb') as dest_file:
        dest_file.write(src_file.read())
    json_cams = []
    camlist = []
    if scene_info.test_cameras:
        camlist.extend(scene_info.test_cameras)
    if scene_info.train_cameras:
        camlist.extend(scene_info.train_cameras)
    for id, cam in enumerate(camlist):
        json_cams.append(camera_to_JSON(id, cam))
    with open(os.path.join(model_path, "cameras.json"), 'w') as file:
        json.dump(json_cams, file)

def load_cams(dataset: ModelParams, scene_info, resolution_scales):
    train_cameras = {}
    test_cameras = {}
    cameras_extent = scene_info.nerf_normalization["radius"]

    for resolution_scale in resolution_scales:
        train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, dataset, scene_info.is_nerf_synthetic, False)
        test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, dataset, scene_info.is_nerf_synthetic, True)

    return train_cameras, test_cameras, cameras_extent

class SeparatedScene:
    gaussians: GaussianModel

    def __init__(self, dataset1: ModelParams, dataset2: ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0]):
        self.model_path = dataset1.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        scene_info1 = load_scene_info(dataset1)
        scene_info2 = load_scene_info(dataset2)

        if not self.loaded_iter:
            write_cam_list(scene_info1, self.model_path)

        if shuffle:
            train_indices = list(range(len(scene_info1.train_cameras)))
            test_indices = list(range(len(scene_info1.test_cameras)))

            random.shuffle(train_indices)
            random.shuffle(test_indices)

            scene_info1.train_cameras = [scene_info1.train_cameras[i] for i in train_indices]
            scene_info2.train_cameras = [scene_info2.train_cameras[i] for i in train_indices]

            scene_info1.test_cameras = [scene_info1.test_cameras[i] for i in test_indices]
            scene_info2.test_cameras = [scene_info2.test_cameras[i] for i in test_indices]

        self.train_cameras1, self.test_cameras1, self.cameras_extent = load_cams(dataset1, scene_info1, resolution_scales)
        self.train_cameras2, self.test_cameras2, _ = load_cams(dataset2, scene_info2, resolution_scales)

        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path, "point_cloud", "iteration_" + str(self.loaded_iter), "point_cloud.ply"), dataset1.train_test_exp)
        else:
            self.gaussians.create_from_pcd(scene_info1.point_cloud, scene_info1.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras1(self, scale=1.0):
        return self.train_cameras1[scale]

    def getTestCameras1(self, scale=1.0):
        return self.test_cameras1[scale]

    def getTrainCameras2(self, scale=1.0):
        return self.train_cameras2[scale]

    def getTestCameras2(self, scale=1.0):
        return self.test_cameras2[scale]

class Scene:
    gaussians : GaussianModel

    def __init__(self, args : ModelParams, gaussians : GaussianModel, load_iteration=None, shuffle=True, resolution_scales=[1.0], load_gaussians=True):
        """b
        :param path: Path to colmap scene main folder.
        """
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        scene_info = load_scene_info(args)

        if not self.loaded_iter:
            write_cam_list(scene_info, self.model_path)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, False)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic, True)

        if load_gaussians:
            if self.loaded_iter:
                self.gaussians.load_ply(os.path.join(self.model_path,
                                                            "point_cloud",
                                                            "iteration_" + str(self.loaded_iter),
                                                            "point_cloud.ply"), args.train_test_exp)
            else:
                self.gaussians.create_from_pcd(scene_info.point_cloud, scene_info.train_cameras, self.cameras_extent)

    def save(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        exposure_dict = {
            image_name: self.gaussians.get_exposure_from_name(image_name).detach().cpu().numpy().tolist()
            for image_name in self.gaussians.exposure_mapping
        }

        with open(os.path.join(self.model_path, "exposure.json"), "w") as f:
            json.dump(exposure_dict, f, indent=2)

    def getTrainCameras(self, scale=1.0):
        return self.train_cameras[scale]

    def getTestCameras(self, scale=1.0):
        return self.test_cameras[scale]
