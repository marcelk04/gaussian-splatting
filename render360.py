import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
import torchvision
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import cv2

from gaussian_renderer import render
from utils.general_utils import safe_state
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from scene.cameras import Camera

try:
	from diff_gaussian_rasterization import SparseGaussianAdam
	SPARSE_ADAM_AVAILABLE = True
except:
	SPARSE_ADAM_AVAILABLE = False

def lookAt(center, target, up):
    f = (target - center); f = f/np.linalg.norm(f)
    s = np.cross(f, up); s = s/np.linalg.norm(s)
    u = np.cross(s, f); u = u/np.linalg.norm(u)

    m = np.zeros((4, 4))
    m[0, :-1] = s
    m[1, :-1] = u
    m[2, :-1] = f
    m[-1, -1] = 1.0

    return m

to8b = lambda x : (255*np.clip(x.cpu().numpy(),0,1)).astype(np.uint8)

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, separate_sh: bool, output: str):
	with torch.no_grad():
		gaussians = GaussianModel(dataset.sh_degree)
		scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

		if output == "":
			render_path = os.path.join(dataset.model_path, "test", "ours_{}".format(scene.loaded_iter), "video")
			video_output = os.path.join(render_path, "output.mp4")
		else:
			render_path = os.path.dirname(output)
			video_output = output

		makedirs(render_path, exist_ok=True)

		bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
		background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

		cams = scene.getTrainCameras()

		w, h = Image.open(os.path.join(dataset.source_path, "images", "0000.png")).size
		resolution = (h, w)
		FoVx = cams[0].FoVx
		FoVy = cams[0].FoVy
		depth_params = None
		image = Image.new("RGB", (1, 1))
		invdepthmap = None
		image_name = cams[0].image_name
		data_device= "cuda"
		train_test_exp = False
		is_test_dataset = False
		is_test_view = False

		center = np.column_stack([-c.R @ c.T for c in cams]).sum(axis=1) / len(cams)
		n = 360
		angles = np.radians(np.linspace(0, 360, n))
		distance = 75
		
		out = cv2.VideoWriter(video_output, cv2.VideoWriter_fourcc(*'mp4v'), 30, resolution)

		for i, angle in tqdm(enumerate(angles), desc="Rendering", total=n):
			uid = i
			colmap_id = i+1

			offset = np.array([distance * np.sin(angle), 0, distance * np.cos(angle)])
			T = center + offset

			V = lookAt(T, center, np.array([0,-1,0]))

			R = V[:3, :3].T
			T = -np.dot(T, R)

			view = Camera(resolution=resolution, colmap_id=colmap_id, R=R, T=T, FoVx=FoVx, FoVy=FoVy, depth_params=depth_params, image=image, invdepthmap=invdepthmap, image_name=image_name, uid=uid, data_device=data_device, train_test_exp=train_test_exp, is_test_dataset=is_test_dataset, is_test_view=is_test_view)

			rendering = render(view, gaussians, pipeline, background, use_trained_exp=train_test_exp, separate_sh=separate_sh)["render"]

			np_img = to8b(rendering).transpose(1, 2, 0)
			cv2_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)

			out.write(cv2_img)

		out.release()
		cv2.destroyAllWindows()

if __name__ == "__main__":
	# Set up command line argument parser
	parser = ArgumentParser(description="Testing script parameters")
	model = ModelParams(parser, sentinel=True)
	pipeline = PipelineParams(parser)
	parser.add_argument("--iteration", default=-1, type=int)
	parser.add_argument("--quiet", action="store_true")
	parser.add_argument("--output", default="", type=str, required=False)
	args = get_combined_args(parser)
	print("Rendering " + args.model_path)

	render_sets(model.extract(args), args.iteration, pipeline.extract(args), SPARSE_ADAM_AVAILABLE, args.output)