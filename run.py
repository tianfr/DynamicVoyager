import gc
import copy
import math
import random
import time
from argparse import ArgumentParser
from pathlib import Path
from PIL import Image
from datetime import datetime
from copy import deepcopy
import json
from einops import rearrange
from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
import numpy as np
import torch
from omegaconf import OmegaConf
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm
from diffusers import StableDiffusionInpaintPipeline, AutoencoderKL, DPMSolverMultistepScheduler, LatentConsistencyModelPipeline
from diffusers import AutoPipelineForImage2Image

import sys,os
sys.path.append('midas_module')
from midas_module.midas.model_loader import load_model
import torch.nn.functional as F

from models.models import KeyframeGen, KeyframeInterp, save_point_cloud_as_ply, inpaint_cv2, PointsRenderer, SoftmaxImportanceCompositor
from pytorch3d.renderer import PerspectiveCameras, PointsRasterizationSettings, PointsRasterizer
from pytorch3d.structures import Pointclouds
from util.finetune_utils import finetune_depth_model, finetune_decoder, finetune_video_depth_model
from util.chatGPT4 import TextpromptGen
from util.general_utils import apply_depth_colormap, save_video
from util.utils import save_depth_map, prepare_scheduler, video2images_new
from util.utils import load_example_yaml, merge_frames, merge_keyframes, empty_cache, clear_all_gpu_variables
from util.segment_utils import create_mask_generator

import cv2
import torchvision.io as io


def save_panoramic_video(kf_interp, run_dir, config, fps=8, panoramic_width=1536, panoramic_height=512, focal=500):
    """Render the scene from a fixed wide-angle panoramic camera.
    
    Uses the accumulated point clouds in kf_interp (kf1 + additional) and
    renders each video frame from a single wide-FoV camera positioned at
    the origin. Saves as video (multiple frames) or image (single frame).
    """
    print("Saving panoramic video...")
    save_path = Path(run_dir) / "panoramic_video.mp4"
    
    num_frames = len(kf_interp.kf1_video_colors)
    device = kf_interp.device
    
    # Create wide-FoV panoramic camera
    K = torch.zeros((1, 4, 4), device=device)
    K[0, 0, 0] = focal  # fx
    K[0, 1, 1] = focal  # fy
    K[0, 0, 2] = panoramic_width / 2   # cx
    K[0, 1, 2] = panoramic_height / 2  # cy
    K[0, 2, 3] = 1
    K[0, 3, 2] = 1
    # R from view_matrix_fixed in WonderWorld: slight -3deg X tilt then xy_negate → rotation_x(+3deg)
    _theta = 3.0 * math.pi / 180.0
    R = torch.tensor([[
        [1.0,            0.0,             0.0],
        [0.0,  math.cos(_theta), -math.sin(_theta)],
        [0.0,  math.sin(_theta),  math.cos(_theta)],
    ]], device=device, dtype=torch.float32)
    T = torch.tensor([[0, -0.00005, 0.0006]], device=device)
    # T = torch.zeros((1,3), device=device)
    panoramic_camera = PerspectiveCameras(
        K=K, R=R, T=T, in_ndc=False,
        image_size=((panoramic_height, panoramic_width),),
        device=device
    )
    
    print(f"Panoramic camera: focal={focal}, resolution={panoramic_width}x{panoramic_height}")
    
    # Render all frames from the panoramic camera
    frames = []
    with torch.no_grad():
        for frame_id in range(num_frames):
            # Gather 3D points and colors for this frame
            if config.get('use_dynamic_point_cloud', False):
                points_3d_aug = torch.cat(
                    [kf_interp.points_3d_video[frame_id],
                     kf_interp.additional_video_points_3d[frame_id]], dim=0)
            else:
                points_3d_aug = torch.cat(
                    [kf_interp.points_3d,
                     kf_interp.additional_points_3d], dim=0)
            
            colors_aug = torch.cat(
                [kf_interp.kf1_video_colors[frame_id],
                 kf_interp.additional_video_colors[frame_id]], dim=0)
            
            # Compute per-point radius
            point_depth = points_3d_aug[..., -1:]
            depth_normalizer = kf_interp.background_hard_depth
            min_ratio = config['point_size_min_ratio']
            radius = config['point_size'] * (
                min_ratio + (1 - min_ratio) * (point_depth.permute([1, 0]) / depth_normalizer))
            radius = radius.clamp(max=config['point_size'] * config['sky_point_size_multiplier'])
            
            raster_settings = PointsRasterizationSettings(
                image_size=(panoramic_height, panoramic_width),
                radius=radius,
                points_per_pixel=8,
            )
            renderer = PointsRenderer(
                rasterizer=PointsRasterizer(
                    cameras=panoramic_camera, raster_settings=raster_settings),
                compositor=SoftmaxImportanceCompositor(
                    background_color=(0, 0, 0), softmax_scale=1.0)
            )
            
            point_cloud = Pointclouds(points=[points_3d_aug], features=[colors_aug])
            images = renderer(point_cloud)  # [1, H, W, 3]
            img = images[0].clamp(0, 1).cpu()  # [H, W, 3]
            frames.append((img * 255).to(torch.uint8))
    
    if len(frames) == 1:
        save_img_path = Path(run_dir) / 'panoramic_single_frame.png'
        ToPILImage()(frames[0].permute(2, 0, 1)).save(save_img_path)
        print(f"Panoramic image saved to {save_img_path}")
        return
    
    video_tensor = torch.stack(frames, dim=0)  # [N, H, W, 3]
    save_video(video_tensor.permute(0, 3, 1, 2), save_path, fps=fps)
    print(f"Panoramic video saved to {save_path} ({len(frames)} frames)")

def evaluate(model):
    fps = model.config["save_fps"]
    save_root = Path(model.run_dir)

    video = (255 * torch.cat(model.images, dim=0)).to(torch.uint8).detach().cpu()
    video_reverse = (255 * torch.cat(model.images[::-1], dim=0)).to(torch.uint8).detach().cpu()

    save_video(video, save_root / "output.mp4", fps=fps)
    save_video(video_reverse, save_root / "output_reverse.mp4", fps=fps)


def evaluate_epoch(model, epoch, vmax=None, save_diffusion_video=True):
    rendered_depth = model.rendered_depths[epoch].clamp(0).cpu().numpy()
    rendered_depth_video = model.rendered_depths_video[epoch].clamp(0).cpu().numpy()
    depth = model.depths[epoch].clamp(0).cpu().numpy()
    depth_video = model.depths_video[epoch].clamp(0).cpu().numpy()
    save_root = Path(model.run_dir) / "images"
    save_root_video = Path(model.run_dir) / "videos"
    save_root_video.mkdir(exist_ok=True, parents=True)
    save_root.mkdir(exist_ok=True, parents=True)
    (save_root / "inpaint_input_image").mkdir(exist_ok=True, parents=True)
    (save_root_video / "inpaint_input_video").mkdir(exist_ok=True, parents=True)
    (save_root / "frames").mkdir(exist_ok=True, parents=True)
    (save_root_video / "frames").mkdir(exist_ok=True, parents=True)
    (save_root / "masks").mkdir(exist_ok=True, parents=True)
    (save_root_video / "masks").mkdir(exist_ok=True, parents=True)
    (save_root / "post_masks").mkdir(exist_ok=True, parents=True)
    (save_root_video / "post_masks").mkdir(exist_ok=True, parents=True)
    (save_root / "rendered_images").mkdir(exist_ok=True, parents=True)
    (save_root / "rendered_depths").mkdir(exist_ok=True, parents=True)
    (save_root_video / "rendered_depths").mkdir(exist_ok=True, parents=True)
    (save_root / "depth").mkdir(exist_ok=True, parents=True)
    (save_root_video / "depth").mkdir(exist_ok=True, parents=True)

    model.inpaint_input_image[epoch].save(save_root / "inpaint_input_image" / f"{epoch}.png")
    ToPILImage()(model.images[epoch][0]).save(save_root / "frames" / f"{epoch}.png")
    ToPILImage()(model.masks[epoch][0]).save(save_root / "masks" / f"{epoch}.png")
    ToPILImage()(model.post_masks[epoch][0]).save(save_root / "post_masks" / f"{epoch}.png")
    ToPILImage()(model.rendered_images[epoch][0]).save(save_root / "rendered_images" / f"{epoch}.png")
    save_depth_map(rendered_depth, save_root / "rendered_depths" / f"{epoch}.png", vmin=0, vmax=vmax)
    save_depth_map(depth, save_root / "depth" / f"{epoch}.png", vmin=0, vmax=vmax, save_clean=True)
    if save_diffusion_video:
        save_video(torch.clamp(model.videos[epoch].cpu() * 255, 0, 255).to(dtype=torch.uint8), save_root_video / "frames" / f"{epoch}.mp4")
        if hasattr(model, "render_output_video"):
            save_video(torch.clamp(model.render_output_video["rendered_image"].cpu() * 255, 0, 255).to(dtype=torch.uint8), save_root_video / "inpaint_input_video" / f"{epoch}.mp4")
            save_video(torch.clamp(model.render_output_video["inpaint_mask"].cpu().repeat(1, 3, 1, 1) * 255, 0, 255).to(dtype=torch.uint8), save_root_video / "inpaint_input_video" / f"mask{epoch}.mp4")
    for i in range(len(depth_video)):
        save_depth_map(depth_video[i], save_root_video / "depth" / f"{epoch}_{i}.png", vmin=0, vmax=vmax, save_clean=True)
        save_depth_map(rendered_depth_video[i], save_root_video / "rendered_depths" / f"{epoch}_{i}.png", vmin=0, vmax=vmax, save_clean=True)
        ToPILImage()(model.masks_video[epoch][i][0]).save(save_root_video / "masks" / f"{epoch}_{i}.png")
        # ToPILImage()(model.post_masks_video[epoch][i][0]).save(save_root / "post_masks" / f"{epoch}_{i}.png")
        
        

    if hasattr(model, "outter_masks"):
        (save_root / "outter_masks").mkdir(exist_ok=True, parents=True)
        ToPILImage()(model.outter_masks[epoch]).save(save_root / "outter_masks" / f"{epoch}.png")
    if epoch == 0:
        with open(Path(model.run_dir) / "config.yaml", "w") as f:
            OmegaConf.save(model.config, f)




def seeding(seed):
    if seed == -1:
        seed = np.random.randint(2 ** 32)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    print(f"running with seed: {seed}.")


def run(config):
    _start_time = time.time()
    _log_dir = Path(config['runs_dir']) / "log"
    _log_dir.mkdir(exist_ok=True, parents=True)
    _log_file = _log_dir / "timing.log"

    ###### ------------------ Load modules ------------------ ######

    # if config['skip_gen']:
    #     kfgen_save_folder = Path(config['runs_dir']) / f"{config['kfgen_load_dt_string']}_kfgen"
    # else:
    #     dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
    #     kfgen_save_folder = Path(config['runs_dir']) / f"{dt_string}_kfgen"
    # kfgen_save_folder = Path(config['runs_dir']) / f"{config['kfgen_load_dt_string']}_kfgen"
    kfgen_save_folder = Path(config['runs_dir']) 
    kfgen_save_folder.mkdir(exist_ok=True, parents=True)
    cutoff_depth = config['fg_depth_range'] + config['depth_shift']
    vmax = cutoff_depth * 2
    inpainting_resolution_gen = config['inpainting_resolution_gen']
    seeding(config["seed"])
    
    segment_processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_coco_swin_large")
    segment_model = OneFormerForUniversalSegmentation.from_pretrained("shi-labs/oneformer_coco_swin_large")

    mask_generator = create_mask_generator()
    # mask_generator.predictor.model = mask_generator.predictor.model.cpu()    

    all_rundir = []
    yaml_data = load_example_yaml(config["example_name"], 'examples/examples.yaml')
    start_keyframe = Image.open(yaml_data['image_filepath']).convert('RGB').resize((512, 512))
    if "video_file_path" in yaml_data.keys():
        config[f'kf1_video_path'] = yaml_data['video_file_path']
        print("use video path:", config[f'kf1_video_path'])
    if os.path.exists(config[f'kf1_video_path']):
        start_keyframe_video = video2images_new(config[f'kf1_video_path'])
        start_keyframe = start_keyframe_video[0]
        frames_len = len(start_keyframe_video)
        selected = list(range(0, frames_len, 1))[:16]
        start_keyframe_video = [start_keyframe_video[frame_i] for frame_i in selected]
        # for i, frame in enumerate(start_keyframe_video):
        #     frame.save(f'debug/rocket/rocket{i}.png')
        # import ipdb; ipdb.set_trace()
    content_prompt, style_prompt, adaptive_negative_prompt, background_prompt, control_text = yaml_data['content_prompt'], yaml_data['style_prompt'], yaml_data['negative_prompt'], yaml_data.get('background', None), yaml_data.get('control_text', None)
    predefined_inpainting_prompt, predefined_video_generation_prompt, predefined_cogvideo_prompt, bg_inpainting_prompt, bg_negative_prompt =  yaml_data.get('inpainting_prompt', None), yaml_data.get('video_generation_prompt', None), yaml_data.get('cogvideo_prompt', None), yaml_data.get('bg_inpainting_prompt', None), yaml_data.get('bg_negative_prompt', None)
    bg_outpainting_prompt = yaml_data.get('bg_outpainting_prompt', None)
    if adaptive_negative_prompt != "":
        adaptive_negative_prompt += ", "
    all_keyframes = [start_keyframe]
    
    if isinstance(control_text, list):
        config['num_scenes'] = len(control_text)
    pt_gen = TextpromptGen(config['runs_dir'], isinstance(control_text, list))
    content_list = content_prompt.split(',')
    scene_name = content_list[0]
    entities = content_list[1:]
    scene_dict = {'scene_name': scene_name, 'entities': entities, 'style': style_prompt, 'background': background_prompt}
    inpainting_prompt = predefined_inpainting_prompt if predefined_inpainting_prompt is not None else style_prompt + ', ' + content_prompt

    inpainter_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
            config["stable_diffusion_checkpoint"],
            safety_checker=None,
            torch_dtype=torch.float16,
            revision="fp16",
        )
    # inpainter_pipeline = AutoPipelineForImage2Image.from_pretrained(config["consistency_model_checkpoint"]).to(config["device"])
    # inpainter_pipeline = StableDiffusionInpaintPipeline.from_pretrained(
    #         config["consistency_model_checkpoint"],
    #         # safety_checker=None,
    #         torch_dtype=torch.float16,
    #         # revision="fp16",
    #     ).to(config["device"])
    
    inpainter_pipeline.scheduler = DPMSolverMultistepScheduler.from_config(inpainter_pipeline.scheduler.config)
    inpainter_pipeline.scheduler = prepare_scheduler(inpainter_pipeline.scheduler)
    
    vae = AutoencoderKL.from_pretrained(config["stable_diffusion_checkpoint"], subfolder="vae")
    vae.enable_tiling()
    vae.enable_slicing()
    inpainter_pipeline.enable_sequential_cpu_offload()
    inpainter_pipeline.enable_model_cpu_offload(device=config["device"])
    # vae = AutoencoderKL.from_pretrained(config["consistency_model_checkpoint"], subfolder="vae").to(config["device"])
    global_video_frame_idx = 9
    clock = -1
    rotation_path = config['rotation_path']
    assert len(rotation_path) >= config['num_scenes'] * config['num_keyframes']

    ###### ------------------ Main loop ------------------ ######
    for i in range(config['num_scenes']):
        if config['use_gpt']:
            control_text_this = control_text[i] if isinstance(control_text, list) else None
            # scene_dict = pt_gen.run_conversation(scene_name=scene_dict['scene_name'], entities=scene_dict['entities'], style=style_prompt, background=scene_dict['background'], control_text=control_text_this)
        # inpainting_prompt = predefined_inpainting_prompt if predefined_inpainting_prompt is not None else pt_gen.generate_prompt(style=style_prompt, entities=scene_dict['entities'], background=scene_dict['background'], scene_name=scene_dict['scene_name'])
        predefined_cogvideo_prompt = predefined_cogvideo_prompt if predefined_cogvideo_prompt is not None else pt_gen.generate_video_prompt(start_keyframe)
        print("cogvideo_prompt: ", predefined_cogvideo_prompt)
        #TODO: Need integration
        # inpainting_prompt = 'Style: Monet painting. Entities: people, cars, buses. Background: sun, horizon, city, '
        # Wrong video_generation_prompt = "A vibrant city avenue, bustling traffic, pedestrians, towering skyscrapers"
        # video_generation_prompt = "Style: Monet painting. Entities: people, cars, buses. Background: busy streets."
        # Wrong inpainting_prompt = 'Style: DSLR 35mm landscape. Entities: There is a person walking towards the camera'
        # Wrong inpainting_prompt = 'Style: DSLR 35mm landscape. Entities: waterfalls'
        for j in range(config['num_keyframes']):

            ###### ------------------ Keyframe (the major part of point clouds) generation ------------------ ######
            if config['skip_gen']:
                kf_gen_dict = torch.load(Path(str(kfgen_save_folder)) / f"s{i:02d}_k{j:01d}_gen_dict.pt", map_location="cpu")
                for key in kf_gen_dict.keys():
                    if isinstance(kf_gen_dict[key], torch.Tensor):
                        kf_gen_dict[key] = kf_gen_dict[key].to(config["device"])
                    elif isinstance(kf_gen_dict[key], list):
                        print(key)
                kf_gen_dict['kf1_camera'], kf_gen_dict['kf2_camera'] = kf_gen_dict['kf1_camera'].to(config["device"]), kf_gen_dict['kf2_camera'].to(config["device"])
                
                kf1_depth, kf2_depth = kf_gen_dict['kf1_depth'], kf_gen_dict['kf2_depth']
                kf1_bg_depth, kf2_bg_depth = kf_gen_dict['kf1_bg_depth'], kf_gen_dict['kf2_bg_depth']
                kf1_depth_video, kf2_depth_video = kf_gen_dict['kf1_depth_video'], kf_gen_dict['kf2_depth_video']
                kf1_bg_depth_video, kf2_bg_depth_video = kf_gen_dict['kf1_bg_depth_video'], kf_gen_dict['kf2_bg_depth_video']
                kf1_image, kf2_image = kf_gen_dict['kf1_image'], kf_gen_dict['kf2_image']
                kf1_bg_image, kf2_bg_image = kf_gen_dict['kf1_bg_image'], kf_gen_dict['kf2_bg_image']
                kf1_video, kf2_video = kf_gen_dict['kf1_video'], kf_gen_dict['kf2_video']
                kf1_bg_video, kf2_bg_video = kf_gen_dict['kf1_bg_video'], kf_gen_dict['kf2_bg_video']
                kf1_dynamic_foreground_mask, kf2_dynamic_foreground_mask = kf_gen_dict['kf1_dynamic_foreground_mask'], kf_gen_dict['kf2_dynamic_foreground_mask']
                kf1_dynamic_background_masks_video, kf2_dynamic_background_masks_video = kf_gen_dict['kf1_dynamic_background_masks_video'], kf_gen_dict['kf2_dynamic_background_masks_video']
                kf1_dynamic_foreground_masks_video, kf2_dynamic_foreground_masks_video = kf_gen_dict['kf1_dynamic_foreground_masks_video'], kf_gen_dict['kf2_dynamic_foreground_masks_video']
                kf1_camera, kf2_camera = kf_gen_dict['kf1_camera'], kf_gen_dict['kf2_camera']
                kf2_mask, kf2_mask_video = kf_gen_dict['kf2_mask'], kf_gen_dict['kf2_mask_video']
                inpainting_prompt, adaptive_negative_prompt = kf_gen_dict['inpainting_prompt'], kf_gen_dict['adaptive_negative_prompt']
                rotation = kf_gen_dict['rotation']


            else:
                # kf_gen_dict = torch.load(Path(config['runs_dir']) / f"{config['kfgen_load_dt_string']}_kfgen_copy"/ f"s{i:02d}_k{j:01d}_gen_dict.pt")
                # kf1_image, kf2_image = kf_gen_dict['kf1_image'], kf_gen_dict['kf2_image']
                # kf1_camera, kf2_camera = kf_gen_dict['kf1_camera'], kf_gen_dict['kf2_camera']
                mask_generator.predictor.model = mask_generator.predictor.model.to(config["device"])
                vae = vae.to(config["device"])
                kf1_video_frames = None
                if OmegaConf.select(config,'use_prerendered_images_videos', default=False) and os.path.exists(config[f'kf{j+1}_video_path']):
                    kf1_video_path = config[f'kf{j+1}_video_path']
                    start_keyframe_video = video2images_new(kf1_video_path)
                    frames_len = len(start_keyframe_video)
                    selected = list(range(0, frames_len, frames_len // 16))[:16]
                    start_keyframe_video = [start_keyframe_video[frame_i] for frame_i in selected]
                # self.kf1_video_frames = rearrange(kf1_video_frames, "b h w c -> b c h w").cuda()
                # kf1_video_colors = rearrange(kf1_video_frames, "b c h w -> b (w h) c").cuda()
                
                rotation = rotation_path[i*config['num_keyframes'] + j]
                regen_negative_prompt = ""
                config['inpainting_resolution_gen'] = inpainting_resolution_gen
                for regen_id in range(config['regenerate_times'] + 1):
                    if regen_id > 0:
                        seeding(-1)
                    depth_model, _, _, _ = load_model(torch.device("cuda"), 'dpt_beit_large_512.pt', 'dpt_beit_large_512', optimize=False)
                    # first keyframe is loaded and estimated depth
                    kf_gen = KeyframeGen(config, inpainter_pipeline, mask_generator, depth_model, vae, rotation, 
                                        start_keyframe, start_keyframe_video, inpainting_prompt, 
                                        regen_negative_prompt + adaptive_negative_prompt, predefined_cogvideo_prompt,
                                        bg_inpainting_prompt, bg_outpainting_prompt, bg_negative_prompt,
                                        segment_model=segment_model, segment_processor=segment_processor).to(config["device"])
                    save_root = Path(kf_gen.run_dir) / "images"
                    kf_idx = 0
                    kf_gen.total_idx = j
                    
                    
                    save_depth_map(kf_gen.depths[kf_idx].detach().cpu().numpy(), save_root / 'kf1_original.png', vmin=0, vmax=vmax)
                    kf_gen.refine_disp_and_dynamics_with_segments(kf_idx, background_depth_cutoff=cutoff_depth)
                    save_depth_map(kf_gen.depths[kf_idx].detach().cpu().numpy(), save_root / 'kf1_processed.png', vmin=0, vmax=vmax)
                    
                    # foreground_mask = kf_gen.segment_foreground(ToPILImage()(kf_gen.images[0].squeeze()))
                    # foreground_mask_video = torch.stack([kf_gen.segment_foreground(x) for x in start_keyframe_video], dim=0)[:, None]
                    foreground_mask = kf_gen.dynamic_foreground_masks[-1]
                    foreground_mask_video = kf_gen.dynamic_foreground_masks_video[-1].unsqueeze(1)
                    # import ipdb; ipdb.set_trace()
                    # ToPILImage()(foreground_mask2.float()) .save('debug/foreground_mask2.png')
                    # import ipdb; ipdb.set_trace()
                    # ToPILImage()(kf_gen.images[-1][0]).save('debug/images_-1.png')
                    inpaint_bg_output_kf1 = kf_gen.inpaint(kf_gen.images[-1], kf_gen.dynamic_foreground_masks[-1][None, None].float().to(kf_gen.images[-1].device), outpainting_bg=True, dilated_factor=6)
                    kf1_bg_image = inpaint_bg_output_kf1['inpainted_image']
                    render_output_video = dict(
                        foreground_mask = foreground_mask_video,
                        inpaint_mask = torch.zeros_like(foreground_mask_video),
                        rendered_image = kf_gen.videos[0].clone(),
                    )
                    if config['have_foreground']:
                        kf1_bg_video = kf_gen.generate_kf_video(ToPILImage()(kf1_bg_image[0]), predefined_cogvideo_prompt, render_output_video=render_output_video, generate_bg=True)
                        kf1_bg_video = torch.stack([ToTensor()(x) for x in kf1_bg_video]).to(kf1_bg_image.device)
                    else:
                        kf1_bg_video = kf_gen.videos[0].clone()
                    with torch.no_grad():
                        kf1_bg_depth, kf1_bg_disp = kf_gen.get_depth(kf1_bg_image)
                        kf1_bg_depth_video, kf1_bg_disp_video = [], []
                        for frame in kf1_bg_video:
                            frame_depth, frame_disp = kf_gen.get_depth(frame[None])
                            kf1_bg_depth_video.append(frame_depth)
                            kf1_bg_disp_video.append(frame_disp)
                    kf1_bg_depth_video = torch.cat(kf1_bg_depth_video, dim=0)
                    kf1_bg_disp_video = torch.cat(kf1_bg_disp_video, dim=0)
                    
                    
                    kf1_bg_depth, kf1_bg_disp, kf1_bg_depth_video, kf1_bg_disp_video = kf_gen.refine_bg_disp_with_segments(kf1_bg_image, kf1_bg_disp, kf1_bg_video, kf1_bg_disp_video, background_depth_cutoff=cutoff_depth)
                    save_depth_map(kf1_bg_depth.detach().cpu().numpy(), save_root / 'kf1_bg_processed.png', vmin=0, vmax=vmax)
                    
                    kf_gen.merge_backgrounds(kf_idx, kf1_bg_image, kf1_bg_depth, kf1_bg_disp, kf1_bg_video, kf1_bg_depth_video, kf1_bg_disp_video)
                    evaluate_epoch(kf_gen, kf_idx, vmax=vmax)
                    kf_idx = 1
                    kf2_video_frames=None
                    if OmegaConf.select(config,'use_prerendered_images_videos', default=False) and OmegaConf.select(config, f'kf{j+2}_video_path', default="None") != "None":
                        kf2_video_path = config[f'kf{j+2}_video_path']
                        kf2_video_frames = video2images_new(kf2_video_path)
                        kf2_video_frames = torch.stack([ToTensor()(x) for x in kf2_video_frames]).to(kf_gen.device)
                    # kf_gen.use_noprompt = True
                    render_output = kf_gen.render(j, kf_idx)
                    render_output_video = kf_gen.render_video(kf_idx)
                    # kf_gen.render_video_pred_cond(kf_idx)
                    
                    # video_numpy = (render_output_video['rendered_image'].permute(0, 2, 3, 1).cpu().numpy() * 255).astype(np.uint8)
                    # output_dir = "debug/reproject_frames"
                    # os.makedirs(output_dir, exist_ok=True)
                    # for i, frame in enumerate(video_numpy):
                    #     frame_image = Image.fromarray(frame)
                    #     frame_image.save(os.path.join(output_dir, f"frame_{i:04d}.png"))
                    # print(f"Saved {video_numpy.shape[0]} frames to the folder '{output_dir}'.")
                    # import cv2; cv2.imwrite("debug/render_output_image_before_inpainting.png", (render_output["rendered_image"][0].permute(1,2,0)*255).cpu().numpy()[:,:,::-1].astype(np.uint8))
                    
                    if config['generate_new_foreground']:
                        rendered_image = render_output["rendered_image_bg"].clone()
                        rendered_image[render_output["foreground_mask"].bool().repeat(1, 3, 1, 1)] = render_output["rendered_image"][render_output["foreground_mask"].bool().repeat(1, 3, 1, 1)]
                        # ToPILImage()(rendered_image[0]).save('debug/kf2_inpainted_image_w_fg.png')
                        # import ipdb; ipdb.set_trace()
                        inpaint_output_fg_kf2 = kf_gen.inpaint(rendered_image, render_output["inpaint_mask_bg"])
                        # ToPILImage()(inpaint_output_fg_kf2['inpainted_image'][0]).save('debug/kf2_inpainted_image_w_fg_after.png')
                        # ToPILImage()(render_output["inpaint_mask_bg"][0]).save('debug/kf2_inpainted_mask_w_fg.png')
                        kf2_foreground_mask = kf_gen.segment_foreground(ToPILImage()(inpaint_output_fg_kf2['inpainted_image'][0].squeeze()))[None, None].to(inpaint_output_fg_kf2['inpainted_image'].device)
                        # ToPILImage()(kf2_foreground_mask.float()).save('debug/kf2_foreground_mask_w_fg.png')
                        inpaint_bg_output_kf2 = kf_gen.inpaint(inpaint_output_fg_kf2['inpainted_image'], kf2_foreground_mask.float(), outpainting_bg=True)
                        # ToPILImage()(inpaint_bg_output_kf2['inpainted_image'][0]).save('debug/kf2_inpainted_image_bg.png')
                    else:
                        # import ipdb; ipdb.set_trace()
                        inpaint_output = kf_gen.inpaint(render_output["rendered_image_bg"], render_output["inpaint_mask_bg"])
                        # import ipdb; ipdb.set_trace()
                        # if j == 0:
                        #     inpaint_output['inpainted_image'][0] = ToTensor()(Image.open('/home/tianfr/data/WonderLive/new_cartoon_kf1_total0_outpainting.png').convert('RGB')).to(inpaint_output['inpainted_image'].device)
                        # elif j == 1:
                        #     inpaint_output['inpainted_image'][0] = ToTensor()(Image.open('/home/tianfr/data/WonderLive/cartoon_total1_kf2_outpainting.png').convert('RGB')).to(inpaint_output['inpainted_image'].device)
                        # ToPILImage()(render_output['rendered_image_bg'][0]).save('debug/kf2_rendereded_image_bg.png')
                        # ToPILImage()(render_output["inpaint_mask_bg"][0]).save('debug/inpaint_mask_bg.png')
                        # ToPILImage()(inpaint_output['inpainted_image'][0]).save('debug/kf2_inpainted_image.png')
                        inpaint_bg_output_kf2 = inpaint_output
                    
                    # correct_prompt = kf_gen.inpainting_prompt
                    # inpaint_bg_output_kf2 = kf_gen.inpaint(inpaint_output['inpainted_image'], render_output["foreground_mask"], outpainting_bg=True)
                    # inpaint_bg_output = inpaint_cv2(inpaint_output['inpainted_image'], render_output["foreground_mask"])
                    # kf_gen.inpainting_prompt = correct_prompt
                    
                    # import cv2; cv2.imwrite("debug/render_output_image_after_inpaintingLCM.png", (inpaint_output["inpainted_image"][0].permute(1,2,0)*255).cpu().numpy()[:,:,::-1].astype(np.uint8))
                    inpainted_image_before_mask = ToPILImage()(inpaint_bg_output_kf2["post_mask"][0])
                    inpainted_image_before_mask.save(save_root/f'inpainted_image_before_mask_{regen_id}.png')
                    inpainted_image_before = ToPILImage()(render_output["rendered_image"][0])
                    inpainted_image_before.save(save_root/f'inpainted_image_before_{regen_id}.png')
                    inpaint_image = ToPILImage()(inpaint_bg_output_kf2['inpainted_image'][0])
                    inpaint_image.save(save_root/f'inpainted_image_{regen_id}.png')
                    reprojection_foreground_mask = ToPILImage()(inpaint_bg_output_kf2["post_mask"][0]) 
                    reprojection_foreground_mask.save(save_root/f'reprojection_foreground_mask_{regen_id}.png')
                    reprojection_image_wo_fg_kf1 = ToPILImage()(((1-inpaint_bg_output_kf1["post_mask"].float().cuda()) * kf_gen.images[-1])[0]) 
                    reprojection_image_wo_fg_kf1.save(save_root/f'reprojection_image_wo_fg_kf1_{regen_id}.png')
                    reprojection_image_wo_fg_kf2 = ToPILImage()(((1-inpaint_bg_output_kf2["post_mask"].float().cuda()) * inpaint_bg_output_kf2['inpainted_image'])[0]) 
                    reprojection_image_wo_fg_kf2.save(save_root/f'reprojection_image_wo_fg_kf2_{regen_id}.png')
                    # inpaint_bg_image = ToPILImage()(inpaint_bg_output[0])
                    inpaint_bg_image_kf1 = ToPILImage()(inpaint_bg_output_kf1['inpainted_image'][0])
                    inpaint_bg_image_kf1.save(save_root/f'inpainted_bg_image_kf1_{regen_id}.png')
                    inpaint_bg_image = ToPILImage()(inpaint_bg_output_kf2['inpainted_image'][0])
                    inpaint_bg_image.save(save_root/f'inpainted_bg_image_kf2_{regen_id}.png')

                    # regenerate_information = {}
                    # if config['enable_regenerate'] and regen_id <= config['regenerate_times'] -1:
                    #     gpt_border, gpt_blur = pt_gen.evaluate_image(ToPILImage()(inpaint_output['inpainted_image'][0]), eval_blur=False)
                    #     regenerate_information['gpt_border'] = gpt_border
                    #     regenerate_information['gpt_blur'] = gpt_blur
                    #     if gpt_border:
                    #         print("chatGPT-4 says the image has border!")
                    #         regen_negative_prompt += "border, "
                    #     if gpt_blur:
                    #         print("chatGPT-4 says the image has blurry effect!")
                    #         regen_negative_prompt += "blur, "
                    #     regenerate = gpt_border
                    # else:
                    #     regenerate = False
                    
                    # with open(save_root / 'regenerate_info.json', 'w') as json_file:
                    #     json.dump(regenerate_information, json_file, indent=4)
                    
                    #!
                    regenerate = False
                    
                    if not regenerate:
                        break
                    # if regen_id == config['regenerate_times'] -1:
                    #     print("Regenerating faild after {} times".format(config['regenerate_times']))
                    #     if gpt_border:
                    #         print("Use crop to solve border problem!")
                    #         config['inpainting_resolution_gen'] = 560
                    #     else:
                    #         break

                    # get memory back
                    depth_model = kf_gen.depth_model.to('cpu')
                    kf_gen.depth_model = None
                    del depth_model
                    empty_cache()
                if config["finetune_decoder_gen"]:
                    ToPILImage()(inpaint_bg_output_kf2["inpainted_image"].detach()[0]).save(save_root / 'kf2_before_ft.png')
                    finetune_decoder(config, kf_gen, render_output, inpaint_bg_output_kf2, config['num_finetune_decoder_steps'], bg_finetuning=True)
                if config['generate_new_foreground']:
                    unique_kf2_foreground_mask = render_output["inpaint_mask_bg"].bool() & kf2_foreground_mask.bool()
                    ToPILImage()(unique_kf2_foreground_mask[0].float()).save('debug/kf2_unique_foreground_mask.png')
                    ToPILImage()(kf2_foreground_mask[0].float()).save('debug/kf2_foreground_mask.png')
                    merged_foreground_mask = render_output["foreground_mask"].bool() | unique_kf2_foreground_mask
                    
                    kf2_partial_video_w_fg = render_output_video['rendered_image_bg'].clone()
                    kf2_partial_video_w_fg[render_output_video['foreground_mask'].bool().repeat(1, 3, 1, 1)] = render_output_video['rendered_image'][render_output_video['foreground_mask'].bool().repeat(1, 3, 1, 1)]
                    render_output_video_kf2_w_fg = dict(
                        foreground_mask = torch.zeros_like(render_output_video['inpaint_mask_bg']),
                        inpaint_mask = render_output_video['inpaint_mask_bg'],
                        rendered_image = kf2_partial_video_w_fg,
                    )
                    
                    kf2_fg_video = kf_gen.generate_kf_video(ToPILImage()(inpaint_output_fg_kf2['inpainted_image'][0]), predefined_cogvideo_prompt, render_output_video=render_output_video_kf2_w_fg)
                    kf2_fg_video = torch.stack([ToTensor()(x) for x in kf2_fg_video]).to(inpaint_output_fg_kf2['inpainted_image'].device)
                    ToPILImage()(kf2_fg_video[0]).save('debug/kf2_fg_video_frame0.png')
                    ToPILImage()(inpaint_bg_output_kf2['inpainted_image'][0]).save('debug/kf2_bg_inpaint.png')
                    ToPILImage()(render_output_video['rendered_image_bg'][0]).save('debug/kf2_bg_before_inpaint.png')
                    kf2_partial_video_wo_fg = kf2_fg_video.clone()
                    kf2_partial_video_wo_fg[~render_output_video['inpaint_mask_bg'].bool().repeat(1, 3, 1, 1)] = render_output_video['rendered_image_bg'][~render_output_video['inpaint_mask_bg'].bool().repeat(1, 3, 1, 1)]
                    foreground_mask_video_kf2 = torch.stack([kf_gen.segment_foreground(ToPILImage()(x)) for x in kf2_fg_video], dim=0)[:, None]
                    unique_foreground_mask_video_kf2 = render_output_video['inpaint_mask_bg'].bool().cpu() & foreground_mask_video_kf2
                    
                    render_output_video_kf2_wo_fg = dict(
                        foreground_mask = unique_foreground_mask_video_kf2,
                        inpaint_mask = torch.zeros_like(unique_foreground_mask_video_kf2),
                        rendered_image = kf2_partial_video_wo_fg,
                        kf2_video_with_fg = kf2_fg_video
                    )
                    
                    kf_gen.update_images_and_masks(inpaint_bg_output_kf2["latent"],
                                                   render_output["inpaint_mask_bg"],
                                                   render_output["rendered_image"], 
                                                   render_output["foreground_mask"], 
                                                   kf2_video_frames, 
                                                   render_output_video['inpaint_mask'], 
                                                   update_kf_video=True, 
                                                   prompt_generator=pt_gen.generate_video_prompt, 
                                                   cogvideo_generation_prompt=predefined_cogvideo_prompt, 
                                                   bg_image=inpaint_bg_output_kf2['inpainted_image'], 
                                                   render_output_video_kf2_wo_fg=render_output_video_kf2_wo_fg
                                                   )

                else:
                    kf_gen.update_images_and_masks(inpaint_output["latent"],
                                                   render_output["inpaint_mask_bg"],
                                                   render_output["rendered_image"],
                                                   render_output["foreground_mask"],
                                                   kf2_video_frames,
                                                   render_output_video['inpaint_mask'],
                                                   update_kf_video=True,
                                                   prompt_generator=pt_gen.generate_video_prompt,
                                                   cogvideo_generation_prompt=predefined_cogvideo_prompt,
                                                   bg_image=inpaint_bg_output_kf2['inpainted_image'])
                # kf_gen.update_images_and_masks(inpaint_bg_output["latent"], render_output["inpaint_mask"], kf2_video_frames, render_output_video['inpaint_mask'], update_kf_video=True, prompt_generator=pt_gen.generate_video_prompt, cogvideo_generation_prompt=predefined_cogvideo_prompt)

                kf2_depth_should_be = render_output['rendered_depth']
                kf2_video_depth_should_be = render_output_video['rendered_depth']
                mask_to_align_depth = ~(render_output["inpaint_mask_512"]>0) & (kf2_depth_should_be < cutoff_depth + kf_gen.kf_delta_t)
                mask_to_align_depth_video = ~(render_output_video["inpaint_mask_512"]>0) & (kf2_video_depth_should_be < cutoff_depth + kf_gen.kf_delta_t)
                mask_to_cutoff_depth = ~(render_output["inpaint_mask_512"]>0) & (kf2_depth_should_be >= cutoff_depth + kf_gen.kf_delta_t)
                mask_to_cutoff_depth_video = ~(render_output_video["inpaint_mask_512"]>0) & (kf2_video_depth_should_be >= cutoff_depth + kf_gen.kf_delta_t)

                # with torch.no_grad():
                    # kf2_before_ft_depth, _ = kf_gen.get_depth(kf_gen.images[kf_idx])  # pix depth under kf2 frame
                if config["finetune_depth_model"]:
                    #TODO: Finetuning with video
                    # import ipdb; ipdb.set_trace()
                    finetune_depth_model(config, kf_gen, kf2_depth_should_be, kf_idx, mask_align=mask_to_align_depth, 
                                        mask_cutoff=mask_to_cutoff_depth, cutoff_depth=cutoff_depth + kf_gen.kf_delta_t)
                    # finetune_video_depth_model(config, kf_gen, kf2_video_depth_should_be, kf_idx, mask_align=mask_to_align_depth_video, 
                    #                     mask_cutoff=mask_to_cutoff_depth_video, cutoff_depth=cutoff_depth + kf_gen.kf_delta_t)
                with torch.no_grad():
                    kf2_ft_depth_original, kf2_ft_disp_original = kf_gen.get_depth(kf_gen.images[kf_idx])
                    kf_gen.depths.append(kf2_ft_depth_original), kf_gen.disparities.append(kf2_ft_disp_original)
                    
                    kf2_bg_image = inpaint_bg_output_kf2['inpainted_image']
                    kf2_bg_depth, kf2_bg_disp = kf_gen.get_depth(kf2_bg_image)
                    kf2_bg_video = kf_gen.bg_videos[kf_idx]
                    kf2_bg_depth_video, kf2_bg_disp_video = kf_gen.get_depth(kf2_bg_video)
                    
                    kf2_bg_depth, kf2_bg_disp, kf2_bg_depth_video, kf2_bg_disp_video = kf_gen.refine_bg_disp_with_segments(kf2_bg_image, kf2_bg_disp, kf2_bg_video, kf2_bg_disp_video, background_depth_cutoff=cutoff_depth + kf_gen.kf_delta_t)
                    
                    kf2_video_ft_depth_original, kf2_video_ft_disp_original = kf_gen.get_depth(kf_gen.videos[kf_idx])
                    kf_gen.depths_video.append(kf2_video_ft_depth_original), kf_gen.disparities_video.append(kf2_video_ft_disp_original)
                    
                    kf2_video_foreground_depth, kf2_video_foreground_disp = kf_gen.get_video_depth(kf_gen.videos[kf_idx])
                    kf_gen.depth_video_w_consistency, kf_gen.disparity_video_w_consistency = kf2_video_foreground_depth, kf2_video_foreground_disp
                # import ipdb; ipdb.set_trace()
                
                
                # ToPILImage()(mask_to_align_depth.detach()[0].float()).save(save_root / 'mask_to_align_depth.png')
                # ToPILImage()(mask_to_cutoff_depth.detach()[0].float()).save(save_root / 'mask_to_cutoff_depth.png')
                # save_depth_map(kf2_before_ft_depth.detach().cpu().numpy(), save_root / 'kf2_before_ft_depth.png', vmin=0, vmax=vmax)
                # save_depth_map(kf2_depth_should_be_processed.detach().cpu().numpy(), save_root / 'kf2_depth_should_be_processed', vmin=0, vmax=vmax)
                # save_depth_map(kf2_depth_should_be.detach().cpu().numpy(), save_root / 'kf2_depth_should_be.png', vmin=0, vmax=vmax)
                # save_depth_map(kf2_ft_depth_original.cpu().numpy(), save_root / 'kf2_ft_depth_original.png', vmin=0, vmax=vmax)
                # get memory back
                depth_model = kf_gen.depth_model.to('cpu')
                kf_gen.depth_model = None
                del depth_model
                empty_cache()

                kf_gen.refine_disp_and_dynamics_with_segments(kf_idx, background_depth_cutoff=cutoff_depth + kf_gen.kf_delta_t)
                save_depth_map(kf_gen.depths[-1].cpu().numpy(), save_root / 'kf2_ft_depth_processed.png', vmin=0, vmax=vmax)
                    
                kf_gen.vae.decoder = deepcopy(kf_gen.decoder_copy)
                evaluate_epoch(kf_gen, kf_idx, vmax=vmax)

                start_keyframe = ToPILImage()(kf_gen.images[1][0])
                start_keyframe_video =[ToPILImage()(x) for x in kf_gen.videos[1]]
                #!
                start_keyframe = start_keyframe_video[-1]
                all_keyframes.append(start_keyframe)
                # import ipdb; ipdb.set_trace()

                kf1_depth, kf2_depth = kf_gen.depths[0].clone().detach(), kf_gen.depths[-1].clone().detach()
                kf1_depth_video, kf2_depth_video = kf_gen.depths_video[0].clone().detach(), kf_gen.depths_video[-1].clone().detach()
                kf1_image, kf2_image = kf_gen.images[0].clone().detach(), kf_gen.images[1].clone().detach()
                kf1_dynamic_foreground_mask, kf2_dynamic_foreground_mask = kf_gen.dynamic_foreground_masks[0].clone().detach(), kf_gen.dynamic_foreground_masks[1].clone().detach()
                kf1_bg_depth, kf2_bg_depth = kf1_bg_depth.clone().detach(), kf2_bg_depth.clone().detach()
                kf1_video, kf2_video = kf_gen.videos[0].clone().detach(), kf_gen.videos[1].clone().detach()
                kf1_camera, kf2_camera = kf_gen.cameras[0].clone(), kf_gen.cameras[1].clone()
                kf2_mask = render_output["inpaint_mask_512"].clone().detach()
                kf2_mask_video = render_output_video["inpaint_mask_512"].clone().detach()
                kf1_dynamic_foreground_masks_video, kf2_dynamic_foreground_masks_video = kf_gen.dynamic_foreground_masks_video[0].clone().detach(), kf_gen.dynamic_foreground_masks_video[1].clone().detach()
                kf1_dynamic_background_masks_video, kf2_dynamic_background_masks_video = kf_gen.dynamic_background_masks_video[0].clone().detach(), kf_gen.dynamic_background_masks_video[1].clone().detach()
                kf_gen_dict = {'kf1_depth': kf1_depth, 'kf2_depth': kf2_depth, 'kf1_image': kf1_image, 'kf2_image': kf2_image,'kf1_bg_depth': kf1_bg_depth, 'kf2_bg_depth': kf2_bg_depth, 'kf1_bg_image': kf1_bg_image, 'kf2_bg_image': kf2_bg_image,
                               'kf1_depth_video': kf1_depth_video, 'kf2_depth_video': kf2_depth_video, 'kf1_video': kf1_video, 'kf2_video': kf2_video, 
                               'kf1_bg_video': kf1_bg_video, 'kf2_bg_video': kf2_bg_video, 'kf1_bg_depth_video': kf1_bg_depth_video, 'kf2_bg_depth_video': kf2_bg_depth_video,
                            'kf1_camera': kf1_camera, 'kf2_camera': kf2_camera, 'kf2_mask': kf2_mask,  'kf2_mask_video': kf2_mask_video,
                            'kf1_dynamic_foreground_mask': kf1_dynamic_foreground_mask, 'kf2_dynamic_foreground_mask': kf2_dynamic_foreground_mask,
                            "kf1_dynamic_foreground_masks_video": kf1_dynamic_foreground_masks_video, "kf2_dynamic_foreground_masks_video": kf2_dynamic_foreground_masks_video,
                            "kf1_dynamic_background_masks_video": kf1_dynamic_background_masks_video, "kf2_dynamic_background_masks_video": kf2_dynamic_background_masks_video,
                            'inpainting_prompt': inpainting_prompt, 'adaptive_negative_prompt': adaptive_negative_prompt, 'rotation': rotation}
                torch.save(kf_gen_dict, kfgen_save_folder / f"s{i:02d}_k{j:01d}_gen_dict.pt")

                if config['skip_interp']:
                    kf_gen = kf_gen.to('cpu')
                    del kf_gen
                    empty_cache()
                    continue
                print(f"Before clearing: {torch.cuda.memory_allocated() / 8 / 1024 /1024} MB")
                clear_all_gpu_variables(kf_gen, depth=1)
                kf_gen = kf_gen.to('cpu')
                del kf_gen
                vae = vae.to('cpu')
                print(f"After clearing: {torch.cuda.memory_allocated() / 8 / 1024 /1024} MB")
                
            # kf_gen.vae = kf_gen.vae.to('cpu')
            # del kf_gen.vae
            # del kf_gen
            ###### ------------------ Keyframe interpolation (completing point clouds and rendering) ------------------ ######
            vae = vae.to(config["device"])
            mask_generator.predictor.model = mask_generator.predictor.model.cpu()
            is_last_scene = i == config['num_scenes'] - 1
            is_last_keyframe = j == config['num_keyframes'] - 1
            try:
                is_next_rotation = rotation_path[i*config['num_keyframes'] + j + 1] != 0
            except IndexError:
                is_next_rotation = False
            try:
                is_previous_rotation = rotation_path[i*config['num_keyframes'] + j - 1] != 0
            except IndexError:
                is_previous_rotation = False
            is_beginning = i == 0 and j == 0
            speed_up = (rotation == 0) and ((is_last_scene and is_last_keyframe) or is_next_rotation)
            speed_down = (rotation == 0) and (is_beginning or is_previous_rotation)
            speed_up = False
            speed_down = False
            total_frames = config["frames"]
            total_frames = total_frames + config["frames"] // 5 if speed_up else total_frames
            total_frames = total_frames + config["frames"] // 5 if speed_down else total_frames
            kf_interp = KeyframeInterp(config, inpainter_pipeline, None, vae, rotation, 
                                   ToPILImage()(kf1_image[0]), [ToPILImage()(x) for x in kf1_video], inpainting_prompt, bg_outpainting_prompt, adaptive_negative_prompt,
                                   kf2_upsample_coef=config['kf2_upsample_coef'], 
                                   kf1_image=kf1_image, kf2_image=kf2_image,
                                   kf1_bg_image=kf1_bg_image, kf2_bg_image=kf2_bg_image,
                                   kf1_video=kf1_video, kf2_video=kf2_video,
                                   kf1_bg_video=kf1_bg_video, kf2_bg_video=kf2_bg_video,
                                   kf1_depth=kf1_depth, kf2_depth=kf2_depth,
                                   kf1_bg_depth=kf1_bg_depth, kf2_bg_depth=kf2_bg_depth,
                                   kf1_depth_video=kf1_depth_video, kf2_depth_video=kf2_depth_video,
                                   kf1_bg_depth_video=kf1_bg_depth_video, kf2_bg_depth_video=kf2_bg_depth_video,
                                   kf1_foreground_mask=kf1_dynamic_foreground_mask, kf2_foreground_mask=kf2_dynamic_foreground_mask,
                                   kf1_dynamic_foreground_masks_video=kf1_dynamic_foreground_masks_video, kf2_dynamic_foreground_masks_video=kf2_dynamic_foreground_masks_video,
                                   kf1_dynamic_background_masks_video=kf1_dynamic_background_masks_video, kf2_dynamic_background_masks_video=kf2_dynamic_background_masks_video,
                                   kf1_camera=kf1_camera, kf2_camera=kf2_camera, kf2_mask=kf2_mask, kf2_mask_video=kf2_mask_video,
                                   speed_up=speed_up, speed_down=speed_down, total_frames=total_frames, keyframe_idx=j+1,
                                   ).to(config["device"])
            
            save_root = Path(kf_interp.run_dir) / "images"
            save_root.mkdir(exist_ok=True, parents=True)
            ToPILImage()(kf1_image[0]).save(save_root / "kf1.png")
            ToPILImage()(kf2_image[0]).save(save_root / "kf2.png")
            

            kf2_camera_upsample, kf2_depth_upsample, kf2_mask_upsample, kf2_foreground_mask_upsample, kf2_image_upsample, kf2_bg_image_upsample, kf2_depth_video_upsample, kf2_bg_depth_video_upsample, kf2_mask_video_upsample, kf2_dynamic_foreground_masks_video_upsample, kf2_video_upsample, kf2_bg_video_upsample = kf_interp.upsample_kf2()
            # import ipdb; ipdb.set_trace()
            # import torchvision.io as io
            # video_frames_new = kf2_depth_video_upsample.permute(0, 2, 3, 1).cpu()  # [F, H, W, C]
            # output_file = "kf2_depth_video_upsample.mp4"
            # fps = 8  # Set the desired frames per second
            # save_depth_map(video_frames_new[10], f"debug/{output_file}_{10}_use_bgdepth.png", vmax=vmax, save_clean=True)
            # io.write_video(output_file, video_frames_new*255, fps=fps, video_codec="libx264", options={"crf": "18"})
            
            kf_interp.update_additional_point_cloud(kf2_depth_upsample, kf2_depth_video_upsample, kf2_image_upsample, kf2_video_upsample, kf2_foreground_mask_upsample, kf2_dynamic_foreground_masks_video_upsample, valid_mask=kf2_mask_upsample, valid_mask_video=kf2_mask_video_upsample, camera=kf2_camera_upsample, points_2d=kf_interp.points_kf2)
            kf2_image_upsample_np = np.array(kf2_image_upsample.cpu()).transpose(0, 2, 3, 1)[...,::-1]
            kf2_mask_upsample_np = np.array(kf2_mask_upsample.cpu()).transpose(0, 2, 3, 1)[...,::-1]
            cv2.imwrite("debug/kf2_image_upsample.png", (kf2_image_upsample_np[0] * 255).astype(np.uint8))
            cv2.imwrite("debug/kf2_mask_upsample.png", (kf2_mask_upsample_np[0] * 255).astype(np.uint8))
            inconsistent_additional_point_index = kf_interp.visibility_check()
            
            inconsistent_additional_point_index_video = kf_interp.visibility_check_video()
            kf2_depth_updated, kf2_depth_video_updated = kf_interp.update_additional_point_depth(inconsistent_additional_point_index, inconsistent_additional_point_index_video, depth=kf2_depth_upsample, mask=kf2_mask_upsample, depth_video=kf2_depth_video_upsample, mask_video=kf2_mask_video_upsample)
            # import ipdb; ipdb.set_trace()
            # save_depth_map(kf2_depth_updated.detach().cpu().numpy(), save_root / 'kf2_depth_updated.png', vmin=0, vmax=vmax)
            kf_interp.reset_additional_point_cloud()
            kf_interp.update_additional_point_cloud(kf2_depth_updated, kf2_depth_video_updated, kf2_image_upsample, kf2_video_upsample, kf2_foreground_mask_upsample, kf2_dynamic_foreground_masks_video_upsample,valid_mask=kf2_mask_upsample, valid_mask_video=kf2_mask_video_upsample, camera=kf2_camera_upsample, points_2d=kf_interp.points_kf2)
            
            kf_interp.depths[0] = F.interpolate(kf2_depth_updated, size=(512, 512), mode="nearest")
            kf_interp.depths_video[0] = F.interpolate(kf2_depth_video_updated, size=(512, 512), mode="nearest")
            # save_depth_map(kf_interp.depths[0].detach().cpu().numpy(), save_root / 'kf2_depth.png', vmin=0, vmax=cutoff_depth*0.95, save_clean=True)
            # save_point_cloud_as_ply(kf_interp.additional_points_3d*500, kf_interp.run_dir / 'kf2_point_cloud.ply', kf_interp.additional_colors)
            # save_point_cloud_as_ply(kf_interp.points_3d *500, kf_interp.run_dir / 'kf1_point_cloud.ply', kf_interp.kf1_colors)
            # evaluate_epoch(kf_interp, 0, vmax=vmax)
            num_fixed_views = 0
            for epoch in tqdm(range(1, total_frames + 1 + num_fixed_views)):
                # global_video_frame_idx = 0
                # fix_view = True if 5<=epoch<=5+num_fixed_views else False
                # fix_view = True if epoch <= total_frames else False
                fix_view = False
                render_output_kf1 = kf_interp.render_kf1(epoch, global_video_frame_idx, fix_view=fix_view)
                # import ipdb; ipdb.set_trace()
                # save_depth_map(render_output_kf1["rendered_depth"].detach().cpu().numpy(), save_root / 'kf2_depth_updated2.png', vmin=0, vmax=cutoff_depth*0.95)
                
                rendered_image_kf1 = np.array(render_output_kf1["rendered_image"][0].cpu()).transpose(1,2,0)[..., ::-1]
                cv2.imwrite(f"debug/kf1_render_wo_inpaint/rendered_image_kf1_{epoch}.png", (rendered_image_kf1 * 255).astype(np.uint8))
                
                inpaint_output = kf_interp.inpaint(render_output_kf1["rendered_image"], render_output_kf1["inpaint_mask"])
                # if j == 1:
                #     import ipdb; ipdb.set_trace()
                    # ToPILImage()(inpaint_output['inpainted_image'][0]).save('debug/kf2_inpainted_image.png')
                    # ToPILImage()(render_output_kf1["inpaint_mask"][0]).save('debug/kf2_inpainted_image.png')
                if config["finetune_decoder_interp"]:
                    finetune_decoder(config, kf_interp, render_output_kf1, inpaint_output, config["num_finetune_decoder_steps_interp"])

                # use latent to get fine-tuned image; center crop if needed; then update image/mask/depth
                # kf_interp.update_images_and_masks(inpaint_output["latent"], render_output_kf1["inpaint_mask"], ori_image=kf_interp.kf2_video_frames[epoch % len(kf_interp.kf2_video_frames)][None, ...].to(kf_interp.images[-1].device))
                kf_interp.update_images_and_masks(inpaint_output["latent"], render_output_kf1["inpaint_mask"], epoch=global_video_frame_idx, foreground_consistency=True, render_output_kf1=render_output_kf1)
                kf_interp.update_additional_point_cloud_from_interpolation(global_video_frame_idx, render_output_kf1["rendered_depth"], kf_interp.images[-1], render_output_kf1['foreground_mask'], append_depth=True)

                # reload decoder
                kf_interp.vae.decoder = deepcopy(kf_interp.decoder_copy)
                with torch.no_grad():
                    kf_interp.images_orig_decoder.append(kf_interp.decode_latents(inpaint_output["latent"]).detach())
                evaluate_epoch(kf_interp, epoch, vmax=cutoff_depth*0.95, save_diffusion_video=False)
                empty_cache()
                global_video_frame_idx += clock
                if global_video_frame_idx == len(kf_interp.kf2_video_frames) - 1:
                    clock = -1
                if global_video_frame_idx == 0:
                    clock = 1
            # kf_interp.images.append(kf1_image)  # so that the last frame is KF1
            if OmegaConf.select(config, 'save_panoramic', default=False):
                save_panoramic_video(kf_interp, kf_interp.run_dir, config, fps=config["save_fps"])
            kf_interp = kf_interp.to('cpu')
            evaluate(kf_interp)
            print(f"Before clearing: {torch.cuda.memory_allocated() / 8 / 1024 /1024} MB")
            clear_all_gpu_variables(kf_interp, depth=1)
            print(f"After clearing: {torch.cuda.memory_allocated() / 8 / 1024 /1024} MB")
            # save_point_cloud_as_ply(torch.cat([kf_interp.points_3d, kf_interp.additional_points_3d], dim=0)*500, kf_interp.run_dir / 'final_point_cloud.ply', torch.cat([kf_interp.kf1_colors, kf_interp.additional_colors], dim=0))

            all_rundir.append(kf_interp.run_dir)
            elapsed = time.time() - _start_time
            _timer_msg = f"[Timer] Finished interp for scene {i}, keyframe {j}. Elapsed time: {elapsed:.2f}s"
            print(_timer_msg)
            with open(_log_file, "a") as _lf:
                _lf.write(_timer_msg + "\n")
            for key in kf_gen_dict.keys():
                if isinstance(kf_gen_dict[key], torch.Tensor):
                    kf_gen_dict[key] = kf_gen_dict[key].cpu()
                elif isinstance(kf_gen_dict[key], list):
                    print(key)
                    kf_gen_dict[key] = [x.cpu() for x in kf_gen_dict[key]]
                else:
                    print(key)
            del kf_interp.kf1_camera, kf_interp.kf2_camera, kf_interp.decoder_copy, kf_interp.current_camera
            del kf_interp.vae, kf_interp
            del kf1_image, kf2_image, kf1_depth, kf2_depth, kf1_bg_image, kf2_bg_image, kf1_bg_depth, kf2_bg_depth, kf1_video, kf2_video, kf1_bg_video, kf2_bg_video
            del kf1_bg_depth_video, kf2_bg_depth_video, kf1_depth_video, kf2_depth_video, kf1_dynamic_foreground_mask, kf2_dynamic_foreground_mask
            del kf1_dynamic_foreground_masks_video, kf2_dynamic_foreground_masks_video, kf1_dynamic_background_masks_video, kf2_dynamic_background_masks_video
            del kf1_camera, kf2_camera, kf2_mask, kf2_mask_video
            vae = vae.to('cpu')
    dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
    save_dir = Path(config['runs_dir']) / f"{dt_string}_merged"
    if not config['skip_interp']:
        merge_frames(all_rundir, save_dir=save_dir, fps=config["save_fps"], is_forward=True, save_depth=False, save_gif=False)
    merge_keyframes(all_keyframes, save_dir=save_dir)
    pt_gen.write_all_content(save_dir=save_dir)



if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="./config/base-config.yaml",
        help="Config path",
    )
    parser.add_argument(
        "--example_config"
    )
    parser.add_argument(
        "--save_panoramic",
        action="store_true",
        default=False,
        help="Enable panoramic camera video/image saving after interpolation",
    )
    args = parser.parse_args()
    base_config = OmegaConf.load(args.base_config)
    example_config = OmegaConf.load(args.example_config)
    config = OmegaConf.merge(base_config, example_config)
    if args.save_panoramic:
        config.save_panoramic = True

    runs_dir = Path(config['runs_dir'])
    if not runs_dir.is_absolute():
        project_root = Path(__file__).parent
        config['runs_dir'] = str(project_root / 'logs' / runs_dir)

    POSTMORTEM = config['debug']
    run(config)
    # if POSTMORTEM:
    #     try:
    #         run(config)
    #     except Exception as e:
    #         print(e)
    #         import ipdb
    #         ipdb.post_mortem()
    # else:
    #     run(config)