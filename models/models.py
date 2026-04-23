import copy
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import skimage
from PIL import Image
from einops import rearrange
from kornia.geometry import PinholeCamera
from pytorch3d.renderer import (
    PerspectiveCameras,
    PointsRasterizationSettings,
    PointsRasterizer,
)
from pytorch3d.renderer.points.compositor import _add_background_color_to_images
from pytorch3d.structures import Pointclouds
from torchvision.transforms import ToTensor, ToPILImage, Resize
from util.midas_utils import dpt_transform, dpt_512_transform
from util.utils import functbl, save_depth_map, load_example_yaml, video2images, video2images_new

from util.segment_utils import refine_disp_with_segments, save_sam_anns
from typing import List, Optional, Tuple, Union
from kornia.morphology import erosion
from kornia.morphology import dilation

from .video_diffusion import inference
from . import video_depth_inference
from omegaconf import OmegaConf
from . import CogVideo
import os
import torchvision.io as io
BG_COLOR=(0, 0, 0)

class PointsRenderer(torch.nn.Module):
    def __init__(self, rasterizer, compositor) -> None:
        super().__init__()
        self.rasterizer = rasterizer
        self.compositor = compositor

    def forward(self, point_clouds, return_z=False, return_bg_mask=False, return_fragment_idx=False, **kwargs) -> torch.Tensor:
        fragments = self.rasterizer(point_clouds, **kwargs)

        r = self.rasterizer.raster_settings.radius

        zbuf = fragments.zbuf.permute(0, 3, 1, 2)
        fragment_idx = fragments.idx.long().permute(0, 3, 1, 2)
        background_mask = fragment_idx[:, 0] < 0  # [B, H, W]
        images = self.compositor(
            fragment_idx,
            zbuf,
            point_clouds.features_packed().permute(1, 0),
            **kwargs,
        )

        # permute so image comes at the end
        images = images.permute(0, 2, 3, 1)

        ret = [images]
        if return_z:
            ret.append(fragments.zbuf)
        if return_bg_mask:
            ret.append(background_mask)
        if return_fragment_idx:
            ret.append(fragments.idx.long())
        
        if len(ret) == 1:
            ret = images
        return ret


class SoftmaxImportanceCompositor(torch.nn.Module):
    """
    Accumulate points using a softmax importance weighted sum.
    """

    def __init__(
        self, background_color: Optional[Union[Tuple, List, torch.Tensor]] = None, softmax_scale=1.0,
    ) -> None:
        super().__init__()
        self.background_color = background_color
        self.scale = softmax_scale

    def forward(self, fragments, zbuf, ptclds, **kwargs) -> torch.Tensor:
        """
        Composite features within a z-buffer using importance sum. Given a z-buffer
        with corresponding features and weights, these values are accumulated
        according to softmax(1/z * scale) to produce a final image.

        Args:
            fragments: int32 Tensor of shape (N, points_per_pixel, image_size, image_size)
                giving the indices of the nearest points at each pixel, sorted in z-order.
                Concretely pointsidx[n, k, y, x] = p means that features[:, p] is the
                feature of the kth closest point (along the z-direction) to pixel (y, x) in
                batch element n. 
            zbuf: float32 Tensor of shape (N, points_per_pixel, image_size,
                image_size) giving the depth value of each point in the z-buffer.
                Value -1 means no points assigned to the pixel.
            pt_clds: Packed feature tensor of shape (C, P) giving the features of each point
                (can use RGB for example).

        Returns:
            images: Tensor of shape (N, C, image_size, image_size)
                giving the accumulated features at each point.
        """
        background_color = kwargs.get("background_color", self.background_color)

        zbuf_processed = zbuf.clone()
        zbuf_processed[zbuf_processed < 0] = - 1e-4
        importance = 1.0 / (zbuf_processed + 1e-6)
        weights = torch.softmax(importance * self.scale, dim=1)

        fragments_flat = fragments.flatten()
        gathered = ptclds[:, fragments_flat]
        gathered_features = gathered.reshape(ptclds.shape[0], fragments.shape[0], fragments.shape[1], fragments.shape[2], fragments.shape[3])
        images = (weights[None, ...] * gathered_features).sum(dim=2).permute(1, 0, 2, 3)

        # images are of shape (N, C, H, W)
        # check for background color & feature size C (C=4 indicates rgba)
        if background_color is not None:
            return _add_background_color_to_images(fragments, images, background_color)
        return images


class FrameSyn(torch.nn.Module):
    def __init__(self, config, inpainter_pipeline, depth_model, vae, rotation,
                 image, video, inpainting_prompt, adaptive_negative_prompt, video_generation_prompt, bg_inpainting_prompt="", bg_outpainting_prompt="", bg_negative_prompt="", depth=None, depth_video=None):
        super().__init__()

        self.device = config["device"]
        self.config = config
        self.background_hard_depth = config['depth_shift'] + config['fg_depth_range']
        self.is_upper_mask_aggressive = False
        self.use_noprompt = False
        self.total_frames = config['frames']

        self.inpainting_prompt = inpainting_prompt
        self.video_generation_prompt = video_generation_prompt
        self.adaptive_negative_prompt = adaptive_negative_prompt
        self.bg_inpainting_prompt = bg_inpainting_prompt
        self.bg_outpainting_prompt = bg_outpainting_prompt
        self.bg_negative_prompt = bg_negative_prompt
        self.inpainting_pipeline = inpainter_pipeline
        self.generate_new_fg = self.config.generate_new_foreground

        # resize image to 512x512
        image = image.resize((512, 512))
        self.image_tensor = ToTensor()(image).unsqueeze(0).to(self.device)
        self.video_tensor = torch.stack([ToTensor()(x) for x in video]).to(self.device)

        self.depth_model = depth_model
        if depth is not None:
            self.depth = depth
            self.disparity = 1 / depth
            self.depth_video = depth_video
            self.disparity_video = 1 / depth_video
        else:
            with torch.no_grad():
                self.depth, self.disparity = self.get_depth(self.image_tensor)
                self.depth_video, self.disparity_video = self.get_depth(self.video_tensor)
                # self.depth_video_w_consistency, self.disparity_video_w_consistency = self.get_video_depth(self.video_tensor)
        self.current_camera = self.get_init_camera()
        if self.config["motion"] == "rotations":
            self.current_camera.rotating = rotation != 0
            self.current_camera.no_rotations_count = 0
            self.current_camera.rotations_count = 0
            self.current_camera.rotating_right = rotation
            self.current_camera.move_dir = torch.tensor([[-config['right_multiplier'], 0.0, -config['forward_speed_multiplier']]], device=self.device)
        elif self.config["motion"] == "predefined":
            intrinsics = np.load(self.config["intrinsics"]).astype(np.float32)
            extrinsics = np.load(self.config["extrinsics"]).astype(np.float32)

            intrinsics = torch.from_numpy(intrinsics).to(self.device)
            extrinsics = torch.from_numpy(extrinsics).to(self.device)

            # Extend intrinsics to 4x4 with zeros and assign 1 to the last row and column as required by the camera class
            Ks = F.pad(intrinsics, (0, 1, 0, 1), value=0)
            Ks[:, 2, 3] = Ks[:, 3, 2] = 1

            Rs, ts = extrinsics[:, :3, :3], extrinsics[:, :3, 3]

            # PerspectiveCameras operate on row-vector matrices while the loaded extrinsics are column-vector matrices
            Rs = Rs.movedim(1, 2)

            self.predefined_cameras = [
                PerspectiveCameras(K=K.unsqueeze(0), R=R.T.unsqueeze(0), T=t.unsqueeze(0), device=self.device)
                for K, R, t in zip(Ks, Rs, ts)
            ]
            self.current_camera = self.predefined_cameras[0]

        self.images = [self.image_tensor]
        self.videos = [self.video_tensor]
        self.inpaint_input_image = [image]
        self.disparities = [self.disparity]
        self.disparities_video = [self.disparity_video]
        self.depths = [self.depth]
        self.depths_video = [self.depth_video]
        self.masks = [torch.ones_like(self.depth)]
        self.masks_video = [torch.ones_like(self.depth_video)]
        self.post_masks = [torch.ones_like(self.depth)]
        self.post_mask_tmp = None
        self.rendered_images = [self.image_tensor]
        self.rendered_videos = [self.video_tensor]
        self.rendered_depths = [self.depth]
        self.rendered_depths_video = [self.depth_video]
        self.dynamic_foreground_masks = []
        self.dynamic_foreground_masks_video = []
        self.dynamic_background_masks_video = []


        self.vae = vae
        self.decoder_copy = copy.deepcopy(self.vae.decoder)

        self.camera_speed = self.config["camera_speed"] if rotation == 0 else self.config["camera_speed"] * self.config["camera_speed_multiplier_rotation"]
        self.cameras = [self.current_camera]

        # create mask for inpainting of the right size, white area around the image in the middle
        self.border_mask = torch.ones(
            (1, 1, self.config["inpainting_resolution"], self.config["inpainting_resolution"])
        ).to(self.device)
        self.border_size = (self.config["inpainting_resolution"] - 512) // 2
        self.border_mask[:, :, self.border_size : -self.border_size, self.border_size : -self.border_size] = 0
        self.border_image = torch.zeros(
            1, 3, self.config["inpainting_resolution"], self.config["inpainting_resolution"]
        ).to(self.device)
        self.images_orig_decoder = [
            Resize((self.config["inpainting_resolution"], self.config["inpainting_resolution"]))(self.image_tensor)
        ]

        x = torch.arange(512).float() + 0.5
        y = torch.arange(512).float() + 0.5
        self.points = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
        self.points = rearrange(self.points, "h w c -> (h w) c").to(self.device)

        self.kf_delta_t = self.camera_speed
        
        self.kf_idx = 0
        self.total_idx = 0

    def get_depth(self, image):
        if self.depth_model is None:
            depth = torch.zeros_like(image[:, 0:1])
            disparity = torch.zeros_like(image[:, 0:1])
            return depth, disparity
        if self.config['depth_model'].lower() == "midas":
            # MiDaS
            disparity = self.depth_model(dpt_transform(image))
            disparity = torch.nn.functional.interpolate(
                disparity.unsqueeze(1),
                size=image.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            disparity = disparity.clip(1e-6, max=None)
            depth = 1 / disparity
        if self.config['depth_model'].lower() == "midas_v3.1":
            img_transformed = dpt_512_transform(image)
            disparity = self.depth_model(img_transformed)
            disparity = torch.nn.functional.interpolate(
                disparity.unsqueeze(1),
                size=image.shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            disparity = disparity.clip(1e-6, max=None)
            depth = 1 / disparity
        elif self.config['depth_model'].lower() == "zoedepth":
            # ZeoDepth
            depth = self.depth_model(image)['metric_depth']
        depth = depth + self.config['depth_shift']
        disparity = 1 / depth
        return depth, disparity
    
    def get_video_depth(self, video):
        video_depth = video_depth_inference.estimate_depth(video)
        video_disparity = 1 / video_depth
        
        return video_depth.cpu().numpy(), video_disparity.cpu().numpy()

    def get_init_camera(self):
        K = torch.zeros((1, 4, 4), device=self.device)
        K[0, 0, 0] = self.config["init_focal_length"]
        K[0, 1, 1] = self.config["init_focal_length"]
        K[0, 0, 2] = 256
        K[0, 1, 2] = 256
        K[0, 2, 3] = 1
        K[0, 3, 2] = 1
        R = torch.eye(3, device=self.device).unsqueeze(0)
        T = torch.zeros((1, 3), device=self.device)
        camera = PerspectiveCameras(K=K, R=R, T=T, in_ndc=False, image_size=((512, 512),), device=self.device)
        return camera

    def finetune_depth_model_step(self, target_depth, inpainted_image, mask_align=None, mask_cutoff=None, cutoff_depth=None):
        next_depth, _ = self.get_depth(inpainted_image.detach().cuda())

        # L1 loss for the mask_align region
        loss_align = F.l1_loss(target_depth.detach(), next_depth, reduction="none")
        if mask_align is not None and torch.any(mask_align):
            mask_align = mask_align.detach()
            loss_align = (loss_align * mask_align)[mask_align > 0].mean()
        else:
            loss_align = torch.zeros(1).to(self.device)

        # Hinge loss for the mask_cutoff region
        if mask_cutoff is not None and cutoff_depth is not None and torch.any(mask_cutoff):
            hinge_loss = (cutoff_depth - next_depth).clamp(min=0)
            hinge_loss = F.l1_loss(hinge_loss, torch.zeros_like(hinge_loss), reduction="none")
            mask_cutoff = mask_cutoff.detach()
            hinge_loss = (hinge_loss * mask_cutoff)[mask_cutoff > 0].mean()
        else:
            hinge_loss = torch.zeros(1).to(self.device)

        total_loss = loss_align + hinge_loss
        if torch.isnan(total_loss):
            raise ValueError("Depth FT loss is NaN")
        # print both losses and total loss
        print(f"(1000x) loss_align: {loss_align.item()*1000:.4f}, hinge_loss: {hinge_loss.item()*1000:.4f}, total_loss: {total_loss.item()*1000:.4f}")

        return total_loss

    def finetune_decoder_step(self, inpainted_image, inpainted_image_latent, rendered_image, inpaint_mask, inpaint_mask_dilated):
        reconstruction = self.decode_latents(inpainted_image_latent)
        new_content_loss = F.mse_loss(inpainted_image * inpaint_mask, reconstruction * inpaint_mask)
        preservation_loss = F.mse_loss(rendered_image * (1 - inpaint_mask_dilated), reconstruction * (1 - inpaint_mask_dilated)) * self.config["preservation_weight"]
        loss = new_content_loss + preservation_loss
        # print(f"(1000x) new_content_loss: {new_content_loss.item()*1000:.4f}, preservation_loss: {preservation_loss.item()*1000:.4f}, total_loss: {loss.item()*1000:.4f}")
        return loss

    @torch.no_grad()
    def inpaint(self, rendered_image, inpaint_mask, fill_mask=None, fill_mode = 'cv2_telea', outpainting_bg = False, dilated_factor = 5):
        # set resolution
        process_width, process_height = self.config["inpainting_resolution"], self.config["inpainting_resolution"]
        if outpainting_bg:
            inpaint_mask = dilation(inpaint_mask, torch.ones(self.config['dilate_mask_decoder_ft']*dilated_factor, self.config['dilate_mask_decoder_ft']*dilated_factor).to(inpaint_mask.device), border_type="constant").bool()
        mask = (inpaint_mask).float()
        rendered_image = (1 - mask) * rendered_image
        # fill in image
        img = (rendered_image[0].cpu().permute([1, 2, 0]).numpy() * 255).astype(np.uint8)
        fill_mask = inpaint_mask if fill_mask is None else fill_mask
        fill_mask_ = (fill_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)

        mask = (inpaint_mask[0, 0].cpu().numpy() * 255).astype(np.uint8)
        for _ in range(3):
            img, _ = functbl[fill_mode](img, fill_mask_)

        # process mask
        if self.config['use_postmask']:
            mask_block_size = 8
            mask_boundary = mask.shape[0] // 2
            mask_upper = skimage.measure.block_reduce(mask[:mask_boundary, :], (mask_block_size, mask_block_size), np.max if self.is_upper_mask_aggressive else np.min)
            mask_upper = mask_upper.repeat(mask_block_size, axis=0).repeat(mask_block_size, axis=1)
            mask_lower = skimage.measure.block_reduce(mask[mask_boundary:, :], (mask_block_size, mask_block_size), np.min)
            mask_lower = mask_lower.repeat(mask_block_size, axis=0).repeat(mask_block_size, axis=1)
            mask = np.concatenate([mask_upper, mask_lower], axis=0)

        init_image = Image.fromarray(img)
        mask_image = Image.fromarray(mask)
        # self.bg_negative_prompt = "people, man, woman, cars, kids, person"
        prompt = self.bg_inpainting_prompt if outpainting_bg or (not self.generate_new_fg) else self.inpainting_prompt
        # prompt = self.bg_inpainting_prompt
        if outpainting_bg:
            prompt = self.bg_inpainting_prompt
        elif self.generate_new_fg:
            prompt = self.inpainting_prompt
        else:
            prompt = self.bg_outpainting_prompt
        
        # init_image.save('debug/inpainting_init_image.png')
        # mask_image.save('debug/inpainting_mask_image.png')
        # ToPILImage()(init_image).save('debug/inpainting_init_image.png')

        inpainted_image_latents = self.inpainting_pipeline(
            prompt='' if self.use_noprompt else prompt,
            negative_prompt=self.bg_negative_prompt + ", " + self.adaptive_negative_prompt + self.config["negative_inpainting_prompt"] if outpainting_bg or (not self.generate_new_fg) else self.adaptive_negative_prompt + self.config["negative_inpainting_prompt"],
            # negative_prompt=self.bg_negative_prompt + ", " + self.adaptive_negative_prompt + self.config["negative_inpainting_prompt"],
            image=init_image,
            mask_image=mask_image,
            num_inference_steps=25,
            guidance_scale=0 if self.use_noprompt else 7.5,
            height=process_height,
            width=process_width,
            output_type='latent',
        ).images

        inpainted_image = self.inpainting_pipeline.vae.decode(inpainted_image_latents / self.inpainting_pipeline.vae.config.scaling_factor, return_dict=False)[0]
        inpainted_image = (inpainted_image / 2 + 0.5).clamp(0, 1).to(torch.float32)
        # ToPILImage()(inpainted_image[0]).save('debug/inpainting_inpainted_image.png')
        
        post_mask = torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).float() * 255
        if not outpainting_bg:
            self.post_mask_tmp = post_mask
            self.inpaint_input_image.append(init_image)

        return {"inpainted_image": inpainted_image, "post_mask":torch.from_numpy(mask).unsqueeze(0).unsqueeze(0).cuda() / 255, "latent": inpainted_image_latents.float()}

    @torch.no_grad()
    def update_images_and_masks(self, 
                                latent, 
                                inpaint_mask, 
                                fg_image=None,
                                fg_mask=None,
                                video=None, 
                                inpaint_mask_video=None, 
                                ori_image=None, 
                                epoch=None, 
                                update_kf_video=False, 
                                prompt_generator=None, 
                                cogvideo_generation_prompt=None, 
                                bg_image=None, 
                                foreground_consistency=False,
                                render_output_kf1=None,
                                render_output_video_kf2_wo_fg=None,):
        decoded_image = self.decode_latents(latent).detach()
        # if foreground_consistency:
        #     foreground_mask = render_output_kf1['foreground_mask'].repeat(1, 3, 1, 1).bool()
        #     decoded_image[foreground_mask] = render_output_kf1["rendered_image"][foreground_mask]
        if ori_image is not None:
            decoded_image = ori_image
        post_mask = inpaint_mask if self.post_mask_tmp is None else self.post_mask_tmp
        # take center crop of 512*512
        if self.config["inpainting_resolution"] > 512:
            raise NotImplementedError("video crop is not implemented!")
            decoded_image = decoded_image[
                :, :, self.border_size : -self.border_size, self.border_size : -self.border_size
            ]
            inpaint_mask = inpaint_mask[
                :, :, self.border_size : -self.border_size, self.border_size : -self.border_size
            ]
            post_mask = post_mask[
                :, :, self.border_size : -self.border_size, self.border_size : -self.border_size
            ]
        else:
            decoded_image = decoded_image
            inpaint_mask = inpaint_mask
            if inpaint_mask_video is not None:
                inpaint_mask_video = inpaint_mask_video
            elif epoch is not None:
                inpaint_mask_video = self.masks_video[-1]
                epoch = epoch % len(inpaint_mask_video)
                inpaint_mask_video[epoch] = inpaint_mask[0]
            post_mask = post_mask
        
        if update_kf_video:
            prompt = cogvideo_generation_prompt if cogvideo_generation_prompt is not None else prompt_generator(ToPILImage()(decoded_image[0]))
            if self.generate_new_fg:
                render_output_video = render_output_video_kf2_wo_fg
            else:
                render_output_video = dict(
                    foreground_mask = torch.zeros_like(self.render_output_video['foreground_mask']),
                    inpaint_mask = self.render_output_video['inpaint_mask'],
                    rendered_image = self.render_output_video['rendered_image_bg']
                )
            video = self.generate_kf_video(ToPILImage()(decoded_image[0]), prompt, 1, render_output_video=render_output_video)
            video = torch.stack([ToTensor()(x) for x in video]).to(self.device)
            video_fg = video.clone()
            video_fg[self.render_output_video['foreground_mask'].repeat(1, 3, 1, 1).bool()] = self.render_output_video['rendered_image'][self.render_output_video['foreground_mask'].repeat(1, 3, 1, 1).bool()]
            if self.generate_new_fg:
                video_fg[render_output_video_kf2_wo_fg["foreground_mask"].repeat(1, 3, 1, 1).bool()] = render_output_video_kf2_wo_fg["kf2_video_with_fg"][render_output_video_kf2_wo_fg["foreground_mask"].repeat(1, 3, 1, 1).bool()]
        if fg_mask is not None:
            decoded_image[fg_mask.repeat(1, 3, 1, 1).bool()] = fg_image[fg_mask.repeat(1, 3, 1, 1).bool()]
        self.images.append(decoded_image)
        self.masks.append(inpaint_mask)
        self.masks_video.append(inpaint_mask_video)
        self.post_masks.append(post_mask)
        if video is not None:
            self.videos.append(video_fg)
            self.bg_videos.append(video)

    def decode_latents(self, latents):
        images = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
        images = (images / 2 + 0.5).clamp(0, 1)

        return images

    def get_next_camera_rotation(self):
        next_camera = copy.deepcopy(self.current_camera)
        
        if next_camera.rotating:
            next_camera.rotating_right = self.current_camera.rotating_right
            theta = torch.tensor(self.config["rotation_range_theta"] * next_camera.rotating_right)
            rotation_matrix = torch.tensor(
                [[torch.cos(theta), 0, torch.sin(theta)], [0, 1, 0], [-torch.sin(theta), 0, torch.cos(theta)]],
                device=self.device,
            )
            next_camera.R[0] = rotation_matrix @ next_camera.R[0]
            # next_camera.T[0] = rotation_matrix @ next_camera.T[0]
            
            if self.current_camera.rotations_count != 0:  # this is for KFInterp only
                theta_current = theta * (self.config['frames'] + 2 - self.current_camera.rotations_count)
                next_camera.move_dir = torch.tensor([-self.config['forward_speed_multiplier'] * torch.sin(theta_current).item(), 0.0, self.config['forward_speed_multiplier'] * torch.cos(theta_current).item()], device=self.device)
                next_camera.rotations_count = self.current_camera.rotations_count + 1
        else:
            if self.current_camera.rotations_count != 0:  # this is for KFInterp only
                v = self.config['forward_speed_multiplier']
                rc = self.current_camera.rotations_count
                k = self.config['camera_speed_multiplier_rotation']
                acceleration_frames = self.config["frames"] // 2
                if self.speed_up and rc <= acceleration_frames:  # will have rotation in next kf; need to speed up in the first 5 frames
                    next_camera.move_dir = torch.tensor([0.0, 0.0, v * (k + (1-k) * (rc/acceleration_frames))], device=self.device)
                elif self.speed_down and rc > self.total_frames - acceleration_frames:  # had rotation in previous kf; need to slow donw in the last 5 frames
                    next_camera.move_dir = torch.tensor([0.0, 0.0, v * (k + (1-k) * ((self.total_frames-rc+1)/acceleration_frames))], device=self.device)
                else:
                    next_camera.move_dir = torch.tensor([0.0, 0.0, v], device=self.device)  # do not change speed

                # random walk                
                theta = torch.tensor(2 * torch.pi * self.current_camera.rotations_count / (self.total_frames + 1))
                next_camera.move_dir[1] = -self.random_walk_scale_vertical * 0.01 * torch.sin(theta).item()

                next_camera.rotations_count = self.current_camera.rotations_count + 1
        # move camera backwards
        # next_camera.rotating_right = 0.5
        # theta = torch.tensor(self.config["rotation_range_theta"] * next_camera.rotating_right)
        # rotation_matrix = torch.tensor(
        #     [[1, 0, 0],
        #     [0, torch.cos(theta), -torch.sin(theta)],
        #     [0, torch.sin(theta), torch.cos(theta)]],
        #     device=self.device,
        # )
        # next_camera.R[0] = rotation_matrix @ next_camera.R[0]
        speed = self.camera_speed
        # next_camera.move_dir[0, 1] = 0.5
        next_camera.T += speed * next_camera.move_dir
        return next_camera
    def generate_kf_video(self, image, prompt, device_id=1, render_output_video=None, generate_bg=False):

        render_output_video = render_output_video if render_output_video is not None else self.render_output_video
        foreground_masks = render_output_video["foreground_mask"]
        inpaint_masks = render_output_video["inpaint_mask"]
        origin_video = render_output_video["rendered_image"].clone()
        # inpaint_masks = dilation(inpaint_masks, torch.ones(self.config['dilate_mask_decoder_ft'], self.config['dilate_mask_decoder_ft']).to('cuda'))
        merged_inpaint_mask = inpaint_masks[0].bool()
        merged_foreground_mask = foreground_masks[0].bool()
        for i in range(1, len(foreground_masks)):
            merged_foreground_mask = merged_foreground_mask | foreground_masks[i].bool()
            merged_inpaint_mask = merged_inpaint_mask | inpaint_masks[i].bool()

        render_output_video["rendered_image"][:][inpaint_masks[:].repeat(1,3,1,1).bool()] = 0
        video_raw = render_output_video["rendered_image"].permute(0, 2, 3, 1).cpu()
        output_file = "input_video_raw.mp4"
        fps = 8  # Set the desired frames per second
        io.write_video(output_file, video_raw*255, fps=fps, video_codec="libx264", options={"crf": "18"})
        
        input_foreground = copy.deepcopy(render_output_video["rendered_image"])
        input_foreground[~foreground_masks.repeat(1, 3, 1, 1).bool()] = 0
        
        
        video_raw = input_foreground.permute(0, 2, 3, 1).cpu()
        output_file = "input_foreground_reprojection.mp4"
        fps = 8  # Set the desired frames per second
        io.write_video(output_file, video_raw*255, fps=fps, video_codec="libx264", options={"crf": "18"})
        #!
        if not generate_bg:
            prompt = self.video_generation_prompt
        else:
            prompt = self.bg_inpainting_prompt
        merged_mask = merged_foreground_mask | merged_inpaint_mask
        merged_mask = dilation(merged_mask[None], torch.ones(self.config['dilate_mask_decoder_ft']*5, self.config['dilate_mask_decoder_ft']*5).to(merged_mask.device), border_type="constant")[0, 0].bool()
        # ToPILImage()(merged_mask.float()).save("debug/"+ f"merged_mask.png")
        
        
        render_output_video["rendered_image"][0] = ToTensor()(image).to(render_output_video["rendered_image"].device)
        render_output_video["rendered_image"][1:, :, merged_mask] = render_output_video["rendered_image"][:1].repeat(len(render_output_video["rendered_image"]), 1, 1, 1)[1:, :, merged_mask]
        render_output_video["merged_mask"] = merged_mask

        video_raw = render_output_video["rendered_image"].permute(0, 2, 3, 1).cpu()
        output_file = "input_video_with_merged_masks.mp4"
        fps = 8  # Set the desired frames per second
        io.write_video(output_file, video_raw*255, fps=fps, video_codec="libx264", options={"crf": "18"})
        # import ipdb; ipdb.set_trace()
        # if self.kf_idx == 0 and self.total_idx == 0 and generate_bg:
        #     video_bg_path = "/home/tianfr/data/WonderLive/cartoon_new_kf0_total0_inpainting.mp4"
        #     video_bg = video2images_new(video_bg_path)
        #     return video_bg
        # elif self.kf_idx == 1 and self.total_idx == 0 and not generate_bg:
        #     video_path = "/home/tianfr/data/WonderLive/new_cartoon_kf1_total0_outpainting.mp4"
        #     video = video_bg = video2images_new(video_path)
        #     return video
        # elif self.kf_idx == 0 and self.total_idx == 1 and generate_bg:
        #     video_bg_path = "/home/tianfr/data/WonderLive/cartoon_kf1_total1_bg_consistent.mp4"
        #     video_bg = video2images_new(video_bg_path)
        #     return video_bg
        # elif self.kf_idx == 1 and self.total_idx == 1 and not generate_bg:
        #     video_path = "/home/tianfr/data/WonderLive/cartoon_total_idx1_kf1.mp4"
        #     video = video2images_new(video_path)
        #     return video
        
        if self.video_generation_model == "cogvideo":
            video = CogVideo.run_inference(self.pretrained_diffusion_model,
                                           image, 
                                           prompt, 
                                           gpu_num=1, 
                                           gpu_no=device_id,
                                           render_output_video=render_output_video,
                                           no_prompt_guidance=generate_bg,
                                           )
            video_frames = torch.stack([ToTensor()(x) for x in video])
            video_frames_new = video_frames.permute(0, 2, 3, 1).cpu()  # [F, H, W, C]
            output_file = "outpainting_video_origin.mp4"
            fps = 8  # Set the desired frames per second
            io.write_video(output_file, video_frames_new*255, fps=fps, video_codec="libx264", options={"crf": "18"})
            print(f"Video saved to {output_file}")
            

            if not generate_bg:
                video_frames[1:, :, ~merged_mask.cpu()] = render_output_video["rendered_image"].cpu()[1:, :, ~merged_mask.cpu()]
                video_frames_new = video_frames.permute(0, 2, 3, 1).cpu()  # [F, H, W, C]
                output_file = "outpainting_video_w_mask_merged.mp4"
                fps = 8  # Set the desired frames per second
                io.write_video(output_file, video_frames_new*255, fps=fps, video_codec="libx264", options={"crf": "18"})
                # import ipdb; ipdb.set_trace()
                # fg_mask, bg_mask = self.segment_video_frame(video_frames)
                # fg_mask = fg_mask[:, None, :, :].repeat(1, 3, 1, 1)
                # render_output_video_wo_fg = copy.deepcopy(self.render_output_video)
                # render_output_video_wo_fg["rendered_image"] = copy.deepcopy(video_frames)
                # render_output_video_wo_fg["rendered_image"][fg_mask] = 0
                # video_bg_raw = render_output_video_wo_fg["rendered_image"].permute(0, 2, 3, 1).cpu()
                video_frames[foreground_masks.repeat(1,3,1,1).bool().cpu()] = render_output_video["rendered_image"][foreground_masks.repeat(1,3,1,1).bool()].cpu()
                video_frames_new = video_frames.permute(0, 2, 3, 1).cpu()  # [F, H, W, C]
                output_file = "outpainting_video_blending.mp4"
                fps = 8  # Set the desired frames per second
                io.write_video(output_file, video_frames_new*255, fps=fps, video_codec="libx264", options={"crf": "18"})
            else:
                # video_frames[~foreground_masks.repeat(1,3,1,1).bool().cpu()] = origin_video[~foreground_masks.repeat(1,3,1,1).bool()].cpu()
                # video_frames[0] = ToTensor()(image)
                video_frames_new = video_frames.permute(0, 2, 3, 1).cpu()  # [F, H, W, C]
                output_file = "outpainting_video_bg_consistent.mp4"
                fps = 8  # Set the desired frames per second
                io.write_video(output_file, video_frames_new*255, fps=fps, video_codec="libx264", options={"crf": "18"})
                
            return [ToPILImage()(x) for x in video_frames]
            video_bg = CogVideo.run_inference(self.pretrained_diffusion_model,
                                image, 
                                "", 
                                gpu_num=1, 
                                gpu_no=device_id,
                                render_output_video=render_output_video_wo_fg if hasattr(self, "render_output_video") else None,
                                no_prompt_guidance = True
                                )
            video_bg = torch.stack([ToTensor()(x) for x in video_bg])
            video_frames[~fg_mask] = video_bg[~fg_mask]
            video_bg = video_bg.permute(0, 2, 3, 1).cpu()  # [F, H, W, C]
            output_file = "outpainting_video_bg_outpainted.mp4"
            fps = 8  # Set the desired frames per second
            io.write_video(output_file, video_bg*255, fps=fps, video_codec="libx264", options={"crf": "18"})
            print(f"Video saved to {output_file}")
            video_fg = copy.deepcopy(video_frames)
            video_fg[~fg_mask] = 0
            video_fg = video_fg.permute(0, 2, 3, 1).cpu()  # [F, H, W, C]
            output_file = "outpainting_video_fg.mp4"
            fps = 8  # Set the desired frames per second
            io.write_video(output_file, video_fg*255, fps=fps, video_codec="libx264", options={"crf": "18"})
            video_frames = video_frames.permute(0, 2, 3, 1).cpu()  # [F, H, W, C]
            output_file = "outpainting_video_merged.mp4"
            fps = 8  # Set the desired frames per second
            io.write_video(output_file, video_frames*255, fps=fps, video_codec="libx264", options={"crf": "18"})
            print(f"Video saved to {output_file}")
            
            import ipdb; ipdb.set_trace()
            
            
            
        else:
            pred_cond_3d = None
            if hasattr(self, "reproj_pred_cond"):
                pred_cond_3d = self.reproj_pred_cond
            video, pred_cond, pred_x0 = inference.run_inference(self.pretrained_diffusion_model,
                                                                image, 
                                                                prompt, 
                                                                pred_cond_3d=pred_cond_3d, 
                                                                gpu_num=1, 
                                                                gpu_no=device_id,
                                                                render_output_video=self.render_output_video if hasattr(self, "render_output_video") else None)
            # video, pred_cond, pred_x0 = inference.run_inference(self.pretrained_diffusion_model,
            #                                                     image, 
            #                                                     prompt, 
            #                                                     pred_cond_3d=pred_cond_3d, 
            #                                                     gpu_num=1, 
            #                                                     gpu_no=device_id)
            self.pred_cond = pred_cond
            self.pred_x0 = pred_x0

        return video


class KeyframeGen(FrameSyn):
    def __init__(self, config, inpainter_pipeline, mask_generator, depth_model, vae, rotation, 
                 image, video, inpainting_prompt, adaptive_negative_prompt="",video_generation_prompt="", 
                 bg_inpainting_prompt="", bg_outpainting_prompt="", bg_negative_prompt="", 
                 segment_model=None, segment_processor=None):
        
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        run_dir_root = Path(config["runs_dir"])
        self.run_dir = run_dir_root / f"Gen-{dt_string}_{inpainting_prompt.replace(' ', '_')[:40]}"
        (self.run_dir / 'images').mkdir(parents=True, exist_ok=True)
        (self.run_dir / 'videos').mkdir(parents=True, exist_ok=True)
        config['rotation_range_theta'] = config['rotation_range']

        if rotation == 0:
            config['forward_speed_multiplier'] = -1.0
            config['right_multiplier'] = 0
            # config['forward_speed_multiplier'] = 0
            # config['right_multiplier'] = -0.6
        else:  # Compute camera movement
            theta = torch.tensor(config['rotation_range_theta'] / (config['frames'] + 1)) * rotation
            sin = torch.sum(torch.stack([torch.sin(i*theta) for i in range(1, config['frames']+2)]))
            cos = torch.sum(torch.stack([torch.cos(i*theta) for i in range(1, config['frames']+2)]))
            config['forward_speed_multiplier'] = -1.0 / (config['frames'] + 1) * cos.item()
            config['right_multiplier'] = -1.0 / (config['frames'] + 1) * sin.item()
        config['inpainting_resolution'] = config['inpainting_resolution_gen']
        
        self.video_generation_model = config["video_generation_model"]
        self.pretrained_diffusion_model = config["pretrained_diffusion_model"]
        if video is not None:
            generated_video = video
        elif 'kf1_video_path' in config.keys() and config[f'kf1_video_path'] != "None":
            kf1_video_path = config[f'kf1_video_path']
            import ipdb; ipdb.set_trace()
            generated_video = video2images_new(kf1_video_path)[:16]
        else:
            if self.video_generation_model == "cogvideo":
                print("************** Use CogVideoX to generate videos! **********************")
                generated_video = self.generate_kf_video(image, video_generation_prompt, device_id=1)
                # generated_video = CogVideo.run_inference(self.pretrained_diffusion_model, image, video_generation_prompt, gpu_num=1, gpu_no=1)
            else:
                print("************** Use DynamiCrafter to generate videos! **********************")
                generated_video, pred_cond, pred_x0 = inference.run_inference(self.pretrained_diffusion_model, 
                                                                            ToTensor()(image).to(config["device"]), 
                                                                            video_generation_prompt, 
                                                                            None, 
                                                                            gpu_num=1, 
                                                                            gpu_no=1)
                self.pred_cond = pred_cond
                self.pred_x0 = pred_x0
                
        #NOTE: Generate keyframe foreground and background image/video
        
        super().__init__(config, inpainter_pipeline, depth_model, vae, rotation,
                         image, generated_video, inpainting_prompt, adaptive_negative_prompt, video_generation_prompt,
                         bg_inpainting_prompt, bg_outpainting_prompt, bg_negative_prompt)
        
        self.mask_generator = mask_generator
        self.segment_model = segment_model
        self.segment_processor = segment_processor
        self.is_upper_mask_aggressive = True
        
    def dynamic_foreground_judgement(self, logits):
        #NOTE: 0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 25: 'umbrella
        return (logits == 0) | (logits == 1) | (logits == 2) | (logits == 3) | \
            (logits == 4) | (logits == 5) | (logits == 6) | (logits == 7) | (logits == 8)
            
    def dynamic_background_judgement(self, logits):
        #NOTE:  90: 'gravel', 99: 'river',  103: 'sea', 113: 'water-other', 119: 'sky-other-merged', 116: 'tree-merged', 124: 'mountain-merged', 125: 'grass-merged', 129: 'building-other-merged', 130: 'rock-merged'
        return (logits == 99) | (logits == 103) | (logits == 113) | \
            (logits == 119) | (logits == 116) | (logits == 129)  | \
            (logits == 124) | (logits == 125) | (logits == 130)
    
    def dynamic_instance_judgement(self, instance_map):
        foreground_instances = []
        for instance in instance_map['segments_info']:
            if instance['label_id'] in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
                foreground_instances.append(instance['id'])
        foreground_mask = torch.zeros_like(instance_map['segmentation'])
        for instance_id in foreground_instances:
            foreground_mask = foreground_mask | (instance_map['segmentation'] == instance_id)
        return foreground_mask, foreground_instances
    def segment_video_frame(self, video, batch_size=2):
        video = [ToPILImage()(x) for x in video]
        pred_semantic_map_video = []
        
        for video_batch in [video[i:i + batch_size] for i in range(0, len(video), batch_size)]:
            
            segmenter_input_video_batch = self.segment_processor(video_batch, ["semantic"]*len(video_batch), return_tensors="pt")
            segmenter_input_video_batch = {name: tensor.to("cuda") for name, tensor in segmenter_input_video_batch.items()}
            segment_output_video_batch = self.segment_model(**segmenter_input_video_batch)
            pred_semantic_map_video_batch = self.segment_processor.post_process_semantic_segmentation(
                        segment_output_video_batch, target_sizes=[video_batch[0].size[::-1]]*len(video_batch))
            pred_semantic_map_video += [x.cpu() for x in pred_semantic_map_video_batch]
        dynamic_foreground_mask_video = torch.stack([self.dynamic_foreground_judgement(x.cpu()) for x in pred_semantic_map_video], dim=0)
        dynamic_background_mask_video = torch.stack([self.dynamic_background_judgement(x.cpu()) for x in pred_semantic_map_video], dim=0)
        
        enlarge_dynamic_foreground_mask_video = dynamic_foreground_mask_video[0]
        enlarge_dynamic_background_mask_video = dynamic_background_mask_video[0]
        for i in range(1, len(dynamic_foreground_mask_video)):
            enlarge_dynamic_foreground_mask_video = enlarge_dynamic_foreground_mask_video | dynamic_foreground_mask_video[i]
            enlarge_dynamic_background_mask_video = enlarge_dynamic_background_mask_video | dynamic_background_mask_video[i]
        enlarge_dynamic_foreground_mask_video = enlarge_dynamic_foreground_mask_video[None].repeat(len(dynamic_foreground_mask_video), 1, 1)
        enlarge_dynamic_background_mask_video = enlarge_dynamic_background_mask_video[None].repeat(len(dynamic_background_mask_video), 1, 1)
        return enlarge_dynamic_foreground_mask_video, enlarge_dynamic_background_mask_video
        # return dynamic_foreground_mask_video, dynamic_background_mask_video
    
    @torch.no_grad()
    def segment_foreground(self, image):
        # image = ToPILImage()(image.squeeze())
        segmenter_input = self.segment_processor(image, ["semantic"], return_tensors="pt")
        segmenter_input = {name: tensor.to("cuda") for name, tensor in segmenter_input.items()}
        segment_output = self.segment_model(**segmenter_input)
        pred_semantic_map = self.segment_processor.post_process_semantic_segmentation(
                        segment_output, target_sizes=[image.size[::-1]])[0]
        dynamic_foreground_mask = self.dynamic_foreground_judgement(pred_semantic_map.cpu())
        
        dynamic_foreground_mask = dilation(dynamic_foreground_mask[None, None], torch.ones(self.config['dilate_mask_decoder_ft'], self.config['dilate_mask_decoder_ft']), border_type="constant")[0,0].bool()
        
        return dynamic_foreground_mask
    
    @torch.no_grad()
    def refine_disp_and_dynamics_with_segments(self, kf_idx, background_depth_cutoff=1./7., batch_size=2, first=False):
        image = ToPILImage()(self.images[kf_idx].squeeze())
        # import ipdb; ipdb.set_trace()
        # image_new = Image.open("/shared_data/p_vidalr/fengruitian/DynamicVoyager_eval/be-your-outpainter/frames_car_3d_consistent/right/right_0015.png")
        # image_new_tensor = ToTensor()(image_new)
        # view_mask = Image.open("/home/tianfr/data/WonderLive/output/cogvideo_outpainting/06_city_monet_1_consistency/Gen-12-02_18-53-34_Style:_Monet_painting._cars_moving_on_th/images/masks/1.png").convert("L")
        # mask_tensor = ToTensor()(view_mask)  # [1, H, W], values in [0,1]
        # view_mask = (mask_tensor > 0.5).float()  # binarize
        video = [ToPILImage()(x) for x in self.videos[kf_idx]]
        segmenter_input = self.segment_processor(image, ["semantic"], return_tensors="pt")
        segmenter_input = {name: tensor.to("cuda") for name, tensor in segmenter_input.items()}
        segment_output = self.segment_model(**segmenter_input)
        pred_semantic_map = self.segment_processor.post_process_semantic_segmentation(
                        segment_output, target_sizes=[image.size[::-1]])[0]
        # instance_sgementer_input = self.segment_processor(image, ["instance"], return_tensors="pt")
        # instance_sgementer_input = {name: tensor.to("cuda") for name, tensor in segmenter_input.items()}
        # instance_segment_output = self.segment_model(**segmenter_input)
        # pred_instance_map = self.segment_processor.post_process_instance_segmentation(
        #                 instance_segment_output, target_sizes=[image.size[::-1]])[0]
        
        sky_mask = pred_semantic_map.cpu() == 119
        sky_mask = erosion(sky_mask.float()[None, None], 
                    kernel=torch.ones(self.config['sky_erode_kernel_size'], self.config['sky_erode_kernel_size'])
                    ).squeeze() > 0.5
        sky_mask = sky_mask.cpu()
        ToPILImage()(sky_mask.float()).save(self.run_dir / 'images' / f"kf{kf_idx+1}_sky_mask.png")
        
        #NOTE: 0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9
        dynamic_foreground_mask = self.dynamic_foreground_judgement(pred_semantic_map.cpu())
        # import ipdb; ipdb.set_trace()
        dynamic_foreground_mask = dilation(dynamic_foreground_mask[None, None], torch.ones(self.config['dilate_mask_decoder_ft'], self.config['dilate_mask_decoder_ft']), border_type="constant")[0,0].bool()
        # ToPILImage()(dynamic_foreground_mask.float()).save("debug/"+ f"kf{kf_idx+1}_dynamic_foreground_mask.png")
        # dynamic_foreground_instance_mask, foreground_ids = self.dynamic_instance_judgement(pred_instance_map)

        #NOTE:  90: 'gravel', 99: 'river',  103: 'sea', 113: 'water-other', 119: 'sky-other-merged', 116: 'tree-merged', 124: 'mountain-merged', 125: 'grass-merged', 129: 'building-other-merged', 130: 'rock-merged'
        dynamic_background_mask = self.dynamic_background_judgement(pred_semantic_map.cpu())
        # if kf_idx == 0 and first:
        #     mask_path = "/home/tianfr/data/WonderLive/examples/videos/cat4d_input_mask.png"
        # else:
        #     mask_path = "/home/tianfr/data/WonderLive/output/cogvideo_outpainting/panda_cat4d/Gen-24-02_16-37-19_Natural_scene._Lake,_Clouds,_Trees./images/kf2_dynamic_foreground_mask.png"
        # mask = Image.open(mask_path).convert("RGB")
        # mask = mask.resize(image.size)
        # mask = ToTensor()(mask).squeeze()
        # mask[mask != 0] = 1
        # mask = mask[0].bool()
        # dynamic_foreground_mask = mask
        # dynamic_background_mask = (~mask) & dynamic_background_mask
        # sky_mask = (~mask) & sky_mask
        if kf_idx == 0 and self.config['mask_path'] != "None":
            mask = Image.open(os.path.join(self.config['mask_path'], "0.png"))
            mask = mask.resize(image.size)
            mask = ToTensor()(mask).squeeze()
            mask[mask != 0] = 1
            mask = mask[0].bool()
            dynamic_foreground_mask = mask
            dynamic_background_mask = (~mask) & dynamic_background_mask
            sky_mask = (~mask) & sky_mask
        if not self.config['have_foreground']:
            dynamic_foreground_mask = torch.zeros_like(dynamic_foreground_mask).bool()
        # dilated_dynamic_background_mask = 1 - dynamic_background_mask.int()
        # dilated_dynamic_background_mask = dilation(dilated_dynamic_background_mask[None, None], torch.ones(self.config['dilate_mask_decoder_ft'], self.config['dilate_mask_decoder_ft']))
        # dynamic_background_mask = (1 - dilated_dynamic_background_mask[0,0]).bool()
        
        ToPILImage()(dynamic_foreground_mask.float()).save(self.run_dir / 'images' / f"kf{kf_idx+1}_dynamic_foreground_mask.png")
        # ToPILImage()(dynamic_foreground_instance_mask.float()).save(self.run_dir / 'images' / f"kf{kf_idx+1}_dynamic_foreground_instance_mask.png")
        ToPILImage()(dynamic_background_mask.float()).save(self.run_dir / 'images' / f"kf{kf_idx+1}_dynamic_background_mask.png")
        
        pred_semantic_map_video = []
        
        for video_batch in [video[i:i + batch_size] for i in range(0, len(video), batch_size)]:
            
            segmenter_input_video_batch = self.segment_processor(video_batch, ["semantic"]*len(video_batch), return_tensors="pt")
            segmenter_input_video_batch = {name: tensor.to("cuda") for name, tensor in segmenter_input_video_batch.items()}
            segment_output_video_batch = self.segment_model(**segmenter_input_video_batch)
            pred_semantic_map_video_batch = self.segment_processor.post_process_semantic_segmentation(
                        segment_output_video_batch, target_sizes=[image.size[::-1]]*len(video_batch))
            pred_semantic_map_video += [x.cpu() for x in pred_semantic_map_video_batch]
        
        # pred_semantic_map_video = torch.cat(pred_semantic_map_video, dim=0)

        sky_mask_video = torch.stack([x.cpu()==119 for x in pred_semantic_map_video],dim=0)
        sky_mask_video = erosion(sky_mask_video.float()[:, None], 
                           kernel=torch.ones(self.config['sky_erode_kernel_size'], self.config['sky_erode_kernel_size'])
                           ).squeeze() > 0.5
        sky_mask_video = sky_mask_video.cpu()
        
        dynamic_foreground_mask_video = torch.stack([self.dynamic_foreground_judgement(x.cpu()) for x in pred_semantic_map_video],dim=0)
        dynamic_background_mask_video = torch.stack([self.dynamic_background_judgement(x.cpu()) for x in pred_semantic_map_video],dim=0)
        if kf_idx == 0 and self.config['mask_path'] != "None":
            dynamic_foreground_mask_video = []
            for i in range(len(pred_semantic_map_video)):
                mask = Image.open(os.path.join(self.config['mask_path'], f"{i}.png"))
                # mask = mask.resize(image.size)
                mask = ToTensor()(mask).squeeze()
                mask[mask != 0] = 1
                mask = mask[0].bool()
                dynamic_foreground_mask_video.append(mask)
            dynamic_foreground_mask_video = torch.stack(dynamic_foreground_mask_video, dim=0)
            dynamic_background_mask_video = (~dynamic_foreground_mask_video) & dynamic_background_mask_video
            sky_mask_video = ~dynamic_foreground_mask_video & sky_mask_video
        # if self.config['mask_path'] != "None":
        #     sky_mask_video = torch.zeros_like(sky_mask_video).bool()
        #     sky_mask = torch.zeros_like(sky_mask).bool()
            
        if not self.config['have_foreground']:
            dynamic_foreground_mask_video = torch.zeros_like(dynamic_foreground_mask_video).bool()
        # dilated_dynamic_background_mask_video = 1 - dynamic_background_mask_video.int()
        # dilated_dynamic_background_mask_video = dilation(dilated_dynamic_background_mask_video[:, None], torch.ones(self.config['dilate_mask_decoder_ft'], self.config['dilate_mask_decoder_ft']))
        # dynamic_background_mask_video = (1 - dilated_dynamic_background_mask_video[:,0]).bool()
        
        os.makedirs(self.run_dir / 'videos' / "dynamic_foreground_mask", exist_ok=True)
        os.makedirs(self.run_dir / 'videos' / "dynamic_background_mask", exist_ok=True)
        for i, sky_mask_i in enumerate(sky_mask_video):
            ToPILImage()(sky_mask_i.float()).save(self.run_dir / 'videos' / f"kf{kf_idx+1}_sky_mask_video_frame{i}.png")
            ToPILImage()(dynamic_foreground_mask_video[i].float()).save(self.run_dir / 'videos' / "dynamic_foreground_mask" / f"kf{kf_idx+1}_dynamic_foreground_mask_video_frame{i}.png")
            ToPILImage()(dynamic_background_mask_video[i].float()).save(self.run_dir / 'videos' / "dynamic_background_mask" / f"kf{kf_idx+1}_dynamic_background_mask_video_frame{i}.png")
            # ToPILImage()((dynamic_foreground_mask_video[i][None] * ToTensor()(video[i])).float()).save(self.run_dir / 'videos' / "dynamic_foreground_mask" / f"kf{kf_idx+1}_dynamic_foreground_frame{i}.png")
        # import ipdb; ipdb.set_trace()


        image_np = np.array(image)
        video_np = np.array([np.array(x) for x in video])
        
        masks = self.mask_generator.generate(image_np)
        masks_video = [self.mask_generator.generate(x) for x in video_np]
        sorted_mask = sorted(masks, key=(lambda x: x['area']), reverse=False)
        sorted_mask_video = [sorted(frame, key=(lambda x: x['area']), reverse=False) for frame in masks_video]
        min_mask_area = 30
        sorted_mask = [m for m in sorted_mask if m['area'] > min_mask_area]
        sorted_mask_video = [[m for m in sorted_mask_frame if m['area'] > min_mask_area ] for sorted_mask_frame  in sorted_mask_video]

        # Remove foreground depth cutoff
        # sorted_mask_foreground = copy.deepcopy(sorted_mask)
        # sorted_mask_foreground = [m for m in sorted_mask_foreground if (m['segmentation'] & dynamic_foreground_mask.numpy()).any()]
        # sorted_mask = [m for m in sorted_mask if not (m['segmentation'] & dynamic_foreground_mask.numpy()).any()]
        # sorted_mask_video_foreground = copy.deepcopy(sorted_mask_video)
        # sorted_mask_video_foreground = [[m for m in sorted_mask_video_foreground[i] if (m['segmentation'] & dynamic_foreground_mask_video[i].numpy()).any()] for i in range(len(sorted_mask_video_foreground))]
        # sorted_mask_video = [[m for m in sorted_mask_video[i] if not (m['segmentation'] & dynamic_foreground_mask_video[i].numpy()).any()] for i in range(len(sorted_mask_video))]

        save_sam_anns(masks, self.run_dir / 'images' / f"SAM_kf{kf_idx+1}.png")
        # save_sam_anns(sorted_mask, self.run_dir / 'images' / f"SAM_kf{kf_idx+1}_filtered.png")
        for i, mask_frame in enumerate(masks_video):
            save_sam_anns(mask_frame, self.run_dir / 'videos' / f"SAM_kf{kf_idx+1}_video{i}.png")
        # for i, mask_frame in enumerate(sorted_mask_video):
        #     save_sam_anns(mask_frame, self.run_dir / 'videos' / f"SAM_kf{kf_idx+1}_video{i}_filtered.png")
        disparity_np = self.disparities[kf_idx].squeeze().cpu().numpy()
        disparity_video_np = self.disparities_video[kf_idx].squeeze().cpu().numpy()
        # if self.config['mask_path'] != "None":
        #     sky_mask = sky_mask & (disparity_np < 1 / 0.01)
        #     sky_mask_video = sky_mask_video & (disparity_video_np < 1 / 0.01)
        keep_threshold_ratio = 0.3
        refined_disparity = refine_disp_with_segments(disparity_np, sorted_mask, dynamic_foreground_mask.numpy(), keep_threshold=1 / background_depth_cutoff * keep_threshold_ratio)
        refined_disparity_video = []
        for i, disparity_frame_np, sorted_mask_frame in zip(range(len(disparity_video_np)), disparity_video_np, sorted_mask_video):
            refined_disparity_frame = refine_disp_with_segments(disparity_frame_np, sorted_mask_frame, dynamic_foreground_mask_video[i].numpy(), keep_threshold=1 / background_depth_cutoff * keep_threshold_ratio)
            save_depth_map(1/refined_disparity_frame, self.run_dir / 'videos' / f"kf{kf_idx+1}_p1_SAM_video{i}.png", vmax=self.config['sky_hard_depth'])
            refined_disparity_video.append(refined_disparity_frame)
        refined_disparity_video = np.array(refined_disparity_video)
        
        save_depth_map(1/refined_disparity, self.run_dir / 'images' / f"kf{kf_idx+1}_p1_SAM.png", vmax=self.config['sky_hard_depth'])

        sky_hard_disp = 1. / self.config['sky_hard_depth']
        bg_hard_disp = 1. / (background_depth_cutoff)
        refined_disparity[sky_mask] = sky_hard_disp
        save_depth_map(1/refined_disparity, self.run_dir / 'images' / f"kf{kf_idx+1}_p2_sky.png", vmax=self.config['sky_hard_depth'])
        
        for i in range(len(refined_disparity_video)):
            refined_disparity_video[i,sky_mask_video[i]] = sky_hard_disp
            save_depth_map(1/refined_disparity_video[i], self.run_dir / 'videos' / f"kf{kf_idx+1}_p2_sky_video{i}.png", vmax=self.config['sky_hard_depth'])
            

        background_cutoff = 1./background_depth_cutoff
        background_mask = refined_disparity < background_cutoff
        background_but_not_sky_mask = np.logical_and(background_mask, np.logical_not(sky_mask.numpy()))
        refined_disparity[background_but_not_sky_mask] = bg_hard_disp
        background_but_not_sky_mask_video = []
        save_depth_map(1/refined_disparity, self.run_dir / 'images' / f"kf{kf_idx+1}_p3_cutoff.png", vmax=self.config['sky_hard_depth'])
        for i in range(len(refined_disparity_video)):
            background_mask_frame = refined_disparity_video[i] < background_cutoff
            background_but_not_sky_mask_frame = np.logical_and(background_mask_frame, np.logical_not(sky_mask_video[i].numpy()))
            background_but_not_sky_mask_video.append(background_but_not_sky_mask_frame)
            refined_disparity_video[i, background_but_not_sky_mask_frame] = bg_hard_disp
            save_depth_map(1/refined_disparity_video[i], self.run_dir / 'videos' / f"kf{kf_idx+1}_p3_cutoff_video{i}.png", vmax=self.config['sky_hard_depth'])
        background_but_not_sky_mask_video = np.stack(background_but_not_sky_mask_video, axis=0)

        refined_disparity = refine_disp_with_segments(refined_disparity, sorted_mask, dynamic_foreground_mask.numpy(), keep_threshold=1 / background_depth_cutoff * keep_threshold_ratio)
        refined_disparity_video_new = []
        for i, refined_disparity_frame, sorted_mask_frame in zip(range(len(refined_disparity_video)), refined_disparity_video, sorted_mask_video):
            refined_disparity_frame = refine_disp_with_segments(refined_disparity_frame, sorted_mask_frame, dynamic_foreground_mask.numpy(), keep_threshold=1 / background_depth_cutoff * keep_threshold_ratio)
            save_depth_map(1/refined_disparity_frame, self.run_dir / 'videos' / f"kf{kf_idx+1}_p4_SAM_video{i}.png", vmax=self.config['sky_hard_depth'])
            refined_disparity_video_new.append(refined_disparity_frame)
        refined_disparity_video = np.array(refined_disparity_video_new)
        save_depth_map(1/refined_disparity, self.run_dir / 'images' / f"kf{kf_idx+1}_p4_SAM.png", vmax=self.config['sky_hard_depth'])
        
        #Step 5: temporal smooth
        smooth_disparity_mask  = np.zeros_like(dynamic_background_mask.numpy())
        smooth_disparity = np.zeros_like(disparity_np)
        for i in range(len(dynamic_background_mask_video)):
            smooth_disparity_mask = smooth_disparity_mask + (np.logical_not(sky_mask_video[i].numpy()) & dynamic_background_mask_video[i].numpy()).astype(int)
            smooth_disparity = smooth_disparity + refined_disparity_video[i] * (np.logical_not(sky_mask_video[i].numpy()) & dynamic_background_mask_video[i].numpy()).astype(int)
        # smooth_disparity_mask = smooth_disparity_mask + (np.logical_not(sky_mask.numpy()) & dynamic_background_mask.numpy()).astype(int)
        # smooth_disparity = smooth_disparity + refined_disparity * (np.logical_not(sky_mask.numpy()) & dynamic_background_mask.numpy()).astype(int)
        
        smooth_disparity_mask[smooth_disparity_mask == 0] = 1
        smooth_disparity = smooth_disparity / smooth_disparity_mask
        for i in range(len(dynamic_background_mask_video)):
            refined_disparity_video[i, (np.logical_not(sky_mask_video[i].numpy()) & dynamic_background_mask_video[i].numpy())] = smooth_disparity[(np.logical_not(sky_mask_video[i].numpy()) & dynamic_background_mask_video[i].numpy())]
        # refined_disparity[(np.logical_not(sky_mask.numpy()) & dynamic_background_mask.numpy())] = smooth_disparity[(np.logical_not(sky_mask.numpy()) & dynamic_background_mask.numpy())]
        
        # Step 6: foreground depth enforcing.
        if self.config['have_foreground']:
            if (dynamic_foreground_mask.numpy() == True).any():
                refined_disparity[dynamic_foreground_mask.numpy()] = np.percentile(refined_disparity[dynamic_foreground_mask.numpy()], 10)
            for i in range(len(dynamic_foreground_mask_video)):
                if (dynamic_foreground_mask_video[i].numpy() == True).any():
                    if self.config['mask_path'] != "None":
                        refined_disparity_video[i, dynamic_foreground_mask_video[i].numpy()] = np.percentile(refined_disparity_video[0, dynamic_foreground_mask_video[i].numpy()], 90)
                    else:
                        refined_disparity_video[i, dynamic_foreground_mask_video[i].numpy()] = np.percentile(refined_disparity_video[i, dynamic_foreground_mask_video[i].numpy()], 90)
        
        # if (dynamic_foreground_mask.numpy() == True).any():
        #     refined_disparity[dynamic_foreground_mask.numpy()] = self.disparity_video_w_consistency[0][dynamic_foreground_mask.numpy()]
        # for i in range(len(dynamic_foreground_mask_video)):
        #     if (dynamic_foreground_mask_video[i].numpy() == True).any():
        #         refined_disparity_video[i, dynamic_foreground_mask_video[i].numpy()] = self.disparity_video_w_consistency[i, dynamic_foreground_mask_video[i].numpy()]
        
        # Video depth alignment
        # if dynamic_foreground_mask_video.numpy().any():
        #     depth_max = refined_disparity_video[dynamic_foreground_mask_video.numpy()].max()
        #     depth_min = refined_disparity_video[dynamic_foreground_mask_video.numpy()].min()
        #     consistent_depth_max = self.disparity_video_w_consistency[dynamic_foreground_mask_video.numpy()].max()
        #     consistent_depth_min = self.disparity_video_w_consistency[dynamic_foreground_mask_video.numpy()].min()
            
        #     self.disparity_video_w_consistency = (self.disparity_video_w_consistency - consistent_depth_min) / (consistent_depth_max - consistent_depth_min) * (depth_max - depth_min) + depth_min
            
        #     if (dynamic_foreground_mask.numpy() == True).any():
        #         refined_disparity[dynamic_foreground_mask.numpy()] = self.disparity_video_w_consistency[0][dynamic_foreground_mask.numpy()]
        #     for i in range(len(dynamic_foreground_mask_video)):
        #         if (dynamic_foreground_mask_video[i].numpy() == True).any():
        #             refined_disparity_video[i, dynamic_foreground_mask_video[i].numpy()] = self.disparity_video_w_consistency[i, dynamic_foreground_mask_video[i].numpy()]

        
        refined_depth = 1 / refined_disparity
        refined_depth_video = 1 / refined_disparity_video

        refined_depth = torch.from_numpy(refined_depth).to(self.device)
        refined_depth_video = torch.from_numpy(refined_depth_video).to(self.device)
        refined_disparity = torch.from_numpy(refined_disparity).to(self.device)
        refined_disparity_video = torch.from_numpy(refined_disparity_video).to(self.device)

        self.depths[kf_idx][0, 0] = refined_depth
        self.depths_video[kf_idx][:, 0]= refined_depth_video
        self.disparities[kf_idx][0, 0] = refined_disparity
        self.disparities_video[kf_idx][:, 0] = refined_disparity_video
        
        self.dynamic_foreground_masks.append(dynamic_foreground_mask)
        
        self.dynamic_foreground_masks_video.append(dynamic_foreground_mask_video)
        self.dynamic_background_masks_video.append(dynamic_background_mask_video)

        return refined_depth, refined_depth_video, refined_disparity, refined_disparity_video, sky_mask, sky_mask_video, background_but_not_sky_mask, background_but_not_sky_mask_video, dynamic_foreground_mask, dynamic_foreground_mask_video, dynamic_background_mask, dynamic_background_mask_video
    
    def merge_backgrounds(self, kf_idx, image_bg, depth_bg, disparity_bg, video_bg, depth_video_bg, disparity_video_bg):
        # image_fg = image_bg.clone()
        # image_fg[self.dynamic_foreground_masks[kf_idx][None, None].repeat(1, 3, 1, 1).bool()] = self.images[kf_idx][self.dynamic_foreground_masks[kf_idx][None, None].repeat(1, 3, 1, 1).bool()]
        # self.image_tensor = image_fg
        # self.images[kf_idx] = image_fg
        # self.inpaint_input_image = [ToPILImage()(image_fg.squeeze())]
        self.rendered_images = [self.image_tensor]
        
        video_fg = video_bg.clone()
        video_fg[self.dynamic_foreground_masks_video[kf_idx][:, None].repeat(1, 3, 1, 1).bool()] = self.videos[kf_idx][self.dynamic_foreground_masks_video[kf_idx][:, None].repeat(1, 3, 1, 1).bool()]
        self.video_tensor = video_fg
        self.videos[kf_idx] = video_fg
        
        depth_fg = depth_bg.clone()
        depth_fg[self.dynamic_foreground_masks[kf_idx][None, None]] = self.depths[kf_idx][self.dynamic_foreground_masks[kf_idx][None, None]]
        disparity_fg = disparity_bg.clone()
        disparity_fg[self.dynamic_foreground_masks[kf_idx][None, None]] = self.disparities[kf_idx][self.dynamic_foreground_masks[kf_idx][None, None]]
        self.depth = depth_fg
        self.depths[kf_idx] = depth_fg
        self.disparities[kf_idx] = disparity_fg
        self.rendered_depths = [self.depth]
        
        depth_video_fg = depth_video_bg.clone()
        depth_video_fg[self.dynamic_foreground_masks_video[kf_idx][:, None]] = self.depths_video[kf_idx][self.dynamic_foreground_masks_video[kf_idx][:, None]]
        disparity_video_fg = disparity_video_bg.clone()
        disparity_video_fg[self.dynamic_foreground_masks_video[kf_idx][:, None]] = self.disparities_video[kf_idx][self.dynamic_foreground_masks_video[kf_idx][:, None]]
        self.depth_video = depth_video_fg
        self.depths_video[kf_idx] = depth_video_fg
        self.disparities_video[kf_idx] = disparity_video_fg
        self.rendered_depths_video = [self.depth_video]
        
        self.bg_images = [image_bg]
        self.bg_depths = [depth_bg]
        self.bg_videos = [video_bg]
        self.bg_depths_video = [depth_video_bg]
        

        
    
    @torch.no_grad()
    def refine_bg_disp_with_segments(self, image, disp, video, video_disp, background_depth_cutoff=1./7., batch_size=2):
        # image = ToPILImage()(self.images[kf_idx].squeeze())
        image = ToPILImage()(image.squeeze())
        video = [ToPILImage()(x) for x in video]
        segmenter_input = self.segment_processor(image, ["semantic"], return_tensors="pt")
        segmenter_input = {name: tensor.to("cuda") for name, tensor in segmenter_input.items()}
        segment_output = self.segment_model(**segmenter_input)
        pred_semantic_map = self.segment_processor.post_process_semantic_segmentation(
                        segment_output, target_sizes=[image.size[::-1]])[0]
        sky_mask = pred_semantic_map.cpu() == 119
        sky_mask = erosion(sky_mask.float()[None, None], 
                    kernel=torch.ones(self.config['sky_erode_kernel_size'], self.config['sky_erode_kernel_size'])
                    ).squeeze() > 0.5
        sky_mask = sky_mask.cpu()
        
        dynamic_background_mask = self.dynamic_background_judgement(pred_semantic_map.cpu())
        
        pred_semantic_map_video = []
        for video_batch in [video[i:i + batch_size] for i in range(0, len(video), batch_size)]:
            
            segmenter_input_video_batch = self.segment_processor(video_batch, ["semantic"]*len(video_batch), return_tensors="pt")
            segmenter_input_video_batch = {name: tensor.to("cuda") for name, tensor in segmenter_input_video_batch.items()}
            segment_output_video_batch = self.segment_model(**segmenter_input_video_batch)
            pred_semantic_map_video_batch = self.segment_processor.post_process_semantic_segmentation(
                        segment_output_video_batch, target_sizes=[image.size[::-1]]*len(video_batch))
            pred_semantic_map_video += [x.cpu() for x in pred_semantic_map_video_batch]
        
        # pred_semantic_map_video = torch.cat(pred_semantic_map_video, dim=0)

        sky_mask_video = torch.stack([x.cpu()==119 for x in pred_semantic_map_video],dim=0)
        sky_mask_video = erosion(sky_mask_video.float()[:, None], 
                           kernel=torch.ones(self.config['sky_erode_kernel_size'], self.config['sky_erode_kernel_size'])
                           ).squeeze() > 0.5
        sky_mask_video = sky_mask_video.cpu()
        # if self.config['mask_path'] != "None":
        #     sky_mask_video = torch.zeros_like(sky_mask_video).bool()
        #     sky_mask = torch.zeros_like(sky_mask).bool()
        
        dynamic_background_mask_video = torch.stack([self.dynamic_background_judgement(x.cpu()) for x in pred_semantic_map_video],dim=0)
        

        image_np = np.array(image)
        video_np = np.array([np.array(x) for x in video])
        
        masks = self.mask_generator.generate(image_np)
        masks_video = [self.mask_generator.generate(x) for x in video_np]
        sorted_mask = sorted(masks, key=(lambda x: x['area']), reverse=False)
        sorted_mask_video = [sorted(frame, key=(lambda x: x['area']), reverse=False) for frame in masks_video]
        min_mask_area = 30
        sorted_mask = [m for m in sorted_mask if m['area'] > min_mask_area]
        sorted_mask_video = [[m for m in sorted_mask_frame if m['area'] > min_mask_area ] for sorted_mask_frame  in sorted_mask_video]

        # disparity_np = self.disparities[kf_idx].squeeze().cpu().numpy()
        disparity_np = disp.squeeze().cpu().numpy()
        disparity_video_np = video_disp.squeeze().cpu().numpy()
        keep_threshold_ratio = 0.3
        refined_disparity = refine_disp_with_segments(disparity_np, sorted_mask, None, keep_threshold=1 / background_depth_cutoff * keep_threshold_ratio)
        refined_disparity_video = []
        for i, disparity_frame_np, sorted_mask_frame in zip(range(len(disparity_video_np)), disparity_video_np, sorted_mask_video):
            refined_disparity_frame = refine_disp_with_segments(disparity_frame_np, sorted_mask_frame, None, keep_threshold=1 / background_depth_cutoff * keep_threshold_ratio)
            refined_disparity_video.append(refined_disparity_frame)
        refined_disparity_video = np.array(refined_disparity_video)
        
        sky_hard_disp = 1. / self.config['sky_hard_depth']
        bg_hard_disp = 1. / (background_depth_cutoff)
        refined_disparity[sky_mask] = sky_hard_disp
        for i in range(len(refined_disparity_video)):
            refined_disparity_video[i,sky_mask_video[i]] = sky_hard_disp

        background_cutoff = 1./background_depth_cutoff
        background_mask = refined_disparity < background_cutoff
        background_but_not_sky_mask = np.logical_and(background_mask, np.logical_not(sky_mask.numpy()))
        refined_disparity[background_but_not_sky_mask] = bg_hard_disp
        background_but_not_sky_mask_video = []
        for i in range(len(refined_disparity_video)):
            background_mask_frame = refined_disparity_video[i] < background_cutoff
            background_but_not_sky_mask_frame = np.logical_and(background_mask_frame, np.logical_not(sky_mask_video[i].numpy()))
            background_but_not_sky_mask_video.append(background_but_not_sky_mask_frame)
            refined_disparity_video[i, background_but_not_sky_mask_frame] = bg_hard_disp
        background_but_not_sky_mask_video = np.stack(background_but_not_sky_mask_video, axis=0)

        refined_disparity = refine_disp_with_segments(refined_disparity, sorted_mask, None, keep_threshold=1 / background_depth_cutoff * keep_threshold_ratio)
        refined_disparity_video_new = []
        for i, refined_disparity_frame, sorted_mask_frame in zip(range(len(refined_disparity_video)), refined_disparity_video, sorted_mask_video):
            refined_disparity_frame = refine_disp_with_segments(refined_disparity_frame, sorted_mask_frame, None, keep_threshold=1 / background_depth_cutoff * keep_threshold_ratio)
            refined_disparity_video_new.append(refined_disparity_frame)
        refined_disparity_video = np.array(refined_disparity_video_new)
        
        #Step 5: temporal smooth
        #! Since this is for refining background, we suppose all the points are background points.
        dynamic_background_mask = torch.ones_like(dynamic_background_mask)
        dynamic_background_mask_video = torch.ones_like(dynamic_background_mask_video)
        smooth_disparity_mask  = np.zeros_like(dynamic_background_mask.numpy())
        smooth_disparity = np.zeros_like(disparity_np)
        for i in range(len(dynamic_background_mask_video)):
            smooth_disparity_mask = smooth_disparity_mask + (np.logical_not(sky_mask_video[i].numpy()) & dynamic_background_mask_video[i].numpy()).astype(int)
            smooth_disparity = smooth_disparity + refined_disparity_video[i] * (np.logical_not(sky_mask_video[i].numpy()) & dynamic_background_mask_video[i].numpy()).astype(int)
        smooth_disparity_mask = smooth_disparity_mask + (np.logical_not(sky_mask.numpy()) & dynamic_background_mask.numpy()).astype(int)
        smooth_disparity = smooth_disparity + refined_disparity * (np.logical_not(sky_mask.numpy()) & dynamic_background_mask.numpy()).astype(int)
        
        smooth_disparity_mask[smooth_disparity_mask == 0] = 1
        smooth_disparity = smooth_disparity / smooth_disparity_mask
        for i in range(len(dynamic_background_mask_video)):
            refined_disparity_video[i, (np.logical_not(sky_mask_video[i].numpy()) & dynamic_background_mask_video[i].numpy())] = smooth_disparity[(np.logical_not(sky_mask_video[i].numpy()) & dynamic_background_mask_video[i].numpy())]
        refined_disparity[(np.logical_not(sky_mask.numpy()) & dynamic_background_mask.numpy())] = smooth_disparity[(np.logical_not(sky_mask.numpy()) & dynamic_background_mask.numpy())]
        
        refined_depth = 1 / refined_disparity
        refined_depth_video = 1 / refined_disparity_video
        
        refined_depth = torch.from_numpy(refined_depth).to(self.device)
        refined_depth_video = torch.from_numpy(refined_depth_video).to(self.device)
        refined_disparity = torch.from_numpy(refined_disparity).to(self.device)
        refined_disparity_video = torch.from_numpy(refined_disparity_video).to(self.device)

        return refined_depth.unsqueeze(0).unsqueeze(0), refined_disparity.unsqueeze(0).unsqueeze(0), refined_depth_video[:, None], refined_disparity_video[:, None]

    @torch.no_grad()
    def render(self, j, epoch):
        self.kf_idx = epoch
        self.total_idx = j
        if self.config["motion"] == "rotations":
            camera = self.get_next_camera_rotation()
        elif self.config["motion"] == "predefined":
            camera = self.predefined_cameras[epoch]
        else:
            raise NotImplementedError
        current_camera = convert_pytorch3d_kornia(self.current_camera, self.config["init_focal_length"])
        point_depth = rearrange(self.depths[epoch - 1], "b c h w -> (w h b) c")
        point_depth_bg = rearrange(self.bg_depths[epoch - 1], "b c h w -> (w h b) c")
        points_3d = current_camera.unproject(self.points, point_depth)
        points_3d_bg = current_camera.unproject(self.points, point_depth_bg)

        colors = rearrange(self.images[epoch - 1], "b c h w -> (w h b) c")
        colors_bg = rearrange(self.bg_images[epoch - 1], "b c h w -> (w h b) c")
        foreground_masks = rearrange(self.dynamic_foreground_masks[epoch - 1][None, None], "b c h w -> (w h b) c")
        depth_normalizer = self.background_hard_depth
        min_ratio = self.config['point_size_min_ratio']
        radius = self.config['point_size'] * (min_ratio + (1 - min_ratio) * (point_depth.permute([1, 0]) / depth_normalizer))
        radius = radius.clamp(max=self.config['point_size']*self.config['sky_point_size_multiplier'])
        radius_bg = self.config['point_size'] * (min_ratio + (1 - min_ratio) * (point_depth_bg.permute([1, 0]) / depth_normalizer))
        radius_bg = radius_bg.clamp(max=self.config['point_size']*self.config['sky_point_size_multiplier'])
        raster_settings = PointsRasterizationSettings(
            image_size=512,
            radius = radius,
            points_per_pixel = 8,
        )
        raster_settings_bg = PointsRasterizationSettings(
            image_size=512,
            radius = radius_bg,
            points_per_pixel = 8,
        )
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
            compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
        )
        
        renderer_bg = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings_bg),
            compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
        )
        points_3d[..., :2] = - points_3d[..., :2]
        points_3d_bg[..., :2] = - points_3d_bg[..., :2]
        point_cloud = Pointclouds(points=[points_3d], features=[colors])
        point_cloud_bg = Pointclouds(points=[points_3d_bg], features=[colors_bg])
        images, zbuf, bg_mask = renderer(point_cloud, return_z=True, return_bg_mask=True)
        images_bg, zbuf_bg, bg_mask_bg = renderer_bg(point_cloud_bg, return_z=True, return_bg_mask=True)

        # bg_mask = dilation(bg_mask[:, None], torch.ones(self.config['dilate_mask_decoder_ft'], self.config['dilate_mask_decoder_ft']).to('cuda'))[:, 0]
        # import ipdb; ipdb.set_trace()
        # import cv2; cv2.imwrite("debug/bgmask.png", np.array(bg_mask[0].cpu()).astype(np.uint8)*255)
        # import cv2; cv2.imwrite("debug/rendered_image.png", np.array(images[0].cpu()*255)[...,::-1].astype(np.uint8))
        
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
            compositor=SoftmaxImportanceCompositor(background_color=(0,), softmax_scale=1.0)
        )
        with torch.no_grad():
            orig_depth, _ = self.get_depth(self.images[epoch - 1])  # kf1_estimate under kf1 frame
        orig_depth = rearrange(orig_depth, "b c h w -> (w h b) c")
        point_cloud_orig_depth = Pointclouds(points=[points_3d], features=[orig_depth])
        rendered_depth_original = renderer(point_cloud_orig_depth) + self.kf_delta_t  # kf1_estimate under kf2 frame, rendered
        rendered_depth_original = rearrange(rendered_depth_original, "b h w c -> b c h w")
        
        renderer_bg = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings_bg),
            compositor=SoftmaxImportanceCompositor(background_color=(0,), softmax_scale=1.0)
        )
        with torch.no_grad():
            orig_depth_bg, _ = self.get_depth(self.bg_images[epoch - 1])  # kf1_estimate under kf1 frame
        orig_depth_bg = rearrange(orig_depth_bg, "b c h w -> (w h b) c")
        point_cloud_orig_depth_bg = Pointclouds(points=[points_3d_bg], features=[orig_depth_bg])
        rendered_depth_original_bg = renderer_bg(point_cloud_orig_depth_bg) + self.kf_delta_t  # kf1_estimate under kf2 frame, rendered
        rendered_depth_original_bg = rearrange(rendered_depth_original_bg, "b h w c -> b c h w")
        
        point_cloud_foreground_mask = Pointclouds(points=[points_3d], features=[foreground_masks.to(points_3d.device)])
        rendered_foreground_masks = renderer(point_cloud_foreground_mask)

        rendered_image = rearrange(images, "b h w c -> b c h w")
        rendered_image_bg = rearrange(images_bg, "b h w c -> b c h w")
        rendered_foreground_mask = rearrange(rendered_foreground_masks, "b h w c -> b c h w")
        
        # rendered_foreground_mask = dilation(rendered_foreground_mask, torch.ones(self.config['dilate_mask_decoder_ft']*4, self.config['dilate_mask_decoder_ft']*4).to('cuda'))
        
        
        # rendered_image = rendered_image * (1 - rendered_foreground_mask)
        # bg_mask = bg_mask | rendered_foreground_mask[:,0].bool()
        
        inpaint_mask = bg_mask.float()[:, None, ...]
        inpaint_mask_bg = bg_mask_bg.float()[:, None, ...]
        rendered_depth = rearrange(zbuf[..., 0:1], "b h w c -> b c h w")
        rendered_depth[rendered_depth < 0] = 0
        rendered_depth_bg = rearrange(zbuf_bg[..., 0:1], "b h w c -> b c h w")
        rendered_depth_bg[rendered_depth_bg < 0] = 0

        self.current_camera = copy.deepcopy(camera)
        self.cameras.append(self.current_camera)
        self.rendered_images.append(rendered_image_bg)
        self.rendered_depths.append(rendered_depth_bg)

        if self.config["inpainting_resolution"] > 512:
            raise NotImplementedError("Not implement foreground_mask")
            padded_inpainting_mask = self.border_mask.clone()
            padded_inpainting_mask[
                :, :, self.border_size : -self.border_size, self.border_size : -self.border_size
            ] = inpaint_mask
            padded_image = self.border_image.clone()
            padded_image[
                :, :, self.border_size : -self.border_size, self.border_size : -self.border_size
            ] = rendered_image
        else:
            padded_inpainting_mask = inpaint_mask
            padded_inpainting_mask_bg = inpaint_mask_bg
            padded_image = rendered_image
            padded_image_bg = rendered_image_bg
        # import ipdb; ipdb.set_trace()
        # save_depth_map(rendered_depth_original.cpu(), "debug/rendered_depths_0.png")

        return {
            "rendered_image": padded_image,
            "rendered_image_bg": padded_image_bg,
            "rendered_depth": rendered_depth,
            "inpaint_mask": padded_inpainting_mask,
            "inpaint_mask_bg": padded_inpainting_mask_bg,
            "inpaint_mask_512": inpaint_mask,
            "rendered_depth_original": rendered_depth_original,
            "foreground_mask": rendered_foreground_mask,
        }
    @torch.no_grad()
    def render_frame(self, epoch, depth_frame, frame, depth_bg_frame, bg_frame, foreground_mask_frame):
        # if self.config["motion"] == "rotations":
        #     camera = self.get_next_camera_rotation()
        # elif self.config["motion"] == "predefined":
        #     camera = self.predefined_cameras[epoch]
        # else:
        #     raise NotImplementedError
        camera = copy.deepcopy(self.cameras[-1])
        current_camera = convert_pytorch3d_kornia(self.cameras[-2], self.config["init_focal_length"])
        point_depth = rearrange(depth_frame, "b c h w -> (w h b) c")
        point_depth_bg = rearrange(depth_bg_frame, "b c h w -> (w h b) c")
        points_3d = current_camera.unproject(self.points, point_depth)
        points_3d_bg = current_camera.unproject(self.points, point_depth_bg)

        colors = rearrange(frame, "b c h w -> (w h b) c")
        colors_bg = rearrange(bg_frame, "b c h w -> (w h b) c")
        foreground_masks = rearrange(foreground_mask_frame, "b c h w -> (w h b) c")
        depth_normalizer = self.background_hard_depth
        min_ratio = self.config['point_size_min_ratio']
        radius = self.config['point_size'] * (min_ratio + (1 - min_ratio) * (point_depth.permute([1, 0]) / depth_normalizer))
        radius = radius.clamp(max=self.config['point_size']*self.config['sky_point_size_multiplier'])
        radius_bg = self.config['point_size'] * (min_ratio + (1 - min_ratio) * (point_depth_bg.permute([1, 0]) / depth_normalizer))
        radius_bg = radius_bg.clamp(max=self.config['point_size']*self.config['sky_point_size_multiplier'])
        raster_settings = PointsRasterizationSettings(
            image_size=512,
            radius = radius,
            points_per_pixel = 8,
        )
        raster_settings_bg = PointsRasterizationSettings(
            image_size=512,
            radius = radius_bg,
            points_per_pixel = 8,
        )
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
            compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
        )
        renderer_bg = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings_bg),
            compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
        )
        points_3d[..., :2] = - points_3d[..., :2]
        points_3d_bg[..., :2] = - points_3d_bg[..., :2]
        point_cloud = Pointclouds(points=[points_3d], features=[colors])
        point_cloud_bg = Pointclouds(points=[points_3d_bg], features=[colors_bg])
        images, zbuf, bg_mask = renderer(point_cloud, return_z=True, return_bg_mask=True)
        images_bg, zbuf_bg, bg_mask_bg = renderer(point_cloud_bg, return_z=True, return_bg_mask=True)
        # import cv2; cv2.imwrite("debug/bgmask.png", np.array(bg_mask[0].cpu()).astype(np.uint8)*255)
        # import cv2; cv2.imwrite("debug/frame.png", (np.array(frame[0].cpu().permute(1,2,0))[...,::-1]*255).astype(np.uint8))

        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
            compositor=SoftmaxImportanceCompositor(background_color=(0,), softmax_scale=1.0)
        )
        with torch.no_grad():
            orig_depth, _ = self.get_depth(frame)  # kf1_estimate under kf1 frame
        orig_depth = rearrange(orig_depth, "b c h w -> (w h b) c")
        point_cloud_orig_depth = Pointclouds(points=[points_3d], features=[orig_depth])
        rendered_depth = renderer(point_cloud_orig_depth) + self.kf_delta_t  # kf1_estimate under kf2 frame, rendered
        rendered_depth = rearrange(rendered_depth, "b h w c -> b c h w")
        
        renderer_bg = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings_bg),
            compositor=SoftmaxImportanceCompositor(background_color=(0,), softmax_scale=1.0)
        )
        with torch.no_grad():
            orig_depth_bg, _ = self.get_depth(bg_frame)  # kf1_estimate under kf1 frame
        orig_depth_bg = rearrange(orig_depth_bg, "b c h w -> (w h b) c")
        point_cloud_orig_depth_bg = Pointclouds(points=[points_3d_bg], features=[orig_depth_bg])
        rendered_depth_bg = renderer_bg(point_cloud_orig_depth_bg) + self.kf_delta_t  # kf1_estimate under kf2 frame, rendered
        rendered_depth_bg = rearrange(rendered_depth_bg, "b h w c -> b c h w")
        
        point_cloud_foreground_mask = Pointclouds(points=[points_3d], features=[foreground_masks.to(points_3d.device)])
        rendered_foreground_masks = renderer(point_cloud_foreground_mask)
        # import cv2; cv2.imwrite("debug/foreground_mask_after_rerender.png", (np.array(rendered_foreground_masks[0].cpu().repeat(1,1,3))[...,::-1]*255).astype(np.uint8))
        

        rendered_image = rearrange(images, "b h w c -> b c h w")
        rendered_image_bg = rearrange(images_bg, "b h w c -> b c h w")
        rendered_foreground_mask = rearrange(rendered_foreground_masks, "b h w c -> b c h w")
        inpaint_mask = bg_mask.float()[:, None, ...]
        inpaint_mask_bg = bg_mask_bg.float()[:, None, ...]
        rendered_depth = rearrange(zbuf[..., 0:1], "b h w c -> b c h w")
        rendered_depth[rendered_depth < 0] = 0
        rendered_depth_bg = rearrange(zbuf_bg[..., 0:1], "b h w c -> b c h w")
        rendered_depth_bg[rendered_depth < 0] = 0

        # self.current_camera = copy.deepcopy(camera)
        # self.cameras.append(self.current_camera)
        # self.rendered_images.append(rendered_image)
        # self.rendered_depths.append(rendered_depth)

        if self.config["inpainting_resolution"] > 512:
            raise NotImplementedError("Not implement foreground_mask")
            padded_inpainting_mask = self.border_mask.clone()
            padded_inpainting_mask[
                :, :, self.border_size : -self.border_size, self.border_size : -self.border_size
            ] = inpaint_mask
            padded_image = self.border_image.clone()
            padded_image[
                :, :, self.border_size : -self.border_size, self.border_size : -self.border_size
            ] = rendered_image
        else:
            padded_inpainting_mask = inpaint_mask
            padded_inpainting_mask_bg = inpaint_mask_bg
            padded_image = rendered_image
            padded_image_bg = rendered_image_bg

        return rendered_image, rendered_depth, {
            "rendered_image": padded_image,
            "rendered_image_bg": padded_image_bg,
            "rendered_depth": rendered_depth,
            "inpaint_mask": padded_inpainting_mask,
            "inpaint_mask_bg": padded_inpainting_mask_bg,
            "inpaint_mask_512": inpaint_mask,
            "rendered_depth_original": rendered_depth,
            "foreground_mask": rendered_foreground_mask,
        }
    def render_video(self, epoch):
        rendered_video, rendered_depth_video = [], []
        rs_dicts = []
        for i in range(len(self.videos[epoch - 1])):
            depth_frame = self.depths_video[epoch - 1][i:i+1]
            # depth_frame = self.depths_video[epoch - 1][0:0+1]
            frame = self.videos[epoch - 1][i:i+1]
            foreground_mask_frame = self.dynamic_foreground_masks_video[epoch - 1][i:i+1].int()[:, None]
            depth_bg_frame = self.bg_depths_video[epoch - 1][i:i+1]
            bg_frame = self.bg_videos[epoch - 1][i:i+1]
            rendered_frame, rendered_depth, rs = self.render_frame(epoch, depth_frame, frame, depth_bg_frame, bg_frame, foreground_mask_frame)
            rendered_video.append(rendered_frame)
            rendered_depth_video.append(rendered_depth)
            rs_dicts.append(rs)
        self.rendered_videos.append(torch.cat(rendered_video, dim=0))
        self.rendered_depths_video.append(torch.cat(rendered_depth_video, dim=0))
        final_dict = {}
        for key in rs_dicts[0].keys():
            final_dict[key] = torch.cat([dict_i[key] for dict_i in rs_dicts], dim=0)
        self.render_output_video = final_dict
        return final_dict
    
    @torch.no_grad()
    def render_feature(self, epoch, depth_frame, frame):
        # if self.config["motion"] == "rotations":
        #     camera = self.get_next_camera_rotation()
        # elif self.config["motion"] == "predefined":
        #     camera = self.predefined_cameras[epoch]
        # else:
        #     raise NotImplementedError
        camera = copy.deepcopy(self.cameras[-1])
        current_camera = convert_pytorch3d_kornia(self.cameras[-2], self.config["init_focal_length"])
        point_depth = rearrange(depth_frame, "b c h w -> (w h b) c")
        points_3d = current_camera.unproject(self.points, point_depth)

        colors = rearrange(frame, "b c h w -> (w h b) c")
        depth_normalizer = self.background_hard_depth
        min_ratio = self.config['point_size_min_ratio']
        radius = self.config['point_size'] * (min_ratio + (1 - min_ratio) * (point_depth.permute([1, 0]) / depth_normalizer))
        radius = radius.clamp(max=self.config['point_size']*self.config['sky_point_size_multiplier'])
        raster_settings = PointsRasterizationSettings(
            image_size=512,
            radius = radius,
            points_per_pixel = 8,
        )
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
            compositor=SoftmaxImportanceCompositor(background_color=(tuple(0 for i in range(colors.shape[-1]))), softmax_scale=1.0)
        )
        points_3d[..., :2] = - points_3d[..., :2]
        point_cloud = Pointclouds(points=[points_3d], features=[colors])
        images, zbuf, bg_mask = renderer(point_cloud, return_z=True, return_bg_mask=True)
        # import cv2; cv2.imwrite("debug/bgmask.png", np.array(bg_mask[0].cpu()).astype(np.uint8)*255)
        # import cv2; cv2.imwrite("debug/frame.png", (np.array(frame[0].cpu().permute(1,2,0))[...,::-1]*255).astype(np.uint8))
        # import cv2; cv2.imwrite("debug/frame_after_rerender.png", (np.array(images[0].cpu())[...,::-1]*255).astype(np.uint8))
        # import ipdb; ipdb.set_trace()



        rendered_image = rearrange(images, "b h w c -> b c h w")
        inpaint_mask = bg_mask.float()[:, None, ...]

        # self.current_camera = copy.deepcopy(camera)
        # self.cameras.append(self.current_camera)
        # self.rendered_images.append(rendered_image)
        # self.rendered_depths.append(rendered_depth)

        if self.config["inpainting_resolution"] > 512:
            padded_inpainting_mask = self.border_mask.clone()
            padded_inpainting_mask[
                :, :, self.border_size : -self.border_size, self.border_size : -self.border_size
            ] = inpaint_mask
            padded_image = self.border_image.clone()
            padded_image[
                :, :, self.border_size : -self.border_size, self.border_size : -self.border_size
            ] = rendered_image
        else:
            padded_inpainting_mask = inpaint_mask
            padded_image = rendered_image

        return rendered_image, inpaint_mask, {
            "rendered_image": padded_image,
            "inpaint_mask": padded_inpainting_mask,
            "inpaint_mask_512": inpaint_mask,
        }
        
    def render_video_pred_cond(self, epoch):
        pred_cond_c = self.pred_cond.shape[1]
        feature_map = torch.cat([self.pred_cond, self.pred_x0], dim=1)
        new_feature_map = torch.zeros_like(feature_map)
        
        steps, channels, frames, cond_h, cond_w = feature_map.shape
        feature_map = rearrange(feature_map, "s c f h w -> (s f) c h w").to(torch.float32)
        resized_feature_map = F.interpolate(feature_map, size=(512, 512), mode='bilinear', align_corners=False)
        resized_feature_map = rearrange(resized_feature_map, "(s f) c h w -> s f c h w", s=steps, f=frames)
        
        feature_masks = torch.zeros((steps, 1, frames, cond_h, cond_w))
        for step in range(steps):
            for i in range(frames):
                depth_frame = self.depths_video[epoch - 1][i:i+1]
                rendered_frame, inpaint_bg_mask, rs = self.render_feature(epoch, depth_frame, resized_feature_map[step, i:i+1].cuda())
                rendered_frame = F.interpolate(rendered_frame.cpu(), size=(64, 64), mode='nearest').to(torch.float32)
                feature_mask = F.interpolate((1-inpaint_bg_mask).cpu(), size=(64, 64), mode='nearest').to(torch.float32)
                
                new_feature_map[step, :, i] = rendered_frame[0]
                feature_masks[step, :, i] = feature_mask[0]

        self.reproj_pred_cond = {
            "pred_cond_map": new_feature_map[:, :pred_cond_c],
            "pred_cond_masks": feature_masks,
            "pred_x0_map": new_feature_map[:, pred_cond_c:],
            }


class KeyframeInterp(FrameSyn):
    def __init__(self, config, inpainter_pipeline, depth_model, vae, rotation, 
                 image, video, inpainting_prompt, bg_outpainting_prompt, adaptive_negative_prompt, video_generation_prompt="", kf2_upsample_coef=1,
                 kf1_image=None, kf2_image=None, 
                 kf1_bg_image=None, kf2_bg_image=None,
                 kf1_video=None, kf2_video=None, 
                 kf1_bg_video=None, kf2_bg_video=None,
                 kf1_depth=None, kf2_depth=None, kf1_bg_depth=None, kf2_bg_depth=None, kf1_depth_video=None, kf2_depth_video=None,
                 kf1_bg_depth_video=None, kf2_bg_depth_video=None,
                 kf1_foreground_mask=None, kf2_foreground_mask=None, 
                 kf1_dynamic_foreground_masks_video=None, kf2_dynamic_foreground_masks_video=None,
                 kf1_dynamic_background_masks_video=None, kf2_dynamic_background_masks_video=None,
                 kf1_camera=None, kf2_camera=None, kf2_mask=None, kf2_mask_video=None,
                 speed_up=False, speed_down=False, total_frames=None, keyframe_idx=1):
        
        dt_string = datetime.now().strftime("%d-%m_%H-%M-%S")
        run_dir_root = Path(config["runs_dir"])
        self.run_dir = run_dir_root / f"Interp-{dt_string}_{inpainting_prompt.replace(' ', '_')[:40]}"
        (self.run_dir / 'images').mkdir(parents=True, exist_ok=True)

        self.speed_up = speed_up
        self.speed_down = speed_down

        self.random_walk_scale_vertical = np.random.uniform(0.1, 0.3)

        config['forward_speed_multiplier'] = -1. / (config['frames'] + 1)
        config['inpainting_resolution'] = config['inpainting_resolution_interp']
        config['right_multiplier'] = 0
        config['rotation_range_theta'] = config['rotation_range'] / (config['frames'] + 1)
        
        kf1_depth[kf1_foreground_mask[None, None]] = kf1_bg_depth[kf1_foreground_mask[None, None]]
        kf1_depth_video_new = kf1_bg_depth.repeat((kf1_depth_video.shape[0], 1, 1, 1) )
        kf1_depth_video_new[kf1_dynamic_foreground_masks_video[:, None, :, :]] = kf1_depth_video[kf1_dynamic_foreground_masks_video[:, None, :, :]]
        kf1_bg_depth_video = kf1_bg_depth.repeat((kf1_depth_video.shape[0], 1, 1, 1) )
        kf2_depth[kf2_foreground_mask[None, None]] = kf2_bg_depth[kf2_foreground_mask[None, None]]
        kf2_depth_video_new = kf2_bg_depth.repeat((kf2_depth_video.shape[0], 1, 1, 1) )
        kf2_depth_video_new[kf2_dynamic_foreground_masks_video[:, None, :, :]] = kf2_depth_video[kf2_dynamic_foreground_masks_video[:, None, :, :]]
        kf2_bg_depth_video = kf2_bg_depth.repeat((kf2_depth_video.shape[0], 1, 1, 1) )
        kf1_depth_video = kf1_depth_video_new.cuda()
        kf2_depth_video = kf2_depth_video_new.cuda()
        
        super().__init__(config, inpainter_pipeline, depth_model, vae, rotation,  
                         image, video, inpainting_prompt, adaptive_negative_prompt, video_generation_prompt,
                         bg_outpainting_prompt=bg_outpainting_prompt,depth=kf2_depth, depth_video=kf2_depth_video)
        self.total_frames = config['frames'] if total_frames is None else total_frames

        self.additional_points_3d = torch.tensor([]).cuda()
        self.additional_colors = torch.tensor([]).cuda()
        self.additional_foreground_masks = torch.tensor([]).cuda()

        self.kf2_upsample_coef = kf2_upsample_coef
        x = torch.arange(512 * kf2_upsample_coef)
        y = torch.arange(512 * kf2_upsample_coef)
        self.points_kf2 = torch.stack(torch.meshgrid(x, y, indexing='ij'), -1)
        self.points_kf2 = rearrange(self.points_kf2, "h w c -> (h w) c").to(self.device)
        self.use_noprompt = True
        
        # kf2_image_path = "/home/tianfr/data/DynamiCrafter/prompts/village/56_village_outpaint.png"
        # if OmegaConf.select(config,'use_prerendered_images_videos', default=False):
        #     kf1_image_path = config[f'kf{keyframe_idx}_image_path']
        #     kf1_image = ToTensor()(Image.open(kf1_image_path).convert('RGB').resize((512, 512))).unsqueeze(0).cuda()
        #     kf2_image_path = config[f'kf{keyframe_idx+1}_image_path']
        #     kf2_image = ToTensor()(Image.open(kf2_image_path).convert('RGB').resize((512, 512))).unsqueeze(0).cuda()
        
        self.kf1_colors = rearrange(kf1_image, "b c h w -> (w h b) c")
        self.kf1_image = kf1_image
        self.kf2_image = kf2_image
        self.kf1_bg_image = kf1_bg_image
        self.kf2_bg_image = kf2_bg_image
        
        # kf1_video_path = "/home/tianfr/data/DynamiCrafter/results/dynamicrafter_1024_seed123/samples_separate/56_village_sample0.mp4"
        # kf1_video_path = config[f'kf{keyframe_idx}_video_path']
        # self.kf1_video_frames = video2images(kf1_video_path)
        # self.kf1_video_frames = rearrange(kf1_video_frames, "b h w c -> b c h w").cuda()
        # self.kf1_video_colors = rearrange(self.kf1_video_frames, "b c h w -> b (w h) c").cuda()
        # kf1_video_new = kf1_image.repeat((kf1_video.shape[0], 1, 1, 1))
        # kf2_video_new = kf2_image.repeat((kf2_video.shape[0], 1, 1, 1))
        # kf1_video[~(kf1_dynamic_foreground_masks_video[:, None, :, :].repeat((1,3,1,1)) | kf1_dynamic_background_masks_video[:, None, :, :].repeat((1,3,1,1)))] = kf1_video_new[~(kf1_dynamic_foreground_masks_video[:, None, :, :].repeat((1,3,1,1)) | kf1_dynamic_background_masks_video[:, None, :, :].repeat((1,3,1,1)))]
        # kf2_video[~(kf2_dynamic_foreground_masks_video[:, None, :, :].repeat((1,3,1,1)) | kf2_dynamic_background_masks_video[:, None, :, :].repeat((1,3,1,1)))] = kf2_video_new[~(kf2_dynamic_foreground_masks_video[:, None, :, :].repeat((1,3,1,1)) | kf2_dynamic_background_masks_video[:, None, :, :].repeat((1,3,1,1)))]
        # kf1_video = kf1_video_new
        # kf2_video = kf2_video_new
        
        self.kf1_video_frames = kf1_video
        self.kf1_video_colors = rearrange(self.kf1_video_frames, "b c h w -> b (w h) c").cuda()
        self.kf1_video_colors = [x for x in self.kf1_video_colors]
        
        # kf2_video_path = "/home/tianfr/data/DynamiCrafter/results/dynamicrafter_1024_seed123/samples_separate/56_village_outpaint_sample0.mp4"
        # kf2_video_path = config[f'kf{keyframe_idx+1}_video_path']
        # self.kf2_video_frames = video2images(kf2_video_path).cuda()
        self.kf2_video_frames = kf2_video
        # self.kf2_video_frames =  rearrange(kf2_video_frames, "b h w c -> b c h w")

        self.kf1_bg_video_frames = kf1_bg_video
        self.kf2_bg_video_frames = kf2_bg_video
        
        
        
        self.additional_video_colors = [torch.tensor([]).cuda() for x in range(len(self.kf2_video_frames))]
        # self.additional_video_colors = torch.stack(self.additional_video_colors)
        
        self.additional_video_points_3d = [torch.tensor([]).cuda() for x in range(len(self.kf2_video_frames))]
        self.additional_video_foreground_masks = [torch.tensor([]).cuda() for x in range(len(self.kf2_video_frames))]
        # self.additional_video_points_3d = torch.stack(self.additional_video_points_3d)
        # self.kf2_image = self.kf2_video_frames

        self.kf1_depth = kf1_depth
        self.kf2_depth = kf2_depth
        

        self.kf1_depth_video = kf1_depth_video
        self.kf2_depth_video = kf2_depth_video
        self.kf1_bg_depth_video = kf1_bg_depth_video
        self.kf2_bg_depth_video = kf2_bg_depth_video
        self.kf1_camera = kf1_camera
        self.kf2_camera = kf2_camera
        
        self.kf1_dynamic_foreground_masks = kf1_foreground_mask[None, None].float().cuda()
        self.kf1_foregrounds = rearrange(self.kf1_dynamic_foreground_masks, "b c h w -> (w h b) c")
        self.kf2_dynamic_foreground_masks = kf2_foreground_mask[None, None].float().cuda()
        self.kf1_dynamic_foreground_masks_video = kf1_dynamic_foreground_masks_video[:, None].float().cuda()
        self.kf2_dynamic_foreground_masks_video = kf2_dynamic_foreground_masks_video[:, None].float().cuda()
        self.kf1_video_foregrounds = rearrange(self.kf1_dynamic_foreground_masks_video, "b c h w -> b (w h) c")
        self.kf1_video_foregrounds = [x for x in self.kf1_video_foregrounds]
        
        self.rendered_foregrounds = [self.kf2_dynamic_foreground_masks]
        self.rendered_foregrounds_video = [self.kf2_dynamic_foreground_masks_video]
        
        
        # self.kf1_dy_fg_masks_video = rearrange(kf1_dynamic_foreground_masks_video, "b h w -> b (w h)").cuda()
        # self.kf1_dy_bg_masks_video = rearrange(kf1_dynamic_background_masks_video, "b h w -> b (w h)").cuda()
        # self.kf2_dy_fg_masks_video = rearrange(kf2_dynamic_foreground_masks_video, "b h w -> b (w h)").cuda()
        # self.kf2_dy_bg_masks_video = rearrange(kf2_dynamic_background_masks_video, "b h w -> b (w h)").cuda()
        
        #! Check dy_bg:
        # self.kf1_video_colors = self.kf1_video_colors * (self.kf1_dy_bg_masks_video[..., None] | self.kf1_dy_fg_masks_video[..., None])
        # self.kf2_video_colors = self.kf2_video_colors * self.kf2_dy_fg_masks_video[..., None]
        # self.kf1_video_frames = self.kf1_video_frames * (kf1_dynamic_background_masks_video[:, None, ...] | kf1_dynamic_foreground_masks_video[:, None, ...]).cuda()
        # self.kf2_video_frames = self.kf2_video_frames * (kf2_dynamic_background_masks_video[:, None, ...] | kf2_dynamic_background_masks_video[:, None, ...]).cuda()
        
        self.kf2_mask = kf2_mask
        # merged_kf2_mask_video = kf2_mask_video[0].bool()
        # for i in range(1, len(kf2_mask_video)):
        #     merged_kf2_mask_video = merged_kf2_mask_video | kf2_mask_video[i].bool()
        # new_kf2_mask_video = merged_kf2_mask_video[None].repeat(len(kf2_mask_video), 1, 1, 1)
        # self.kf2_mask_video = new_kf2_mask_video.float()
        self.kf2_mask_video = kf2_mask_video
        
        #!Use foreground+background video as initialization
        self.point_depth = rearrange(kf1_depth, "b c h w -> (w h b) c")
        self.point_depth_video = rearrange(kf1_depth_video, "b c h w -> b (w h) c")
        self.point_depth_video = [x for x in self.point_depth_video]
        
        kf1_camera = convert_pytorch3d_kornia(kf1_camera, self.config["init_focal_length"])
        point_foreground_mask = rearrange(self.kf1_dynamic_foreground_masks, "b c h w -> (w h b) c")
        
        self.points_3d = kf1_camera.unproject(self.points, self.point_depth)
        self.points_3d[..., :2] = - self.points_3d[..., :2]
        point_depth_bg = rearrange(kf1_bg_depth, "b c h w -> (w h b) c")
        points_3d_bg = kf1_camera.unproject(self.points, point_depth_bg)
        points_3d_bg[..., :2] = - points_3d_bg[..., :2]
        points_3d_bg = points_3d_bg[point_foreground_mask[:, 0].bool()]
        self.points_3d = torch.cat([self.points_3d, points_3d_bg], dim=0)
        

        kf1_bg_color = rearrange(kf1_bg_image, "b c h w -> (w h b) c")
        kf1_bg_color = kf1_bg_color[point_foreground_mask[:, 0].bool()]
        self.kf1_colors = torch.cat([self.kf1_colors, kf1_bg_color], dim=0)
        
        self.points_3d_video = []
        point_bg_depth_video = rearrange(kf1_bg_depth_video, "b c h w -> b (w h) c")
        bg_color_video = rearrange(kf1_bg_video, "b c h w -> b (w h) c")
        point_foreground_mask_video = rearrange(self.kf1_dynamic_foreground_masks_video, "b c h w -> b (w h) c")
        bg_color_video = [bg_color_video[i][point_foreground_mask_video[i, :, 0].bool()] for i in range(len(point_foreground_mask_video))]
        for i in range(len(self.point_depth_video)):
            point_bg_depth_video[i][point_foreground_mask_video[i, :, 0].bool()] = torch.maximum(point_bg_depth_video[i][point_foreground_mask_video[i, :, 0].bool()], self.point_depth_video[i][point_foreground_mask_video[i, :, 0].bool()] +1e-4)
            if self.config['use_dynamic_point_cloud']:
                points_frame_3d = kf1_camera.unproject(self.points, self.point_depth_video[i])
            else:
                points_frame_3d = kf1_camera.unproject(self.points, self.point_depth)
            points_bg_frame_3d = kf1_camera.unproject(self.points, point_bg_depth_video[i])
            points_bg_frame_3d = points_bg_frame_3d[point_foreground_mask_video[i, :, 0].bool()]
            points_frame_3d = torch.cat([points_frame_3d, points_bg_frame_3d], dim=0)
            self.kf1_video_colors[i] = torch.cat([self.kf1_video_colors[i], bg_color_video[i]], dim=0)
            points_frame_3d[..., :2] = - points_frame_3d[..., :2]
            self.points_3d_video.append(points_frame_3d)
            self.point_depth_video[i] = torch.cat([self.point_depth_video[i], point_bg_depth_video[i][point_foreground_mask_video[i, :, 0].bool()]], dim=0)
            self.kf1_video_foregrounds[i] = torch.cat([self.kf1_video_foregrounds[i], torch.zeros_like(self.kf1_video_foregrounds[i])[point_foreground_mask_video[i, :, 0].bool()]], dim=0)

        self.reinit()

    @torch.no_grad()
    def reinit(self):
        # Image logs
        self.images = [self.kf2_image]
        self.videos = [self.kf2_video_frames]
        self.inpaint_input_image = [self.inpaint_input_image[-1]]
        self.depths = [self.kf2_depth]
        self.depths_video = [self.kf2_depth_video]
        self.masks = [self.masks[-1]]
        self.post_masks = [self.post_masks[-1]]
        self.post_mask_tmp = None
        self.rendered_images = [self.kf2_image]
        self.rendered_videos = [self.kf2_video_frames]
        self.rendered_depths = [self.kf2_depth]
        self.rendered_depths_video = [self.kf2_depth_video]
        self.rendered_foregrounds = [self.kf2_dynamic_foreground_masks]
        self.rendered_foregrounds_video = [self.kf2_dynamic_foreground_masks_video]

        # Cameras
        self.current_camera = copy.deepcopy(self.kf2_camera)
        if self.config["motion"] == "rotations":
            self.current_camera.no_rotations_count = 0
            self.current_camera.rotations_count = 1
            self.current_camera.rotating_right = -self.current_camera.rotating_right
            self.current_camera.move_dir = torch.tensor([[0.0, 0.0, self.config['forward_speed_multiplier']]], device=self.device)
        else:
            raise NotImplementedError
        
    @torch.no_grad()
    def upsample_kf2(self, time=0):
        kf2_size = 512 * self.kf2_upsample_coef
        kf2_focal = self.config["init_focal_length"] * self.kf2_upsample_coef
        kf2_camera_upsample = convert_pytorch3d_kornia(self.kf2_camera, kf2_focal, size=kf2_size)
        kf2_depth_upsample = F.interpolate(self.kf2_depth, size=(kf2_size, kf2_size), mode="nearest")
        kf2_mask_upsample = F.interpolate(self.kf2_mask, size=(kf2_size, kf2_size), mode="nearest")
        
        kf2_dynamic_foreground_mask_upsample = F.interpolate(self.kf2_dynamic_foreground_masks, size=(kf2_size, kf2_size), mode="nearest")
        
        kf2_depth_video_upsample = F.interpolate(self.kf2_depth_video, size=(kf2_size, kf2_size), mode="nearest")
        kf2_bg_depth_video_upsample = F.interpolate(self.kf2_bg_depth_video, size=(kf2_size, kf2_size), mode="nearest")
        kf2_mask_video_upsample = F.interpolate(self.kf2_mask_video, size=(kf2_size, kf2_size), mode="nearest")
        kf2_dynamic_foreground_masks_video_upsample = F.interpolate(self.kf2_dynamic_foreground_masks_video, size=(kf2_size, kf2_size), mode="nearest")
        
        kf2_video_frames_upsample = []
        kf2_bg_video_frames_upsample = []
        for frame, frame_bg in zip(self.kf2_video_frames, self.kf2_bg_video_frames):
            frame_pil_upsample = ToPILImage()(frame).resize((kf2_size, kf2_size), resample=Image.LANCZOS)
            frame_bg_pil_upsample = ToPILImage()(frame_bg).resize((kf2_size, kf2_size), resample=Image.LANCZOS)
            frame_image_upsample = ToTensor()(frame_pil_upsample).unsqueeze(0).to(self.config['device'])
            frame_bg_image_upsample = ToTensor()(frame_bg_pil_upsample).unsqueeze(0).to(self.config['device'])
            kf2_video_frames_upsample.append(frame_image_upsample)
            kf2_bg_video_frames_upsample.append(frame_bg_image_upsample)
        kf2_video_frames_upsample = torch.cat(kf2_video_frames_upsample, dim=0)
        kf2_bg_video_frames_upsample = torch.cat(kf2_bg_video_frames_upsample, dim=0)
        
        kf2_pil_upsample = ToPILImage()(self.kf2_image[0]).resize((kf2_size, kf2_size), resample=Image.LANCZOS)
        kf2_image_upsample = ToTensor()(kf2_pil_upsample).unsqueeze(0).to(self.config['device'])
        kf2_bg_pil_upsample = ToPILImage()(self.kf2_bg_image[0]).resize((kf2_size, kf2_size), resample=Image.LANCZOS)
        kf2_bg_image_upsample = ToTensor()(kf2_bg_pil_upsample).unsqueeze(0).to(self.config['device'])
        return kf2_camera_upsample, kf2_depth_upsample, kf2_mask_upsample, kf2_dynamic_foreground_mask_upsample, kf2_image_upsample, kf2_bg_image_upsample, kf2_depth_video_upsample, kf2_bg_depth_video_upsample, kf2_mask_video_upsample, kf2_dynamic_foreground_masks_video_upsample, kf2_video_frames_upsample, kf2_bg_video_frames_upsample
    
    @torch.no_grad()
    def render_kf1(self, epoch, frame_id, fix_view=False):
        #TODO: Instert time axis to correctly select the target point clouds and cameras.
        if fix_view: 
            camera = copy.deepcopy(self.current_camera)
        elif self.config["motion"] == "rotations":
            camera = self.get_next_camera_rotation()
        elif self.config["motion"] == "predefined":
            camera = self.predefined_cameras[epoch]
        else:
            raise NotImplementedError
        # here we assume that the z coord of self.additional_points_3d is the same as the depth; only true when NO ROTATION
        point_depth_aug = torch.cat([self.point_depth_video[frame_id], self.additional_video_points_3d[frame_id][..., -1:]], dim=0)

        depth_normalizer = self.background_hard_depth
        min_ratio = self.config['point_size_min_ratio']
        radius = self.config['point_size'] * (min_ratio + (1 - min_ratio) * (point_depth_aug.permute([1, 0]) / depth_normalizer))
        radius = radius.clamp(max=self.config['point_size']*self.config['sky_point_size_multiplier'])
        raster_settings = PointsRasterizationSettings(
            image_size=512,
            radius = radius,
            points_per_pixel = 8,
        )
        renderer = PointsRenderer(
                    rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
                    compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
                )
        # points_3d_aug = torch.cat([self.points_3d, self.additional_points_3d], dim=0)
        if self.config['use_dynamic_point_cloud']:
            points_3d_aug = torch.cat([self.points_3d_video[frame_id], self.additional_video_points_3d[frame_id]], dim=0)
        else:
            points_3d_aug = torch.cat([self.points_3d, self.additional_points_3d], dim=0)
        if self.additional_video_colors is not None:
            kf1_video_frame_colors = self.kf1_video_colors[frame_id]
            kf1_video_frame_foregrounds = self.kf1_video_foregrounds[frame_id]
            # print(len(self.additional_colors), len(self.additional_video_colors[0]))
            # import ipdb; ipdb.set_trace()
            additional_colors = self.additional_video_colors[frame_id]
            additional_foregrounds = self.additional_video_foreground_masks[frame_id]
            colors_aug = torch.cat([kf1_video_frame_colors, additional_colors], dim=0)
            foregrounds_aug = torch.cat([kf1_video_frame_foregrounds, additional_foregrounds], dim=0)
        # colors_aug = torch.cat([self.kf1_colors, self.additional_colors], dim=0)
        point_cloud = Pointclouds(points=[points_3d_aug], features=[colors_aug])
        images, zbuf, bg_mask = renderer(point_cloud, return_z=True, return_bg_mask=True)
        
        renderer = PointsRenderer(
            rasterizer=PointsRasterizer(cameras=camera, raster_settings=raster_settings),
            compositor=SoftmaxImportanceCompositor(background_color=(0,), softmax_scale=1.0)
        )
        foreground_point_cloud = Pointclouds(points=[points_3d_aug], features=[foregrounds_aug])
        foregrounds = renderer(foreground_point_cloud)
        
        

        rendered_image = rearrange(images, "b h w c -> b c h w")
        rendered_foreground = rearrange(foregrounds, "b h w c -> b c h w")
        inpaint_mask = bg_mask.float()[:, None, ...]
        rendered_depth = rearrange(zbuf[..., 0:1], "b h w c -> b c h w")
        rendered_depth[rendered_depth < 0] = 0

        self.rendered_images.append(rendered_image)
        curr_rendered_videos = self.rendered_videos[-1]
        curr_rendered_videos[frame_id] = rendered_image[0]
        self.rendered_videos.append(curr_rendered_videos)
        
        self.rendered_depths.append(rendered_depth)
        curr_rendered_depths_video = self.rendered_depths_video[-1]
        curr_rendered_depths_video[frame_id] = rendered_depth[0]
        self.rendered_depths_video.append(curr_rendered_depths_video)
        
        self.rendered_foregrounds.append(rendered_foreground)
        curr_rendered_foregrounds_video = self.rendered_foregrounds_video[-1]
        curr_rendered_foregrounds_video[frame_id] = rendered_foreground[0]
        self.rendered_foregrounds_video.append(curr_rendered_foregrounds_video)
        
        self.current_camera = copy.deepcopy(camera)
        self.cameras.append(self.current_camera)

        if self.config["inpainting_resolution"] > 512:
            raise NotImplementedError()
            padded_inpainting_mask = self.border_mask.clone()
            padded_inpainting_mask[
                :, :, self.border_size : -self.border_size, self.border_size : -self.border_size
            ] = inpaint_mask
            padded_image = self.border_image.clone()
            padded_image[
                :, :, self.border_size : -self.border_size, self.border_size : -self.border_size
            ] = rendered_image
        else:
            padded_inpainting_mask = inpaint_mask
            padded_image = rendered_image
            padded_foreground_mask = rendered_foreground

        return {
            "rendered_image": padded_image,
            "rendered_depth": rendered_depth,
            "inpaint_mask": padded_inpainting_mask,
            "foreground_mask": padded_foreground_mask,
        }

    @torch.no_grad()
    def visibility_check(self):
        radius = self.config['point_size']
        K = 32
        raster_settings = PointsRasterizationSettings(
            image_size=512,
            radius = radius,
            points_per_pixel = K,
        )
        renderer = PointsRenderer(
                    rasterizer=PointsRasterizer(cameras=self.kf1_camera, raster_settings=raster_settings),
                    compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
                )
        points_3d = self.points_3d
        n_kf1_points = points_3d.shape[0]
        colors = self.kf1_colors
        point_cloud = Pointclouds(points=[points_3d], features=[colors])
        images = renderer(point_cloud)

        re_rendered = rearrange(images, "b h w c -> b c h w")
        points_3d_aug = torch.cat([self.points_3d, self.additional_points_3d], dim=0)
        colors_aug = torch.cat([self.kf1_colors, self.additional_colors], dim=0)
        point_cloud_aug = Pointclouds(points=[points_3d_aug], features=[colors_aug])
        images_aug, fragment_idx = renderer(point_cloud_aug, return_fragment_idx=True)  # fragment_idx: [B, H, W, K]
        re_rendered_aug = rearrange(images_aug, "b h w c -> b c h w")

        difference_image = torch.abs(re_rendered - re_rendered_aug).sum(dim=1)  # [B, H, W]
        inconsistent_px = difference_image > 0
        inconsistent_px_point_idx = fragment_idx[inconsistent_px]  # [N, K]
        inconsistent_px_point_from_kf1 = (inconsistent_px_point_idx < n_kf1_points) & (inconsistent_px_point_idx >= 0)  # [N, K], only one True in each

        def find_nearer_points(x):
            """
            args:
                x: [N, 32]. x has exactly one True in each of the N entries. For example, x might look like this:
                    x = [[T, F, F], [F, T, F], [F, F, T]]
            return:
                y: [N, 32]. y[n, i] is True if its position is before the only True in x[n]. For other y[n, i], they all are False.
                    y = [[F, F, F], [T, F, F], [T, T, F]]
            """
            # Convert x to an integer tensor for argmax
            x_int = x.int()
            # Find the indices of the True values in each row of x
            true_indices = torch.argmax(x_int, dim=1)
            # Create a tensor of indices for each position in x
            indices = torch.arange(x.shape[1]).unsqueeze(0).expand_as(x).to(x.device)
            # Compare these indices with the True indices to determine if they come before or after
            y_vectorized = indices < true_indices.unsqueeze(1)
            return y_vectorized
        
        inconsistent_px_point_from_addi = find_nearer_points(inconsistent_px_point_from_kf1)  # [N, K]
        inconsistent_px_point_from_addi_ = inconsistent_px_point_idx[inconsistent_px_point_from_addi]
        inconsistent_px_point_from_addi_ = inconsistent_px_point_from_addi_.unique()  # [T]
        inconsistent_addi_point_idx = inconsistent_px_point_from_addi_ - n_kf1_points  # [T]

        return inconsistent_addi_point_idx
    
    @torch.no_grad()
    def visibility_check_video(self):
        def find_nearer_points(x):
            """
            args:
                x: [N, 32]. x has exactly one True in each of the N entries. For example, x might look like this:
                    x = [[T, F, F], [F, T, F], [F, F, T]]
            return:
                y: [N, 32]. y[n, i] is True if its position is before the only True in x[n]. For other y[n, i], they all are False.
                    y = [[F, F, F], [T, F, F], [T, T, F]]
            """
            # Convert x to an integer tensor for argmax
            x_int = x.int()
            # Find the indices of the True values in each row of x
            true_indices = torch.argmax(x_int, dim=1)
            # Create a tensor of indices for each position in x
            indices = torch.arange(x.shape[1]).unsqueeze(0).expand_as(x).to(x.device)
            # Compare these indices with the True indices to determine if they come before or after
            y_vectorized = indices < true_indices.unsqueeze(1)
            return y_vectorized
        radius = self.config['point_size']
        K = 32
        raster_settings = PointsRasterizationSettings(
            image_size=512,
            radius = radius,
            points_per_pixel = K,
        )
        renderer = PointsRenderer(
                    rasterizer=PointsRasterizer(cameras=self.kf1_camera, raster_settings=raster_settings),
                    compositor=SoftmaxImportanceCompositor(background_color=BG_COLOR, softmax_scale=1.0)
                )
        points_3d = self.points_3d_video
        
        colors = self.kf1_video_colors
        point_cloud = Pointclouds(points=[x for x in points_3d], features=[x for x in colors])
        images = renderer(point_cloud)

        re_rendered = rearrange(images, "b h w c -> b c h w")
        additional_points_3d = self.additional_video_points_3d
        additional_colors = self.additional_video_colors
        points_3d_aug = [torch.cat([points_3d[i], additional_points_3d[i]], dim=0) for i in range(len(points_3d))]
        colors_aug = [torch.cat([colors[i], additional_colors[i]], dim=0) for i in range(len(points_3d))]
        # images_aug, fragment_idx = [], []
        inconsistent_addi_point_idx_all = []
        for i in range(len(additional_points_3d)):
            n_kf1_points = points_3d[i].shape[0]
            
            point_cloud_aug = Pointclouds(points=[points_3d_aug[i]], features=[colors_aug[i]])
            images_aug_i, fragment_idx_i = renderer(point_cloud_aug, return_fragment_idx=True)  # fragment_idx: [B, H, W, K]
            # images_aug.append(images_aug_i)
            # fragment_idx.append(fragment_idx_i)
            # images_aug = torch.cat(images_aug, dim=0)
            # fragment_idx = torch.cat(fragment_idx, dim=0)
            # import ipdb; ipdb.set_trace()
        
            re_rendered_aug = rearrange(images_aug_i, "b h w c -> b c h w")

            difference_image = torch.abs(re_rendered[i] - re_rendered_aug).sum(dim=1)  # [B, H, W]
            inconsistent_px = difference_image > 0
            inconsistent_px_point_idx = fragment_idx_i[inconsistent_px]  # [N, K]
            inconsistent_px_point_from_kf1 = (inconsistent_px_point_idx < n_kf1_points) & (inconsistent_px_point_idx >= 0)  # [N, K], only one True in each
            
            inconsistent_px_point_from_addi = find_nearer_points(inconsistent_px_point_from_kf1)  # [N, K]
            inconsistent_px_point_from_addi_ = inconsistent_px_point_idx[inconsistent_px_point_from_addi]
            inconsistent_px_point_from_addi_ = inconsistent_px_point_from_addi_.unique()  # [T]
            inconsistent_addi_point_idx = inconsistent_px_point_from_addi_ - n_kf1_points  # [T]
            inconsistent_addi_point_idx_all.append(inconsistent_addi_point_idx)

        return inconsistent_addi_point_idx_all

    @torch.no_grad()
    def update_additional_point_cloud(self, rendered_depth, rendered_depth_video, image, video_frames, foreground_mask, foreground_mask_video, valid_mask=None, valid_mask_video=None, camera=None, points_2d=None, append_depth=False):
        """
        args:
            rendered_depth: Depth relative to camera. Note that KF2 camera is represented in KF1 camera-centered coord frame.
            valid_mask: if None, then use inpaint_mask (given by rendered_depth == 0) to extract new points.
                if not None, then just valid_mask to extract new points.
        return:
            Does not really return anything, but updates the following attributes:
            - additional_points_3d: 3D points in KF1 camera-centered coord frame.
            - additional_colors: corresponding colors
        """

        inpaint_mask = rendered_depth == 0
        rendered_depth_filled = rendered_depth.clone()
        inpaint_mask_onthefly = inpaint_mask.clone()
        inpaint_mask_video = rendered_depth_video == 0
        rendered_depth_video_filled = rendered_depth_video.clone()
        inpaint_mask_video_onthefly = inpaint_mask_video.clone()

        def nearest_neighbor_inpainting(inpaint_mask, rendered_depth, window_size=20):
            """
            Perform nearest neighbor inpainting with a local search window.

            Parameters:
            inpaint_mask (torch.Tensor): Binary mask indicating missing values.
            rendered_depth (torch.Tensor): Input depth image.
            window_size (int): Size of the local search window.

            Returns:
            torch.Tensor: Inpainted depth image.
            """

            # Step 1: Find coordinates of invalid and valid pixels
            invalid_coords = torch.nonzero(inpaint_mask.squeeze(), as_tuple=False)
            valid_coords = torch.nonzero(~inpaint_mask.squeeze(), as_tuple=False)

            # Step 4: Use indices to copy depth values from valid to invalid pixels
            rendered_depth_copy = rendered_depth.clone()

            # Define half window size
            hw = window_size // 2

            # Iterate through invalid coordinates
            for idx in range(invalid_coords.size(0)):
                x, y = invalid_coords[idx, 0], invalid_coords[idx, 1]

                # Define local search window
                x_start, x_end = max(0, x - hw), min(rendered_depth.size(2), x + hw + 1)
                y_start, y_end = max(0, y - hw), min(rendered_depth.size(3), y + hw + 1)

                # Extract valid coordinates within the window
                local_valid_coords = valid_coords[(valid_coords[:, 0] >= x_start) & (valid_coords[:, 0] < x_end) & 
                                                (valid_coords[:, 1] >= y_start) & (valid_coords[:, 1] < y_end)]

                # Compute distances and find nearest neighbor
                if local_valid_coords.size(0) > 0:
                    dists = torch.cdist(invalid_coords[idx, :].unsqueeze(0).float(), local_valid_coords.float())
                    min_idx = torch.argmin(dists)
                    rendered_depth_copy[0, 0, x, y] = rendered_depth[0, 0, local_valid_coords[min_idx, 0], local_valid_coords[min_idx, 1]]

            return rendered_depth_copy

        while inpaint_mask_onthefly.sum() > 0:  # iteratively inpaint depth until all depth holes are filled
            rendered_depth_filled = nearest_neighbor_inpainting(inpaint_mask_onthefly, rendered_depth_filled, window_size=50)
            inpaint_mask_onthefly = rendered_depth_filled == 0
        for i in range(len(inpaint_mask)):
            while inpaint_mask_video_onthefly[i].sum() > 0:  # iteratively inpaint depth until all depth holes are filled
                rendered_depth_video_filled[i] = nearest_neighbor_inpainting(inpaint_mask_video_onthefly[i], rendered_depth_video_filled[i], window_size=50)
                inpaint_mask_video_onthefly[i] = rendered_depth_video_filled[i] == 0

        current_camera = convert_pytorch3d_kornia(self.current_camera, self.config["init_focal_length"]) if camera is None else camera
        points_2d = self.points if points_2d is None else points_2d
        points_3d = current_camera.unproject(points_2d, rearrange(rendered_depth_filled, "b c h w -> (w h b) c"))
        points_3d[..., :2] = - points_3d[..., :2]
        inpaint_mask = rearrange(inpaint_mask, "b c h w -> (w h b) c")
        colors = rearrange(image, "b c h w -> (w h b) c")
        foreground_mask = rearrange(foreground_mask, "b c h w -> (w h b) c")

        if valid_mask is None:
            extract_mask = inpaint_mask[:, 0].bool()
        else:
            extract_mask = rearrange(valid_mask, "b c h w -> (w h b) c")[:, 0].bool()
        additional_points_3d = points_3d[extract_mask]

        # original_points_3d = points_3d[~inpaint_mask[:, 0]]
        # save_point_cloud_as_ply(original_points_3d, "tmp/original_points_3d.ply", colors[~inpaint_mask[:, 0]])
        # save_point_cloud_as_ply(additional_points_3d, "tmp/additional_points_3d.ply", colors[inpaint_mask[:, 0]])

        additional_colors = colors[extract_mask]
        additional_foreground_masks = foreground_mask[extract_mask]

        # remove additional points that are behind the camera
        backward_points = (- additional_points_3d[..., 2]) > current_camera.tz
        additional_points_3d = additional_points_3d[~backward_points]
        additional_colors = additional_colors[~backward_points]
        additional_foreground_masks = additional_foreground_masks[~backward_points]
        
        # import ipdb; ipdb.set_trace()
        print("update before:", len(self.additional_colors), len(self.additional_video_colors[0]), len(self.additional_points_3d), len(self.additional_video_points_3d[0]))
        inpaint_mask_video = rearrange(inpaint_mask_video, "b c h w -> b (w h) c")
        colors_video = rearrange(video_frames, "b c h w -> b (w h) c")
        foreground_masks_video = rearrange(foreground_mask_video, "b c h w -> b (w h) c")
        for i in range(len(video_frames)):
            points_2d_frame = self.points if points_2d is None else points_2d
            points_3d_frame = current_camera.unproject(points_2d_frame, rearrange(rendered_depth_video_filled[i:i+1], "b c h w -> (w h b) c"))
            points_3d_frame[..., :2] = - points_3d_frame[..., :2]
            if valid_mask_video is None:
                extract_mask_frame = inpaint_mask_video[i][:, 0].bool()
            else:
                extract_mask_frame = rearrange(valid_mask_video[i:i+1], "b c h w -> (w h b) c")[:, 0].bool()
            
            if self.config['use_dynamic_point_cloud']:
                additional_points_3d_frame = points_3d_frame[extract_mask_frame]
                additional_colors_frame = colors_video[i][extract_mask_frame]
                additional_foreground_masks_frame = foreground_masks_video[i][extract_mask_frame]
            else:
                additional_points_3d_frame = points_3d_frame[extract_mask]
                additional_colors_frame = colors_video[i][extract_mask]
            # remove additional points that are behind the camera
            backward_points_frame = (- additional_points_3d_frame[..., 2]) > current_camera.tz
            additional_points_3d_frame = additional_points_3d_frame[~backward_points_frame]
            additional_colors_frame = additional_colors_frame[~backward_points_frame]
            additional_foreground_masks_frame = additional_foreground_masks_frame[~backward_points_frame]
            # if self.config['use_dynamic_point_cloud']:
            self.additional_video_colors[i] = torch.cat([self.additional_video_colors[i], additional_colors_frame], dim=0)
            self.additional_video_points_3d[i] = torch.cat([self.additional_video_points_3d[i], additional_points_3d_frame], dim=0)
            self.additional_video_foreground_masks[i] = torch.cat([self.additional_video_foreground_masks[i], additional_foreground_masks_frame], dim=0)
            # else:
            #     self.additional_video_colors[i] = torch.cat([self.additional_video_colors[i], additional_colors], dim=0)
            #     self.additional_video_points_3d[i] = torch.cat([self.additional_video_points_3d[i], additional_points_3d], dim=0)
            

        
        # if video_frames is not None:
        #     for i, frame in enumerate(video_frames):
        #         colors = rearrange(frame, "c h w -> (w h) c")
        #         additional_colors_video = colors[extract_mask]
        #         additional_colors_video = additional_colors_video[~backward_points]
        #         self.additional_video_colors[i] = torch.cat([self.additional_video_colors[i], additional_colors_video], dim=0)
        # else:self.kf1_dynamic_foreground_points
        #     for i in range(len(self.additional_video_colors)):
        #         self.additional_video_colors[i] = torch.cat([self.additional_video_colors[i], additional_colors], dim=0)

        self.additional_points_3d = torch.cat([self.additional_points_3d, additional_points_3d], dim=0)
        self.additional_colors = torch.cat([self.additional_colors, additional_colors], dim=0)
        self.additional_foreground_masks = torch.cat([self.additional_foreground_masks, additional_foreground_masks], dim=0)
        print("update after:", len(self.additional_colors), len(self.additional_video_colors[0]), len(self.additional_points_3d), len(self.additional_video_points_3d[0]))
        if append_depth:
            import ipdb; ipdb.set_trace()
            self.depths.append(rendered_depth_filled.cpu())
            for i in range(len(self.depth_video)):
                self.depths_video[-1][i].append(rendered_depth_video_filled[i].cpu())
                
    @torch.no_grad()
    def update_additional_point_cloud_from_interpolation(self, epoch, rendered_depth, image, foreground_mask, valid_mask=None, camera=None, points_2d=None, append_depth=False):
        """
        args:
            rendered_depth: Depth relative to camera. Note that KF2 camera is represented in KF1 camera-centered coord frame.
            valid_mask: if None, then use inpaint_mask (given by rendered_depth == 0) to extract new points.
                if not None, then just valid_mask to extract new points.
        return:
            Does not really return anything, but updates the following attributes:
            - additional_points_3d: 3D points in KF1 camera-centered coord frame.
            - additional_colors: corresponding colors
        """

        inpaint_mask = rendered_depth == 0
        rendered_depth_filled = rendered_depth.clone()
        inpaint_mask_onthefly = inpaint_mask.clone()
        epoch = epoch % len(self.additional_video_colors)

        def nearest_neighbor_inpainting(inpaint_mask, rendered_depth, window_size=20):
            """
            Perform nearest neighbor inpainting with a local search window.

            Parameters:
            inpaint_mask (torch.Tensor): Binary mask indicating missing values.
            rendered_depth (torch.Tensor): Input depth image.
            window_size (int): Size of the local search window.

            Returns:
            torch.Tensor: Inpainted depth image.
            """

            # Step 1: Find coordinates of invalid and valid pixels
            invalid_coords = torch.nonzero(inpaint_mask.squeeze(), as_tuple=False)
            valid_coords = torch.nonzero(~inpaint_mask.squeeze(), as_tuple=False)

            # Step 4: Use indices to copy depth values from valid to invalid pixels
            rendered_depth_copy = rendered_depth.clone()

            # Define half window size
            hw = window_size // 2

            # Iterate through invalid coordinates
            for idx in range(invalid_coords.size(0)):
                x, y = invalid_coords[idx, 0], invalid_coords[idx, 1]

                # Define local search window
                x_start, x_end = max(0, x - hw), min(rendered_depth.size(2), x + hw + 1)
                y_start, y_end = max(0, y - hw), min(rendered_depth.size(3), y + hw + 1)

                # Extract valid coordinates within the window
                local_valid_coords = valid_coords[(valid_coords[:, 0] >= x_start) & (valid_coords[:, 0] < x_end) & 
                                                (valid_coords[:, 1] >= y_start) & (valid_coords[:, 1] < y_end)]

                # Compute distances and find nearest neighbor
                if local_valid_coords.size(0) > 0:
                    dists = torch.cdist(invalid_coords[idx, :].unsqueeze(0).float(), local_valid_coords.float())
                    min_idx = torch.argmin(dists)
                    rendered_depth_copy[0, 0, x, y] = rendered_depth[0, 0, local_valid_coords[min_idx, 0], local_valid_coords[min_idx, 1]]

            return rendered_depth_copy

        while inpaint_mask_onthefly.sum() > 0:  # iteratively inpaint depth until all depth holes are filled
            rendered_depth_filled = nearest_neighbor_inpainting(inpaint_mask_onthefly, rendered_depth_filled, window_size=50)
            inpaint_mask_onthefly = rendered_depth_filled == 0

        current_camera = convert_pytorch3d_kornia(self.current_camera, self.config["init_focal_length"]) if camera is None else camera
        points_2d = self.points if points_2d is None else points_2d
        points_3d = current_camera.unproject(points_2d, rearrange(rendered_depth_filled, "b c h w -> (w h b) c"))
        points_3d[..., :2] = - points_3d[..., :2]
        inpaint_mask = rearrange(inpaint_mask, "b c h w -> (w h b) c")
        colors = rearrange(image, "b c h w -> (w h b) c")
        foregrounds = rearrange(foreground_mask, "b c h w -> (w h b) c")
        if valid_mask is None:
            extract_mask = inpaint_mask[:, 0].bool()
        else:
            extract_mask = rearrange(valid_mask, "b c h w -> (w h b) c")[:, 0].bool()
        additional_points_3d = points_3d[extract_mask]

        # original_points_3d = points_3d[~inpaint_mask[:, 0]]
        # save_point_cloud_as_ply(original_points_3d, "tmp/original_points_3d.ply", colors[~inpaint_mask[:, 0]])
        # save_point_cloud_as_ply(additional_points_3d, "tmp/additional_points_3d.ply", colors[inpaint_mask[:, 0]])

        additional_colors = colors[extract_mask]
        additional_foregrounds = foregrounds[extract_mask]

        # remove additional points that are behind the camera
        backward_points = (- additional_points_3d[..., 2]) > current_camera.tz
        additional_points_3d = additional_points_3d[~backward_points]
        additional_colors = additional_colors[~backward_points]
        additional_foregrounds = additional_foregrounds[~backward_points]
        # import ipdb; ipdb.set_trace()
        print("update before:", len(self.additional_colors), len(self.additional_video_colors[epoch]), len(self.additional_points_3d), len(self.additional_video_points_3d[epoch]))
        
        if self.config['use_dynamic_point_cloud']:
            self.additional_video_colors[epoch] = torch.cat([self.additional_video_colors[epoch], additional_colors], dim=0)
            self.additional_video_points_3d[epoch] = torch.cat([self.additional_video_points_3d[epoch], additional_points_3d], dim=0)
            self.additional_video_foreground_masks[epoch] = torch.cat([self.additional_video_foreground_masks[epoch], additional_foregrounds], dim=0)
        else:
            for i in range(len(self.additional_video_points_3d)):
                self.additional_video_colors[i] = torch.cat([self.additional_video_colors[i], additional_colors], dim=0)
                self.additional_video_points_3d[i] = torch.cat([self.additional_video_points_3d[i], additional_points_3d], dim=0)
                self.additional_video_foreground_masks[i] = torch.cat([self.additional_video_foreground_masks[i], additional_foregrounds], dim=0)
                

        
        # if video_frames is not None:
        #     for i, frame in enumerate(video_frames):
        #         colors = rearrange(frame, "c h w -> (w h) c")
        #         additional_colors_video = colors[extract_mask]
        #         additional_colors_video = additional_colors_video[~backward_points]
        #         self.additional_video_colors[i] = torch.cat([self.additional_video_colors[i], additional_colors_video], dim=0)
        # else:
        #     for i in range(len(self.additional_video_colors)):
        #         self.additional_video_colors[i] = torch.cat([self.additional_video_colors[i], additional_colors], dim=0)

        self.additional_points_3d = torch.cat([self.additional_points_3d, additional_points_3d], dim=0)
        self.additional_colors = torch.cat([self.additional_colors, additional_colors], dim=0)
        print("update after:", len(self.additional_colors), len(self.additional_video_colors[epoch]), len(self.additional_points_3d), len(self.additional_video_points_3d[epoch]))
        if append_depth:
            rendered_depth_filled_video =  self.depths_video[-1]
            rendered_depth_filled_video[epoch] = rendered_depth_filled[0].cpu()
            self.depths.append(rendered_depth_filled.cpu())
            self.depths_video.append(rendered_depth_filled_video)



    @torch.no_grad()
    def update_additional_point_depth(self, inconsistent_point_index, inconsistent_point_index_video, depth, mask, depth_video, mask_video):
        h, w = depth.shape[2:]
        depth = rearrange(depth.clone(), "b c h w -> (w h b) c")
        depth_video = rearrange(depth_video.clone(), "b c h w -> b (w h) c")
        extract_mask = rearrange(mask, "b c h w -> (w h b) c")[:, 0].bool()
        extract_mask_video = rearrange(mask_video, "b c h w -> b (w h) c")[:, :, 0].bool()
        depth_extracted = depth[extract_mask]
        if self.config['use_dynamic_point_cloud']:
            depth_extracted_video = [depth_video[i][extract_mask_video[i]] for i in range(len(depth_video))]
        else:
            depth_extracted_video = [depth_video[i][extract_mask] for i in range(len(depth_video))]
            
        if inconsistent_point_index.shape[0] > 0:
            assert depth_extracted.shape[0] >= inconsistent_point_index.max() + 1
            for i in range(len(depth_extracted_video)):
                if inconsistent_point_index_video[i].shape[0] > 0:
                    assert depth_extracted_video[i].shape[0] >= inconsistent_point_index_video[i].max() + 1
        depth_extracted[inconsistent_point_index] = self.config['sky_hard_depth'] * 2
        depth[extract_mask] = depth_extracted
        depth = rearrange(depth, "(w h b) c -> b c h w", w=w, h=h)
        
        for i in range(len(depth_extracted_video)):
            depth_extracted_video[i][inconsistent_point_index_video[i]] = self.config['sky_hard_depth'] * 2
            if self.config['use_dynamic_point_cloud']:
                depth_video[i][extract_mask_video[i]] = depth_extracted_video[i]
            else:
                depth_video[i][extract_mask] = depth_extracted_video[i]
        depth_video = rearrange(depth_video, "b (w h) c -> b c h w", w=w, h=h)
        return depth, depth_video

    @torch.no_grad()
    def reset_additional_point_cloud(self):
        self.additional_colors = torch.tensor([]).cuda()
        self.additional_points_3d = torch.tensor([]).cuda()
        self.additional_foreground_masks = torch.tensor([]).cuda()
        self.additional_video_colors = [x.cpu() for x in self.additional_video_colors]
        self.additional_video_colors = [torch.tensor([]).cuda() for x in range(len(self.kf2_video_frames))]
        # self.additional_video_colors = torch.stack(self.additional_video_colors)
        self.additional_video_points_3d = [x.cpu() for x in self.additional_video_points_3d]
        self.additional_video_points_3d = [torch.tensor([]).cuda() for x in range(len(self.kf2_video_frames))]
        self.additional_video_foreground_masks = [x.cpu() for x in self.additional_video_foreground_masks]
        self.additional_video_foreground_masks = [torch.tensor([]).cuda() for x in range(len(self.kf2_video_frames))]
        # self.additional_video_points_3d = torch.stack(self.additional_video_points_3d)
        


def get_extrinsics(camera):
    extrinsics = torch.cat([camera.R[0], camera.T.T], dim=1)
    padding = torch.tensor([[0, 0, 0, 1]], device=extrinsics.device)
    extrinsics = torch.cat([extrinsics, padding], dim=0)
    return extrinsics

def save_point_cloud_as_ply(points, filename="output.ply", colors=None):
    """
    Save a PyTorch tensor of shape [N, 3] as a PLY file. Optionally with colors.
    
    Parameters:
    - points (torch.Tensor): The point cloud tensor of shape [N, 3].
    - filename (str): The name of the output PLY file.
    - colors (torch.Tensor, optional): The color tensor of shape [N, 3] with values in [0, 1]. Default is None.
    """
    
    assert points.dim() == 2 and points.size(1) == 3, "Input tensor should be of shape [N, 3]."
    
    if colors is not None:
        assert colors.dim() == 2 and colors.size(1) == 3, "Color tensor should be of shape [N, 3]."
        assert points.size(0) == colors.size(0), "Points and colors tensors should have the same number of entries."
    
    # Header for the PLY file
    header = [
        "ply",
        "format ascii 1.0",
        f"element vertex {points.size(0)}",
        "property float x",
        "property float y",
        "property float z"
    ]
    
    # Add color properties to header if colors are provided
    if colors is not None:
        header.extend([
            "property uchar red",
            "property uchar green",
            "property uchar blue"
        ])
    
    header.append("end_header")
    
    # Write to file
    with open(filename, "w") as f:
        for line in header:
            f.write(line + "\n")
        
        for i in range(points.size(0)):
            line = f"{points[i, 0].item()} {points[i, 1].item()} {points[i, 2].item()}"
            
            # Add color data to the line if colors are provided
            if colors is not None:
                # Scale color values from [0, 1] to [0, 255] and convert to integers
                r, g, b = (colors[i] * 255).clamp(0, 255).int().tolist()
                line += f" {r} {g} {b}"
            
            f.write(line + "\n")

def convert_pytorch3d_kornia(camera, focal_length, size=512):
    R = torch.clone(camera.R)
    T = torch.clone(camera.T)
    T[0, 0] = -T[0, 0]
    extrinsics = torch.eye(4, device=R.device).unsqueeze(0)
    extrinsics[:, :3, :3] = R
    extrinsics[:, :3, 3] = T
    h = torch.tensor([size], device=R.device)
    w = torch.tensor([size], device=R.device)
    K = torch.eye(4)[None].to(R.device)
    K[0, 0, 2] = size // 2
    K[0, 1, 2] = size // 2
    K[0, 0, 0] = focal_length
    K[0, 1, 1] = focal_length
    return PinholeCamera(K, extrinsics, h, w)

def inpaint_cv2(rendered_image, mask_diff):
    mask = (mask_diff).float()
    rendered_image = (1 - mask) * rendered_image
    
    image_cv2 = rendered_image[0].permute(1, 2, 0).cpu().numpy()
    image_cv2 = (image_cv2 * 255).astype(np.uint8)
    mask_cv2 = mask_diff[0, 0].cpu().numpy()
    mask_cv2 = (mask_cv2 * 255).astype(np.uint8)
    inpainting = cv2.inpaint(image_cv2, mask_cv2, 3, cv2.INPAINT_TELEA)
    inpainting = torch.from_numpy(inpainting).permute(2, 0, 1).float() / 255
    return inpainting.unsqueeze(0)