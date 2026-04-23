import torch
from diffusers import CogVideoXPipeline, CogVideoXImageToVideoPipeline, CogVideoXTransformer3DModel, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from diffusers.utils import load_video, load_image
from torchvision import transforms
from torchvision.transforms import ToPILImage, ToTensor
from torchvision.transforms.functional import resize
from kornia.morphology import dilation

from PIL import Image
import os
import numpy as np
from openai import OpenAI

from PIL import Image, ImageOps
import gc
from .outpainting_utils import MaskGenerator, get_rays_np
import random
from tqdm import tqdm
import torch.nn.functional as F
import copy
from util.utils import clear_all_gpu_variables
import decord  # isort:skip
decord.bridge.set_bridge("torch")
def scale_transform(x):
    return x / 255.0
video_transforms = transforms.Compose(
            [
                # transforms.Lambda(scale_transform),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )

def image_resize_with_padding(image, desired_width, desired_height):
    """
    Resizes an image to the desired resolution with black padding if needed.

    Args:
        image_path (str): Path to the input image.
        desired_width (int): Desired width of the output image.
        desired_height (int): Desired height of the output image.
        output_path (str): Path to save the output image.
    """

    # Calculate the aspect ratio of the original and desired resolutions
    original_width, original_height = image.size
    original_aspect = original_width / original_height
    desired_aspect = desired_width / desired_height

    # Resize the image while maintaining aspect ratio
    if original_aspect > desired_aspect:
        # Fit to width
        new_width = desired_width
        new_height = int(desired_width / original_aspect)
    else:
        # Fit to height
        new_height = desired_height
        new_width = int(desired_height * original_aspect)

    resized_image = image.resize((new_width, new_height), Image.LANCZOS)

    # Create a new image with a black background and desired resolution
    new_image = Image.new("RGB", (desired_width, desired_height), (0, 0, 0))

    # Paste the resized image onto the new image centered
    paste_x = (desired_width - new_width) // 2
    paste_y = (desired_height - new_height) // 2
    new_image.paste(resized_image, (paste_x, paste_y))

    return new_image


def convert_prompt(prompt: str, retry_times: int = 3) -> str:
    if not os.environ.get("OPENAI_API_KEY"):
        return prompt
    client = OpenAI()
    text = prompt.strip()
    sys_prompt = """You are part of a team of bots that creates videos. You work with an assistant bot that will draw anything you say in square brackets.

    For example , outputting " a beautiful morning in the woods with the sun peaking through the trees " will trigger your partner bot to output an video of a forest morning , as described. You will be prompted by people looking to create detailed , amazing videos. The way to accomplish this is to take their short prompts and make them extremely detailed and descriptive.
    There are a few rules to follow:

    You will only ever output a single video description per user request.

    When modifications are requested , you should not simply make the description longer . You should refactor the entire description to integrate the suggestions.
    Other times the user will not want modifications , but instead want a new image . In this case , you should ignore your previous conversation with the user.
    
    Try to give detailed description of dynamics in the sky like moving clouds and vivid birds.

    Video descriptions must have the same num of words as examples below. Extra words will be ignored.
    """

    for i in range(retry_times):
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "a girl is on the beach"',
                },
                {
                    "role": "assistant",
                    "content": "A radiant woman stands on a deserted beach, arms outstretched, wearing a beige trench coat, white blouse, light blue jeans, and chic boots, against a backdrop of soft sky and sea. Moments later, she is seen mid-twirl, arms exuberant, with the lighting suggesting dawn or dusk. Then, she runs along the beach, her attire complemented by an off-white scarf and black ankle boots, the tranquil sea behind her. Finally, she holds a paper airplane, her pose reflecting joy and freedom, with the ocean's gentle waves and the sky's soft pastel hues enhancing the serene ambiance.",
                },
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : "A man jogging on a football field"',
                },
                {
                    "role": "assistant",
                    "content": "A determined man in athletic attire, including a blue long-sleeve shirt, black shorts, and blue socks, jogs around a snow-covered soccer field, showcasing his solitary exercise in a quiet, overcast setting. His long dreadlocks, focused expression, and the serene winter backdrop highlight his dedication to fitness. As he moves, his attire, consisting of a blue sports sweatshirt, black athletic pants, gloves, and sneakers, grips the snowy ground. He is seen running past a chain-link fence enclosing the playground area, with a basketball hoop and children's slide, suggesting a moment of solitary exercise amidst the empty field.",
                },
                {
                    "role": "user",
                    "content": 'Create an imaginative video descriptive caption or modify an earlier caption for the user input : " A woman is dancing, HD footage, close-up"',
                },
                {
                    "role": "assistant",
                    "content": "A young woman with her hair in an updo and wearing a teal hoodie stands against a light backdrop, initially looking over her shoulder with a contemplative expression. She then confidently makes a subtle dance move, suggesting rhythm and movement. Next, she appears poised and focused, looking directly at the camera. Her expression shifts to one of introspection as she gazes downward slightly. Finally, she dances with confidence, her left hand over her heart, symbolizing a poignant moment, all while dressed in the same teal hoodie against a plain, light-colored background.",
                },
                {
                    "role": "user",
                    "content": f'Create an imaginative video descriptive caption or modify an earlier caption in ENGLISH for the user input: "{text}"',
                },
            ],
            # model="glm-4-plus",
            # temperature=0.01,
            # top_p=0.7,
            # stream=False,
            # max_tokens=200,
            model="gpt-4o",
            temperature=0.01,
            top_p=0.7,
            stream=False,
            max_tokens=200,
        )
        if response.choices:
            return response.choices[0].message.content
    return prompt

def enhance_prompt_func(prompt):
    return convert_prompt(prompt, retry_times=1)


def run_inference(pretrained_diffusion_model, image, text_prompt, render_output_video=None, gpu_num=1, gpu_no=1, guidance_scale=7.0, num_inference_steps=50, seed=-1, no_prompt_guidance=False):
    device = f"cuda:{gpu_no}"
    prev_device = torch.cuda.current_device()
    # i2v_transformer = CogVideoXTransformer3DModel.from_pretrained(
    # "THUDM/CogVideoX-5b-I2V", subfolder="transformer", torch_dtype=torch.bfloat16)
    
    prompt = text_prompt
    #     prompt = '''In a vibrant scene reminiscent of a Monet painting, the bustling city streets come alive with a kale
    # idoscope of colors and movement. People, depicted in soft, impressionistic brushstrokes, stroll along the sidewalks, their f
    # orms blending into the lively urban tapestry. Cars and buses, rendered in muted tones, navigate the busy roads, their outlin
    # es slightly blurred to capture the essence of motion. The background is a symphony of activity, with towering buildings and 
    # streetlights casting dappled shadows on the pavement. The sky above is a swirl of pastel hues, with fluffy clouds drifting l
    # azily, adding a serene contrast to the energetic scene below.'''
    # if len(prompt) < 100 and not no_prompt_guidance:
    #     print("Prompt before enhancing: ", prompt)
    #     prompt = enhance_prompt_func(prompt)
    print("video diffusion prompt:", prompt)
    
    # pipe = CogVideoXPipeline.from_pretrained(
    #     "THUDM/CogVideoX-5b",
    #     torch_dtype=torch.bfloat16
    # )

    # pipe.enable_model_cpu_offload(device=device)
    # pipe.vae.enable_tiling()
    # pipe_image = CogVideoXImageToVideoPipeline.from_pretrained(
    #     "THUDM/CogVideoX-5b-I2V",
    #     transformer=i2v_transformer,
    #     vae=pipe.vae,
    #     scheduler=pipe.scheduler,
    #     tokenizer=pipe.tokenizer,
    #     text_encoder=pipe.text_encoder,
    #     torch_dtype=torch.bfloat16,
    # )
    pipe_image = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16)
    # pipe_image = CogVideoXImageToVideoPipeline.from_pretrained("THUDM/CogVideoX1.5-5B-I2V", torch_dtype=torch.bfloat16)
    # render_output_video = {}
    if (pretrained_diffusion_model != "None") and (pretrained_diffusion_model is not None) and (render_output_video is not None):
        # pipe_image.scheduler = CogVideoXDPMScheduler.from_config(pipe_image.scheduler.config)
        # These changes will also be required when trying to run inference with the trained lora
        del pipe_image.transformer.patch_embed.pos_embedding
        pipe_image.transformer.patch_embed.use_learned_positional_embeddings = False
        pipe_image.transformer.config.use_learned_positional_embeddings = False
        
        print("load lora weights from: ", pretrained_diffusion_model)
        lora_scaling = 128 / 256
        pipe_image.load_lora_weights(pretrained_diffusion_model, adapter_name="cogvideox-lora")
        pipe_image.set_adapters(["cogvideox-lora"], [lora_scaling])
        
        height, width = 512, 512
        guidance_scale = 7.0
        use_dynamic_cfg = True
        num_frames = 16
        image_input = image.resize(size=(height, width))
        
        # mask_generator = MaskGenerator(height, width)
        
        # prompt = "a serene scene of a waterfall cascading over a rocky cliff into a turquoise river. The waterfall, located on the left side of the scene, is composed of multiple cascades, with the water appearing white and frothy as it rushes down the cliff. The river, exhibiting a bright turquoise color, flows from the bottom right to the top left of the scene. The river is nestled amidst an abundance of green trees and shrubs, adding a touch of nature's vibrancy to the scene. Above, the sky is a clear blue, providing a beautiful contrast to the lush greenery below. The scene does not contain any discernible text or countable objects, and there are no visible actions taking place other than the waterfall cascading and the river flowing. The relative positions of the objects are such that the waterfall is upstream from the river, and the trees and shrubs surround both the river and the waterfall. The scene does not contain any discernible text or countable objects, and there are no visible actions taking place other than the waterfall cascading and the river flowing. The relative positions of the objects are such that the waterfall is upstream from the river, and the trees and shrubs surround both the river and the waterfall. The scene does not contain any discernible text or countable objects, and there are no visible actions taking place other than the waterfall cascading and the river flowing. The relative positions of the objects are such that the waterfall is upstream from the river, and the trees and shrubs surround both the river and the waterfall. The scene does not contain any discernible text or countable objects, and there are no visible actions taking place other than the waterfall cascading and the river flowing. The relative positions of the objects are such that the waterfall is upstream from the river, and the trees and shrubs surround both the river and the waterfall. The scene does not contain any discernible text or countable objects, and there are no visible actions taking place other than the waterfall cascading and the river flowing. The relative positions of the objects are such that the waterfall is upstream from the river, and the trees and shrubs surround both the river and the waterfall. The scene does not contain any discernible text or countable objects, and there are no visible actions taking place other than the water"
        # video_path = "/home/tianfr/data/cogvideox-factory/openvid/videos/bb0d5d30-3c93-441c-9ab0-89cc5e3ad84e.mp4"
        # video_path = "/home/tianfr/data/WonderLive/output/cogvideo_outpainting/06_city_monet_1_checkpoint-3000/Gen-13-12_19-29-00_Style:_Monet_painting._Entities:_people,/videos/inpaint_input_video/1.mp4"
        # video_reader = decord.VideoReader(uri=video_path)
        # indices = list(range(16))
        # frames = video_reader.get_batch(indices)
        # frames = frames[: 16].float() / 255.
        # frames = frames.permute(0, 3, 1, 2).contiguous()
        # # frames, mask = mask_generator.apply_mask(frames)
        # masked_video = frames.to(device)
        
        render_output_video["rendered_image"] = render_output_video["rendered_image"].to(torch.bfloat16)
        masked_video = render_output_video["rendered_image"].clone()
        # # masked_video[0] = ToTensor()(image_input)
        masked_video = torch.stack([video_transforms(frame) for frame in masked_video], dim=0)
        masked_video = masked_video.unsqueeze(0).permute(0, 2, 1, 3, 4) #(B, C, F, H, W)

        if False:
            # Ray-to-point distance pseudo-depth estimation for masked pixels.
            # For each masked pixel ray, find the closest visible 3D point and use
            # its projection depth as the pseudo-depth estimate. Mirrors the same
            # if False block in MaskGenerator_be_your_outpainter.apply_mask().
            #
            # Requires render_output_video to contain:
            #   "disparity"    : torch.Tensor (F, 1, H, W)  – per-frame disparity maps
            #   "inpaint_mask" : torch.Tensor (F, 1, H, W)  – True where pixels are masked
            focal_r = 500
            c2w_r = np.eye(4)
            H_r, W_r = 512, 512
            scale = 8
            rays_o_r, rays_d_r = get_rays_np(H_r, W_r, focal_r, c2w_r)
            rays_d_torch = F.normalize(
                torch.from_numpy(rays_d_r).float(), p=2, dim=-1
            )  # (H, W, 3)

            disparity = render_output_video["disparity"].unsqueeze(0)      # (1, F, 1, H, W)
            video_mask = render_output_video["inpaint_mask"].unsqueeze(0).bool()  # (1, F, 1, H, W)

            depth_map = 1.0 / disparity.clamp(min=1e-6)  # (1, F, 1, H, W)
            B_r, F_r, C_r, H_r2, W_r2 = depth_map.shape

            # Downsample for efficiency
            downsampled_depth_map = F.interpolate(
                depth_map.view(B_r * F_r, C_r, H_r2, W_r2),
                size=(H_r // scale, W_r // scale), mode='area'
            ).view(B_r, F_r, C_r, H_r // scale, W_r // scale)

            downsampled_video_mask = F.interpolate(
                video_mask.float().view(B_r * F_r, 1, H_r2, W_r2),
                size=(H_r // scale, W_r // scale), mode='nearest'
            ).view(B_r, F_r, 1, H_r // scale, W_r // scale)

            downsampled_rays_d = F.interpolate(
                rays_d_torch.permute(2, 0, 1).unsqueeze(0),  # (1, 3, H, W)
                size=(H_r // scale, W_r // scale), mode='nearest'
            ).squeeze(0).permute(1, 2, 0)  # (H//scale, W//scale, 3)

            masked_depth = disparity.clone()
            pseudo_depth_map = torch.zeros_like(masked_depth)

            for fi in range(F_r):
                visible_mask = (1 - downsampled_video_mask[0, fi, 0]).bool()
                point_positions = (
                    downsampled_depth_map[0, fi, 0][visible_mask][..., None]
                    * downsampled_rays_d[visible_mask]
                )  # (N_visible, 3)

                query_mask = video_mask[0, fi, 0]  # (H, W)
                pseudo_depth = rays_d_torch[query_mask] @ point_positions.permute(1, 0)  # (N_masked, N_visible)
                distance = (point_positions ** 2).sum(-1)[None] - pseudo_depth ** 2    # (N_masked, N_visible)
                min_dis_index = torch.min(distance, dim=-1)[1]
                mid_dis_depth = pseudo_depth.gather(dim=1, index=min_dis_index.unsqueeze(1)).squeeze(1)
                pseudo_depth_map[0, fi, 0][query_mask] = mid_dis_depth

            denom = pseudo_depth_map.max() - pseudo_depth_map.min()
            pseudo_depth_map = (pseudo_depth_map - pseudo_depth_map.min()) / denom.clamp(min=1e-8)
            masked_depth[video_mask] = pseudo_depth_map[video_mask]

    else:
        # image_input = image.resize(size=(1360, 768))
        height, width = 480, 720
        image_input = image.resize(size=(width, height))
        num_frames = 49
        use_dynamic_cfg = True
    
    pipe_image.vae.enable_tiling()
    pipe_image.vae.enable_slicing()
    pipe_image.enable_sequential_cpu_offload()
    pipe_image.enable_model_cpu_offload(device=device)
    print("num_frames: ", num_frames)

    # image_input = image_resize_with_padding(image, 720, 480)# Convert to PIL
    image = load_image(image_input)
    if render_output_video is not None:
        assert (pretrained_diffusion_model != "None") and (pretrained_diffusion_model is not None)
    video = pipe_image(
        image=image,
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        num_videos_per_prompt=1,
        use_dynamic_cfg=use_dynamic_cfg,
        output_type="pt",
        guidance_scale=guidance_scale,
        generator=torch.Generator(device="cpu").manual_seed(seed),
        masked_video=masked_video if render_output_video is not None else None, # (B, C, F, H, W)
        num_frames=num_frames,
        height=height,
        width=width
    ).frames[0]
    # latents = pipe_image(
    #     image=image,
    #     prompt=prompt,
    #     num_inference_steps=num_inference_steps,
    #     num_videos_per_prompt=1,
    #     use_dynamic_cfg=use_dynamic_cfg,
    #     output_type="latent",
    #     guidance_scale=guidance_scale,
    #     generator=torch.Generator(device="cpu").manual_seed(seed),
    #     masked_video=masked_video if render_output_video is not None else None, # (B, C, F, H, W)
    #     num_frames=num_frames,
    #     height=height,
    #     width=width
    # ).frames
    
    # latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
    # latents = 1 / pipe_image.vae_scaling_factor_image * latents
    
    # video = pipe_image.vae.decode(latents).sample
    # video = (video / 2 + 0.5).clamp(0, 1)
    # pipe_image.maybe_free_model_hooks()
    # video = pipe_image.video_processor.postprocess_video(video=video, output_type="pt")
    # inpainted_mask = render_output_video["merged_mask"].detach()[None, None, None].repeat(1, 3, 16, 1, 1).bfloat16().to(video.device)
    # rendered_partial_video = render_output_video["rendered_image"].detach()[None].permute(0, 2, 1, 3, 4).to(video.device)
    # with torch.enable_grad():
    #     pipe_image = finetune_decoder(pipe_image, inpainted_mask, rendered_partial_video, video, latents)
    # video = pipe_image.vae.decode(latents).sample
    # video = pipe_image.video_processor.postprocess_video(video=video, output_type="pt")[0]
    print(f"Before clearing: {torch.cuda.memory_allocated(device) / 8 / 1024 /1024} MB")
    clear_all_gpu_variables(pipe_image)
    
    # del pipe_image.vae, pipe_image.tokenizer, pipe_image.text_encoder, pipe_image.transformer, pipe_image.scheduler
    del pipe_image
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.reset_accumulated_memory_stats(device)
    gc.collect()
    
    print(f"After clearing: {torch.cuda.memory_allocated(device) / 8 / 1024 /1024} MB")
    
    # Resize each frame
    # new_size = (512, 512)
    # resized_video = torch.stack([resize(frame, new_size) for frame in video])
    resized_video =  video
    render_output_video["rendered_image"] = render_output_video["rendered_image"].float()
    # import ipdb; ipdb.set_trace()
    # import torchvision.io as io
    # video[~inpainted_mask.bool()] = rendered_partial_video[~inpainted_mask.bool()]
    # resized_video = video[0].permute(1, 2, 3, 0).cpu()  # [F, H, W, C]
    # output_file = "outpainting_video_dtype_aligned.mp4"
    # fps = 8  # Set the desired frames per second
    # io.write_video(output_file, resized_video*255, fps=fps, video_codec="libx264", options={"crf": "18"})
    # print(f"Video saved to {output_file}")
    pil_images = [ToPILImage()(frame) for frame in resized_video][:16]
    return pil_images

def finetune_decoder(pipe_image, inpainted_mask, rendered_partial_video, decoded_video, latent, n_steps=1000):
    finetune_params = []
    vae = pipe_image.vae
    vae.requires_grad_(False)
    # for i, up_block in enumerate(vae.decoder.up_blocks):
    #     for param in up_block.parameters():
    #         param.requires_grad = True
    #         finetune_params.append(param)
        
    for param in vae.decoder.norm_out.parameters():
        param.requires_grad = True
        finetune_params.append(param)
    for param in vae.decoder.conv_out.parameters():
        param.requires_grad = True
        finetune_params.append(param)
    # params = [{"params": decoder.parameters(), "lr": config["decoder_learning_rate"]}]
    params = [{"params": finetune_params, "lr": 0.0001}]
    optimizer = torch.optim.Adam(params)
    warped_mask = 1 - dilation(inpainted_mask[0], torch.ones(3*2, 3*2).to(inpainted_mask.device))[None]
    inpainted_mask = 1 - dilation(1 - inpainted_mask[0], torch.ones(3*2, 3*2).to(inpainted_mask.device))[None]
    

    for _ in tqdm(range(n_steps), leave=False):
        
        optimizer.zero_grad()
        frames = vae.decode(latent).sample
        output_video = frames / 2 + 0.5

        new_content_loss = F.mse_loss(output_video * inpainted_mask, decoded_video * inpainted_mask)
        # preservation_loss = F.mse_loss(curr_rendered_partial_frame * (1 - curr_inpainted_frame_mask), manually_decoded_frame * (1 - curr_inpainted_frame_mask)) * self.config["preservation_weight"]
        preservation_loss = F.mse_loss(output_video * warped_mask, rendered_partial_video * warped_mask) * 20
        loss = new_content_loss + preservation_loss
        print(f"(1000x) new_content_loss: {new_content_loss.item()*1000:.4f}, preservation_loss: {preservation_loss.item()*1000:.4f}, total_loss: {loss.item()*1000:.4f}")
        loss.backward()
        optimizer.step()
    del optimizer
    print(len(vae.decoder.up_blocks))
    return pipe_image