import torch
import random
import matplotlib.pyplot as plt
import os

from PIL import Image, ImageFilter
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import decord  # isort:skip
import PIL
import cv2
from torchvision.transforms import ToPILImage, ToTensor
import torch.nn.functional as torch_F
import io

decord.bridge.set_bridge("torch")

def save_depth_map(depth_map, file_name, vmin=None, vmax=None, save_clean=False):
    depth_map = np.squeeze(depth_map)
    if depth_map.ndim != 2:
        raise ValueError("Depth map after squeezing must be 2D.")

    dpi = 100  # Adjust this value if necessary
    figsize = (depth_map.shape[1] / dpi, depth_map.shape[0] / dpi)  # Width, Height in inches

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    cax = ax.imshow(depth_map, cmap='viridis', vmin=vmin, vmax=vmax)

    if not save_clean:
        # Standard save with labels and color bar
        cbar = fig.colorbar(cax)
        ax.set_title("Depth Map")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")
    else:
        # Clean save without labels, color bar, or axis
        plt.axis('off')
        ax.set_aspect('equal', adjustable='box')

    # Save to buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    buf.seek(0)
    img = Image.open(buf)
    img = img.convert('RGB')  # Convert to RGB
    img = img.resize((depth_map.shape[1], depth_map.shape[0]), Image.LANCZOS)  # Resize to original dimensions
    img.save(file_name, format='png')
    buf.close()
    plt.close()


def cv2_telea(img, mask, radius=5):
    ret = cv2.inpaint(img, mask, radius, cv2.INPAINT_TELEA)
    return ret, mask

# mask_config:
#     mask_l: [0., 0.4]
#     mask_r: [0., 0.4]
#     mask_t: [0., 0.4]
#     mask_b: [0., 0.4]


def get_rays_np(H, W, focal, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1)
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d


    
class MaskGenerator_be_your_outpainter:
    def __init__(self, mask_l=[0., 0.4], mask_r=[0., 0.4], mask_t=[0., 0.4], mask_b=[0., 0.4], hold_input_image=True, use_r2p_distance=False, image2video_mode=False) -> None:
        self.mask_l = mask_l
        self.mask_r = mask_r
        self.mask_t = mask_t
        self.mask_b = mask_b
        self.hold_input_image = hold_input_image
        
        self.use_r2p_distance = use_r2p_distance
        focal = 500
        c2w = np.eye(4)
        H, W = 512, 512
        rays_o, rays_d = get_rays_np(H, W, focal, c2w)
        self.rays_d_torch = torch_F.normalize(torch.from_numpy(rays_d).float(), p=2, dim=-1)
        self.image2video_mode = image2video_mode

    # def __call__(self, control):
    #     mask = -torch.ones_like(control)
    #     b, c, f, h, w = mask.shape

    #     l = np.random.rand() * (self.mask_l[1] - self.mask_l[0]) + self.mask_l[0]
    #     r = np.random.rand() * (self.mask_r[1] - self.mask_r[0]) + self.mask_r[0]
    #     t = np.random.rand() * (self.mask_t[1] - self.mask_t[0]) + self.mask_t[0]
    #     b = np.random.rand() * (self.mask_b[1] - self.mask_b[0]) + self.mask_b[0]
    #     l, r, t, b = int(l * w), int(r * w), int(t * h), int(b * h)

    #     if r == 0 and b == 0:
    #         mask[..., t:, l:] = control[..., t:, l:]
    #     elif b == 0:
    #         mask[..., t:, l:-r] = control[..., t:, l:-r]
    #     elif r == 0:
    #         mask[..., t:-b, l:] = control[..., t:-b, l:]
    #     else:
    #         mask[..., t:-b, l:-r] = control[..., t:-b, l:-r]

    #     return mask
    
    def apply_mask(self, video_frames, eval_mode=False, return_control=False, disparity=None):
        """
        Apply a randomly selected mask to the video frames.

        Args:
            video_frames (torch.Tensor): A video tensor of shape (B, C, F, H, W).

        Returns:
            torch.Tensor: The masked video tensor of the same shape as the input.
        """
        single_video = False
        if len(video_frames.shape) == 4:
            video_frames = video_frames.unsqueeze(0)
            single_video = True
        if disparity is not None:
            disparity = disparity.unsqueeze(0)

        B, F, C, H, W = video_frames.shape
        assert C == 3
        # if not eval_mode:
        #     mask = self.masks[0]
        # else:
        #     mask = self.masks[1]
        mask = torch.ones_like(video_frames)
        # mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # Shape (1, 1, 1, H, W)
        b, f, c, h, w = mask.shape

        l = np.random.rand() * (self.mask_l[1] - self.mask_l[0]) + self.mask_l[0]
        r = np.random.rand() * (self.mask_r[1] - self.mask_r[0]) + self.mask_r[0]
        t = np.random.rand() * (self.mask_t[1] - self.mask_t[0]) + self.mask_t[0]
        b = np.random.rand() * (self.mask_b[1] - self.mask_b[0]) + self.mask_b[0]
        l, r, t, b = int(l * w), int(r * w), int(t * h), int(b * h)

        if r == 0 and b == 0:
            mask[..., t:, l:] = 0
        elif b == 0:
            mask[..., t:, l:-r] = 0
        elif r == 0:
            mask[..., t:-b, l:] = 0
        else:
            mask[..., t:-b, l:-r] = 0

        mask_blurred = ToPILImage()(mask[0,0])
        mask_blurred = mask_blurred.filter(ImageFilter.GaussianBlur(20))
        # mask_blurred.save("mask_blurred.png")
        mask_blurred = ToTensor()(mask_blurred).unsqueeze(0).unsqueeze(0)
        mask_blurred = mask_blurred.repeat(B, F, 1, 1, 1)  # Broadcast to (B, C, F, H, W)
        
        if eval_mode:
            mask_3d_path = "/home/tianfr/data/WonderLive/output/cogvideo_outpainting/06_city_monet_1_checkpoint-3000/Gen-16-12_11-15-29_Style:_Monet_painting._Entities:_people,/images/masks/1.png"
            mask_3d_path = "/home/tianfr/data/WonderLive/output/cogvideo_outpainting/06_city_monet_1_consistency/Gen-17-12_15-21-18_Style:_Monet_painting._Entities:_people,/images/masks/1.png"
            mask_3d_path = "/home/tianfr/data/WonderLive/output/cogvideo_outpainting/06_city_monet_1_consistency/Gen-12-02_18-53-34_Style:_Monet_painting._cars_moving_on_th/images/masks/1.png"
            mask_3d_path = "/shared_data/p_vidalr/fengruitian/WonderLive/output/rebuttal/walking_on_the_beach_human_depth_revised/Gen-11-05_14-08-42_Style:_DSLR_35mm_landscape._Entities:_se/images/masks/1.png"
            mask_3d = Image.open(mask_3d_path)
            mask_3d = ToTensor()(mask_3d).unsqueeze(0).unsqueeze(0)
            mask_3d = mask_3d.repeat(B, F, C, 1, 1)  # Broadcast to (B, C, F, H, W)
            mask = mask_3d
            mask = torch.zeros_like(mask)
        
        # Apply the mask
        masked_video_frames = video_frames.clone()
        if eval_mode:
            bg_frame = "/home/tianfr/data/WonderLive/examples/videos/people_walking_on_the_beach.png"
            bg_frame = Image.open(bg_frame).resize((512, 512), Image.LANCZOS)
            bg_frame = ToTensor()(bg_frame)
            masked_video_frames[:, 0] = bg_frame * 255
        if self.image2video_mode:
            mask = torch.ones_like(mask).bool()
        mask = mask.bool()
        
        if self.hold_input_image:
            # masked_video_frames[:, :, 1:] *= ~mask[:, :, 1:]  # Apply the inverted mask (0 keeps, 1 masks)

            #NOTE: work for solving color jitter issue
            masked_video_frames[:, 1:, :][mask[:, 1:, :]] = masked_video_frames[:, :1].repeat(1, F-1, 1, 1, 1)[mask[:, 1:, :]]  # Apply the inverted mask (0 keeps, 1 masks)
            # masked_video_frames[:, 1:][mask[:, 1:, :]] = 0  # Apply the inverted mask (0 keeps, 1 masks)
            # masked_video_frames[:, 1:] = masked_video_frames[:, 1:] * (1 - mask_blurred[:, 1:]) +  masked_video_frames[:, :1].repeat(1, F-1, 1, 1, 1) * mask_blurred[:, 1:]# Apply the inverted mask (0 keeps, 1 masks)
            video_mask = torch.zeros_like(masked_video_frames).bool()
            video_mask[:, 1:, :][mask[:, 1:, :]] = True
        else:
            masked_video_frames[mask] = 0  # Apply the inverted mask (0 keeps, 1 masks)
            video_mask = torch.zeros_like(masked_video_frames).bool()
            video_mask[mask] = True

        if disparity is not None:
            # masked_depth = 1./ disparity
            masked_depth = disparity.clone()

            if self.hold_input_image:
                masked_depth[:, 1:, :][mask[:, 1:, :1]] = masked_depth[:, :1].repeat(1, F-1, 1, 1, 1)[mask[:, 1:, :1]]  # Apply the inverted mask (0 keeps, 1 masks)
            else:
                masked_depth[mask] = 0
            # masked_depth[video_mask[:, :, :1]] = -1
            # normalize the depth map to [0, 1] across the whole video
            masked_depth = (masked_depth - masked_depth.min()) / (masked_depth.max() - masked_depth.min())
            if self.hold_input_image:
                masked_depth[:, 1:, :][mask[:, 1:, :1]] = masked_depth[:, 1:, :][mask[:, 1:, :1]]
            else:
                masked_depth[mask] = masked_depth[mask]
            if False:
                
                pseudo_depth_map = torch.zeros_like(masked_depth)
                scale = 8
                H, W = 512, 512
                depth_map = 100./disparity
                downsampled_depth_map = torch_F.interpolate(depth_map, size=(1, H //scale , W //scale), mode='area')
                downsampled_video_mask = torch_F.interpolate(video_mask.float(), size=(1, H //scale , W //scale), mode='nearest')
                downsampled_rays_d = torch_F.interpolate(self.rays_d_torch.unsqueeze(0).unsqueeze(0), size=(H //scale , W //scale, 3), mode='nearest').squeeze(0)
                for i in range(masked_video_frames.shape[1]):
                    # save_depth_map(depth_map[i], f"depth_map_{i}.png", vmin=0, vmax=None, save_clean=True)
                    point_positions = downsampled_depth_map[0, i][(1 - downsampled_video_mask[0, i]).bool()][..., None] * downsampled_rays_d[(1 - downsampled_video_mask[0, i]).bool()]
                    pseudo_depth = self.rays_d_torch.unsqueeze(0)[video_mask[0, i, :1]] @ point_positions.permute(1, 0)
                    distance = (point_positions**2).sum(-1)[None] - pseudo_depth ** 2
                    # pseudo_depth_map[0, i][video_mask[i]] = torch.min(distance, dim=-1)[0]
                    min_dis_index = torch.min(distance, dim=-1)[1]
                    mid_dis_depth = pseudo_depth.gather(dim=1, index=min_dis_index.unsqueeze(1)).squeeze(1)
                    pseudo_depth_map[0, i][video_mask[0, i, :1]] = mid_dis_depth
                    # pseudo_depth_map[0, i][video_mask[0, i, :1]] = torch.min(distance, dim=-1)[0]
                pseudo_depth_map = (pseudo_depth_map - pseudo_depth_map.min()) / (pseudo_depth_map.max() - pseudo_depth_map.min())
                masked_depth[video_mask[:, :, :1]] = pseudo_depth_map[video_mask[:, :, :1]]
            # from torchvision.transforms import ToPILImage
            # ToPILImage()(self.rays_d_torch.permute(2, 0, 1)+1).save("rays_d_torch.png")
            # import ipdb; ipdb.set_trace()
            # save_depth_map(masked_depth[0, 5, 0], f"depth_map_{5}.png", vmin=0, vmax=None, save_clean=True)
        if single_video:
            masked_video_frames = masked_video_frames.squeeze(0)
            video_mask = video_mask.squeeze(0)
            if disparity is not None:
                masked_depth = masked_depth.squeeze(0)

        return_rs = (masked_video_frames, video_mask)

        if return_control:
            control = torch.zeros_like(masked_video_frames)
            control[video_mask] = -1 
            # control = control * (1 - torch.rand_like(control[1])[None] * 0.2)
            # control[~video_mask] = 0.2 * torch.rand_like(control[~video_mask])
            
            return_rs += (control,)
        
        if disparity is not None:
            return_rs += (masked_depth,)
        return return_rs
    
    def set_i2v_mode(self, mode):
        self.image2video_mode = mode

    def set_r2p_distance_mode(self, mode):
        self.use_r2p_distance = mode
    
class MaskGenerator:
    def __init__(self, height, width, hold_input_image=True, max_unmask_ratio=0.25, num_masks=100, save_dir="./masks", save_image_dir="./masks"):
        """
        Initialize the MaskGeneration class.

        Args:
            height (int): Height of the masks.
            width (int): Width of the masks.
            max_unmask_ratio (float): Maximum allowable unmasked region as a fraction of total pixels.
            num_masks (int): Number of masks to pre-generate.
            save_dir (str): Directory to save or load the masks.
            save_image_dir (str): Directory to save visualized masks as images.
        """
        self.height = height
        self.width = width
        self.hold_input_image = hold_input_image
        self.max_unmask_ratio = max_unmask_ratio
        self.num_masks = num_masks
        self.save_dir = save_dir
        self.save_image_dir = save_image_dir
        self.masks = []

        # Create save directories if they don't exist
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(save_image_dir, exist_ok=True)

        # Generate masks or load existing ones
        self.generate_or_load_masks()

    def generate_connected_mask_old(self, init_rectangle_ratio=(0.05, 0.1), position_offset=0.05):
        """
        Generate a connected mask with a random unmasked region.

        The initial rectangle is randomly positioned near middle-top, middle-left, or middle-right, 
        with added randomness for start_row and start_col.

        Args:
            init_rectangle_ratio (tuple): Fraction of (height, width) for the initial unmasked rectangle.
            position_offset (float): Maximum randomness factor as a fraction of frame dimensions.

        Returns:
            torch.Tensor: A boolean mask of shape (height, width) where True means "masked".
        """
        total_pixels = self.height * self.width
        max_unmasked_pixels = int(total_pixels * self.max_unmask_ratio)

        rect_height = int(self.height * init_rectangle_ratio[0])
        rect_width = int(self.width * init_rectangle_ratio[1])

        # Determine the base position and add randomness
        position = random.choice(["middle-top", "middle-left", "middle-right"])
        offset_h = int(self.height * position_offset)
        offset_w = int(self.width * position_offset)

        if position == "middle-top":
            base_row = self.height // 5
            base_col = (self.width // 2) - (rect_width // 2)
        elif position == "middle-left":
            base_row = (self.height // 2) - (rect_height // 2)
            base_col = self.width // 5
        elif position == "middle-right":
            base_row = (self.height // 2) - (rect_height // 2)
            base_col = (self.width * 4) // 5 - rect_width

        # Add random offsets
        start_row = base_row + random.randint(-offset_h, offset_h)
        start_col = base_col + random.randint(-offset_w, offset_w)

        # Ensure the start position is within bounds
        start_row = max(0, min(start_row, self.height - rect_height))
        start_col = max(0, min(start_col, self.width - rect_width))

        # Initialize the mask
        mask = torch.ones((self.height, self.width), dtype=torch.bool)
        frontier = []

        # Create the initial unmasked rectangle
        for r in range(start_row, start_row + rect_height):
            for c in range(start_col, start_col + rect_width):
                if 0 <= r < self.height and 0 <= c < self.width:
                    mask[r, c] = False
                    frontier.append((r, c))

        # Randomly grow the unmasked region
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        unmasked_count = len(frontier)

        while unmasked_count < max_unmasked_pixels and frontier:
            row, col = random.choice(frontier)

            for direction in random.sample(directions, len(directions)):
                new_row, new_col = row + direction[0], col + direction[1]

                if 0 <= new_row < self.height and 0 <= new_col < self.width and mask[new_row, new_col]:
                    mask[new_row, new_col] = False
                    frontier.append((new_row, new_col))
                    unmasked_count += 1

                    if unmasked_count >= max_unmasked_pixels:
                        break

            frontier.remove((row, col))

        return mask


    def generate_connected_mask(self, init_rectangle_ratio=(0.05, 0.1), position_offset=0.05, elongation_bias=0.3):
        """
        Generate a connected mask with diverse shapes using random walk.

        The initial rectangle is randomly positioned with added randomness, and the growth process
        favors creating elongated or irregular shapes, ensuring connectivity.

        Args:
            init_rectangle_ratio (tuple): Fraction of (height, width) for the initial unmasked rectangle.
            position_offset (float): Maximum randomness factor as a fraction of frame dimensions.
            elongation_bias (float): Probability of favoring growth in one direction for elongation.

        Returns:
            torch.Tensor: A boolean mask of shape (height, width) where True means "masked".
        """
        total_pixels = self.height * self.width
        max_unmasked_pixels = int(total_pixels * self.max_unmask_ratio)

        rect_height = int(self.height * init_rectangle_ratio[0])
        rect_width = int(self.width * init_rectangle_ratio[1])

        # Determine the base position and add randomness
        position = random.choice(["middle-top", "middle-left", "middle-right"])
        offset_h = int(self.height * position_offset)
        offset_w = int(self.width * position_offset)

        if position == "middle-top":
            base_row = self.height // 5
            base_col = (self.width // 2) - (rect_width // 2)
        elif position == "middle-left":
            base_row = (self.height // 2) - (rect_height // 2)
            base_col = self.width // 5
        elif position == "middle-right":
            base_row = (self.height // 2) - (rect_height // 2)
            base_col = (self.width * 4) // 5 - rect_width

        # Add random offsets
        start_row = base_row + random.randint(-offset_h, offset_h)
        start_col = base_col + random.randint(-offset_w, offset_w)

        # Ensure the start position is within bounds
        start_row = max(0, min(start_row, self.height - rect_height))
        start_col = max(0, min(start_col, self.width - rect_width))

        # Initialize the mask
        mask = torch.ones((self.height, self.width), dtype=torch.bool)
        
        # Create the initial unmasked rectangle
        mask[start_row : start_row + rect_height, start_col : start_col + rect_width] = False

        # Initialize the frontier
        frontier = [(r, c) for r in range(start_row, start_row + rect_height) for c in range(start_col, start_col + rect_width)]
        unmasked_count = len(frontier)

        # Directions for growth
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        unmasked_count = len(frontier)

        while unmasked_count < max_unmasked_pixels and frontier:
            # Choose a random point from the frontier
            row, col = random.choice(frontier)

            for direction in random.sample(directions, len(directions)):
                if random.random() < elongation_bias:
                    # Attempt to elongate in the same direction
                    for step in range(1, random.randint(2, 4)):  # Multi-step elongation
                        new_row = row + direction[0] * step
                        new_col = col + direction[1] * step

                        if (
                            0 <= new_row < self.height
                            and 0 <= new_col < self.width
                            and mask[new_row, new_col]
                        ):
                            # Ensure connectivity by unmasking intermediate steps
                            for inter_step in range(1, step + 1):
                                inter_row = row + direction[0] * inter_step
                                inter_col = col + direction[1] * inter_step
                                if (
                                    0 <= inter_row < self.height
                                    and 0 <= inter_col < self.width
                                    and mask[inter_row, inter_col]
                                ):
                                    mask[inter_row, inter_col] = False
                                    frontier.append((inter_row, inter_col))
                                    unmasked_count += 1

                                    if unmasked_count >= max_unmasked_pixels:
                                        break
                        if unmasked_count >= max_unmasked_pixels:
                            break
                else:
                    # Regular random growth
                    new_row = row + direction[0]
                    new_col = col + direction[1]

                    if 0 <= new_row < self.height and 0 <= new_col < self.width and mask[new_row, new_col]:
                        mask[new_row, new_col] = False
                        frontier.append((new_row, new_col))
                        unmasked_count += 1

                if unmasked_count >= max_unmasked_pixels:
                    break

            # Occasionally remove points from the frontier to promote branching
            if random.random() < 0.3:
                frontier.remove((row, col))

        return mask
    def _generate_and_save_mask(self, index):
        """
        Generate a single mask and save it.
        """
        mask = self.generate_connected_mask()
        self.save_mask(index, mask)
        return mask

    def generate_or_load_masks(self):
        """
        Generate masks using multiprocessing or load pre-existing ones from disk.
        """
        # Check if masks already exist
        loaded_masks = []
        mask_path = "/home/tianfr/data/WonderLive/output/cogvideo_outpainting/salient_city_at_night-3000/Gen-11-12_22-07-01_Style:_DSLR_35mm_landscape._Entities:_pa/videos/inpaint_input_video/1.mp4"
        
        mask_path = "/home/tianfr/data/WonderLive/output/cogvideo_outpainting/06_city_monet_1_checkpoint-3000/Gen-11-12_23-22-25_Style:_Monet_painting._Entities:_people,/videos/inpaint_input_video/mask1.mp4"
        mask_path = "/home/tianfr/data/WonderLive/output/cogvideo_outpainting/06_city_monet_1_checkpoint-3000/Gen-13-12_19-29-00_Style:_Monet_painting._Entities:_people,/videos/inpaint_input_video/mask1.mp4"
        mask_paths = [
            "/home/tianfr/data/WonderLive/output/cogvideo_outpainting/06_city_monet_1_checkpoint-3000/Gen-11-12_23-22-25_Style:_Monet_painting._Entities:_people,/videos/inpaint_input_video/mask1.mp4",
            "/home/tianfr/data/WonderLive/output/cogvideo_outpainting/06_city_monet_1_checkpoint-3000/Gen-13-12_19-29-00_Style:_Monet_painting._Entities:_people,/videos/inpaint_input_video/mask1.mp4"
        ]
        for mask_path in mask_paths:
            video_reader = decord.VideoReader(uri=mask_path)
            video_num_frames = len(video_reader)

            indices = list(range(0, video_num_frames, video_num_frames // 16))
            frames = video_reader.get_batch(indices)
            frames = frames[: 16] > 200
            frames = frames.permute(0, 3, 1, 2).unsqueeze(0).contiguous() # (F, C, H, W)
            # frames = frames.bool()
            # frames = mask_generator.apply_mask(frames)
            # video = frames.unsqueeze(0).permute(0, 2, 1, 3, 4) #(B, C, F, H, W)
            loaded_masks.append(frames)
        self.masks = loaded_masks
        return 
        for i in range(self.num_masks):
            mask_path = os.path.join(self.save_dir, f"mask_{i}.pt")
            if os.path.exists(mask_path):
                loaded_masks.append(torch.load(mask_path))

        if len(loaded_masks) == self.num_masks:
            print("All masks loaded from disk.")
            self.masks = loaded_masks
            return

        print(f"Generating {self.num_masks - len(loaded_masks)} masks...")

        # Generate missing masks with multiprocessing
        with Pool(processes=cpu_count()) as pool:
            # Use tqdm with imap_unordered to track progress
            new_masks = list(
                tqdm(
                    pool.imap_unordered(self._generate_and_save_mask, range(len(loaded_masks), self.num_masks)),
                    total=self.num_masks - len(loaded_masks),
                    desc="Generating Masks",
                )
            )

        # Combine loaded and newly generated masks
        self.masks = loaded_masks + new_masks
        self.save_all_masks()
            

    def save_mask(self, index, mask):
        """
        Save a single mask as a .pt file.

        Args:
            index (int): The index of the mask.
            mask (torch.Tensor): The mask tensor to save.
        """
        file_path = os.path.join(self.save_dir, f"mask_{index}.pt")
        torch.save(mask, file_path)
        print(f"Mask {index} saved to {file_path}")

    def save_all_masks(self):
        """
        Save all pre-generated masks as image files in the save_image_dir using PIL.
        """
        if not self.masks:
            print("No masks available to save.")
            return

        for i, mask in enumerate(self.masks):
            # Convert the tensor mask to a numpy array (0 or 255 values for black and white)
            mask_array = mask.numpy().astype(np.uint8) * 255

            # Convert numpy array to a PIL Image
            mask_image = Image.fromarray(mask_array)

            # Save the image to the disk
            filename = f"mask_{i}.png"
            filepath = os.path.join(self.save_image_dir, filename)
            mask_image.save(filepath)
            print(f"Mask {i} saved to {filepath}")

    # def visualize_and_save_mask(self, index=None, save_as_image=True):
    #     """
    #     Visualize and save a mask as an image from the stored masks.

    #     Args:
    #         index (int or None): Index of the mask to visualize. If None, a random mask is visualized.
    #         save_as_image (bool): Whether to save the visualized mask as an image file.
    #     """
    #     if not self.masks:
    #         print("No masks available to visualize.")
    #         return

    #     mask = self.masks[index] if index is not None else random.choice(self.masks)

    #     # Convert the tensor mask to a numpy array (0 or 255 values for black and white)
    #     mask_array = mask.numpy().astype(np.uint8) * 255

    #     # Convert numpy array to a PIL Image
    #     mask_image = Image.fromarray(mask_array)

    #     # Save the image if save_as_image is True
    #     if save_as_image:
    #         filename = f"mask_{index if index is not None else 'random'}.png"
    #         filepath = os.path.join(self.save_image_dir, filename)
    #         mask_image.save(filepath)
    #         print(f"Mask image saved to {filepath}")

    #     # Optionally show the mask
    #     # mask_image.show()

    def apply_mask(self, video_frames, eval_mode=False):
        """
        Apply a randomly selected mask to the video frames.

        Args:
            video_frames (torch.Tensor): A video tensor of shape (B, C, F, H, W).

        Returns:
            torch.Tensor: The masked video tensor of the same shape as the input.
        """
        single_video = False
        if len(video_frames.shape) == 4:
            video_frames = video_frames.unsqueeze(0)
            single_video = True
        B, F, C, H, W = video_frames.shape
        assert C == 3
        if not eval_mode:
            mask = self.masks[0]
        else:
            mask = self.masks[1]
        # mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(2)  # Shape (1, 1, 1, H, W)
        mask = mask.expand(B, F, C, H, W).to(video_frames.device)  # Broadcast to (B, C, F, H, W)
        # Apply the mask
        masked_video_frames = video_frames.clone()
        if self.hold_input_image:
            # masked_video_frames[:, :, 1:] *= ~mask[:, :, 1:]  # Apply the inverted mask (0 keeps, 1 masks)
            masked_video_frames[:, 1:, :][mask[:, 1:, :]] = 0  # Apply the inverted mask (0 keeps, 1 masks)
            video_mask = torch.ones_like(masked_video_frames).bool()
            video_mask[:, 1:, :][mask[:, 1:, :]] = False
        else:
            masked_video_frames[mask] = 0  # Apply the inverted mask (0 keeps, 1 masks)
            video_mask = torch.ones_like(masked_video_frames).bool()
            video_mask[mask] = False
        # import ipdb; ipdb.set_trace()
        if single_video:
            masked_video_frames = masked_video_frames.squeeze(0)
            video_mask = video_mask.squeeze(0)

        return masked_video_frames, video_mask

import torch


def save_video_as_gif(video_tensor, filepath, fps=10):
    """
    Save a video tensor in the format [B, C, F, H, W] as a GIF.
    
    Args:
        video_tensor (torch.Tensor): The video tensor with shape [B, C, F, H, W].
        filepath (str): The file path where the GIF will be saved.
        fps (int): Frames per second for the GIF.
    """
    import imageio
    
    # Ensure tensor is on the CPU and in the correct format
    video_tensor = video_tensor.squeeze(0).permute(1, 2, 3, 0).cpu()  # [F, H, W, C]
    
    # Scale to [0, 255] and convert to uint8
    video_frames = (video_tensor * 255).clamp(0, 255).byte().numpy()  # Convert to NumPy array
    
    # Save as GIF
    imageio.mimsave(filepath, video_frames, fps=fps)
    print(f"Saved GIF at {filepath}")

# Example usage
# Assume a video tensor with shape [1, 3, 10, 64, 64] (Batch size of 1, 10 frames, 64x64 RGB)
# video_tensor = torch.rand(1, 3, 10, 64, 64, device='cuda')  # Random video tensor
# save_video_as_gif(video_tensor, "output.gif", fps=10)




def load_video_as_tensor(filepath):
    """
    Load a video from a file into a PyTorch tensor with shape [B, C, F, H, W].

    Args:
        filepath (str): Path to the video file.

    Returns:
        torch.Tensor: Video tensor with shape [1, C, F, H, W].
    """
    from torchvision.io import read_video
    # Read the video
    video, _, _ = read_video(filepath, pts_unit="sec")  # Video is [F, H, W, C]
    
    # Permute the dimensions to [C, F, H, W]
    video = video.permute(3, 0, 1, 2)
    
    # Add batch dimension [B, C, F, H, W]
    video = video.unsqueeze(0)
    
    return video

# Example usage
video_path = "/home/tianfr/data/cogvideox-factory/openvid/videos/0ef8ed3c-9e78-477a-bf75-b485c1e89321.mp4"
video_tensor = load_video_as_tensor(video_path)
print(f"Loaded video shape: {video_tensor.shape}")