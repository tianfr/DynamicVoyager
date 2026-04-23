<p align="center">
    <img src="assets\dynamicvoyager_icon2.jpg" width="20%">
</p>
<div align="center">

# ✨Voyaging into Perpetual Dynamic Scenes from a Single View

<p align="center">
<a href="https://tianfr.github.io/">Fengrui Tian</a>,
<a href="https://tianjiaoding.com/">Tianjiao Ding</a>,
<a href="https://peterljq.github.io/">Jinqi Luo</a>,
<a href="https://hanchmin.github.io/">Hancheng Min</a>,
<a href="http://vision.jhu.edu/rvidal.html">René Vidal</a>
<br>
    University of Pennsylvania
</p>
<h3 align="center">🌟ICCV 2025🌟</h3>
<a href="https://arxiv.org/abs/2507.04183"><img src='https://img.shields.io/badge/arXiv-2507.04183-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://tianfr.github.io/project/DynamicVoyager/index.html"><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://www.youtube.com/watch?v=DycfO7DTu98"><img src='https://img.shields.io/badge/Youtube-Video-blue'></a> &nbsp;&nbsp;&nbsp;&nbsp;

<br>

<img src="assets\dynamicvoyager_teaser.gif" alt="gif1" style="flex: 1 1 20%; max-width: 98%;">
</div>
<br>


<table>
  <tr>
    <td><img src="assets\village.gif" alt="gif5" width="150"></td>
    <td><img src="assets\rose.gif" alt="gif1" width="150"></td>
    <td><img src="assets\umbrella.gif" alt="gif2" width="150"></td>
    <td><img src="assets\village1.gif" alt="gif3" width="150"></td>
    <td><img src="assets\cat.gif" alt="gif4" width="150"></td>
    <td><img src="assets\cartoon.gif" alt="gif5" width="150"></td>
    <td><img src="assets\village2.gif" alt="gif5" width="150"></td>
  </tr>
</table>



This is the official implementation of our ICCV 2025 paper "Voyaging into Perpetual Dynamic Scenes from a Single View".



## Abstract
 We study the problem of generating a perpetual dynamic scene from a single view. Since the scene is changing over time, different generated views need to be consistent with the underlying 3D motions. We propose DynamicVoyager that reformulates the dynamic scene generation as a scene outpainting process for new dynamic content. As 2D outpainting models can hardly generate 3D consistent motions from only 2D pixels at a single view, we consider pixels as rays to enrich the pixel input with the ray context, so that the 3D motion consistency can be learned from the ray information. More specifically, we first map the single-view video input to a dynamic point cloud with the estimated video depths. Then we render the partial video at a novel view and outpaint the video with ray contexts from the point cloud to generate 3D consistent motions. We employ the outpainted video to update the point cloud, which is used for scene outpainting from future novel views.

## Installation

### Requirements

- Linux (tested on RHEL 8)
- NVIDIA GPU with ≥ 40 GB VRAM (A100 / A6000 recommended)
- CUDA 12.1
- Conda

### 1. Create conda environment

```bash
conda create -n dynamicworld python=3.10 -y
conda activate dynamicworld
```

### 2. Install PyTorch (CUDA 12.1)

```bash
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install PyTorch3D

```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.5"
```

> If the build fails, download a pre-built wheel matching your environment from the [PyTorch3D releases page](https://github.com/facebookresearch/pytorch3d/releases).

### 4. Install remaining dependencies

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

---

## Model Weights

Place the following weight files in the **project root** (same folder as `run.py`):

### SAM (Segment Anything)

```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
```

### MiDaS depth model

```bash
wget https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt
```

### HuggingFace models (auto-downloaded on first run)

| Model | HuggingFace ID |
|---|---|
| CogVideoX I2V | `THUDM/CogVideoX-5b-I2V` |
| Stable Diffusion Inpainting | `stabilityai/stable-diffusion-2-inpainting` |
| OneFormer segmentation | `shi-labs/oneformer_coco_swin_large` |

### Outpainting LoRA checkpoint

Download CogVideoX outpainting LoRA checkpoint at [https://drive.google.com/file/d/1yRjOYfCLYTndDoVMKH9Wb7vCQLrwlClV/view?usp=sharing](https://drive.google.com/file/d/1yRjOYfCLYTndDoVMKH9Wb7vCQLrwlClV/view?usp=sharing). The directory should contain `pytorch_lora_weights.safetensors`. Place it into the path:
```
checkpoints/cogvideox_outpainting_lora/
```

---

## Running

### 1. Configure the LoRA path

Open a config file under `config/dynamics/` and set `pretrained_diffusion_model` to your checkpoint directory:

```yaml
pretrained_diffusion_model: "checkpoints/cogvideox_outpainting_lora"
```

### 2. (Optional) Set OpenAI API key

Some configs use GPT-4o to auto-generate prompts (`use_gpt: True`). To enable this:

```bash
export OPENAI_API_KEY="sk-..."
```

To skip GPT entirely, set `use_gpt: False` in the config.

### 3. Run

```bash
conda activate dynamicworld
python run.py --example_config config/dynamics/waterfall_cogvideo_outpainting.yaml
```

### Output

Results are written to the `runs_dir` specified in the config:

```
output/<name>/
    Gen-<timestamp>_<prompt>/
        images/         # depth maps, masks, inpainted keyframes
        videos/         # per-frame diffusion videos
    <timestamp>_merged/
        output.mp4          ← main result (looping video)
        output_reverse.mp4
```

---

## Adding Your Own Scene

**Step 1.** Add an entry to `examples/examples.yaml`:

```yaml
- name: my_scene
  image_filepath: examples/images/my_scene.png
  style_prompt: DSLR 35mm landscape
  content_prompt: Mountain valley, river, pine trees
  negative_prompt: ""
  background: A river flowing through a mountain valley
  cogvideo_prompt: "camera slowly panning across a mountain valley with a flowing river"
```

**Step 2.** Create `config/dynamics/my_scene.yaml`:

```yaml
runs_dir: output/my_scene

example_name: my_scene

seed: 42
frames: 10
num_scenes: 1
num_keyframes: 2
use_gpt: False

rotation_path: [0, 0, 0, 0, 0, 0, 0, 0]
rotation_range: 0.35
save_fps: 10

video_generation_model: "cogvideo"
pretrained_diffusion_model: "checkpoints/cogvideox_outpainting_lora"

kf1_video_path: ""
```

**Step 3.** Run:

```bash
python run.py --example_config config/dynamics/my_scene.yaml
```

---

## Key Config Options

| Option | Default | Description |
|---|---|---|
| `video_generation_model` | `"cogvideo"` | Video backbone (`"cogvideo"` or `"dynamicrafter"`) |
| `pretrained_diffusion_model` | `None` | Path to outpainting LoRA checkpoint |
| `num_scenes` | `1` | Number of camera scenes |
| `num_keyframes` | `2` | Keyframes per scene |
| `frames` | `10` | Interpolation frames between keyframes |
| `seed` | `2` | Random seed (`-1` for random) |
| `use_gpt` | `True` | Use GPT-4o to auto-generate prompts |
| `skip_gen` | `False` | Skip generation, reuse cached `.pt` files |
| `skip_interp` | `False` | Skip interpolation, only run generation |
| `finetune_decoder_gen` | `True` | Fine-tune VAE decoder during generation |
| `finetune_depth_model` | `True` | Fine-tune MiDaS per keyframe |

---

We are currently open-sourcing part of the code. The full codebase will be released progressively.

---

## Citation
```
@InProceedings{25iccv/tian_dynvoyager,
    author    = {Tian, Fengrui and Ding, Tianjiao and Luo, Jinqi and Min, Hancheng and Vidal, Ren\'e},
    title     = {Voyaging into Perpetual Dynamic Scenes from a Single View},
    booktitle = {Proceedings of the International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2025}
}
```

## Contact
If you have any questions, please feel free to contact [Fengrui Tian](https://tianfr.github.io).