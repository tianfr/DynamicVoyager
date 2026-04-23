import torch
from tqdm import tqdm
from kornia.morphology import dilation
from torchvision.transforms import ToPILImage


def finetune_decoder(config, model, render_output, inpaint_output, n_steps=100, bg_finetuning=False):
    params = [{"params": model.vae.decoder.parameters(), "lr": config["decoder_learning_rate"]}]
    optimizer = torch.optim.Adam(params)
    decoder_ft_mask = render_output["inpaint_mask"].detach() if not bg_finetuning else render_output["inpaint_mask_bg"].detach()
    ToPILImage()(decoder_ft_mask[0]).save(model.run_dir / 'images' / 'decoder_ft_mask.png')
    if config['dilate_mask_decoder_ft'] > 1:
        decoder_ft_mask_dilated = dilation(decoder_ft_mask, torch.ones(config['dilate_mask_decoder_ft'], config['dilate_mask_decoder_ft']).to('cuda'))
    else:
        decoder_ft_mask_dilated = decoder_ft_mask
    ToPILImage()(decoder_ft_mask_dilated[0]).save(model.run_dir / 'images' / 'decoder_ft_mask_dilated.png')
    for _ in tqdm(range(n_steps), leave=False):
        optimizer.zero_grad()
        loss = model.finetune_decoder_step(
            inpaint_output["inpainted_image"].detach(),
            inpaint_output["latent"].detach(),
            render_output["rendered_image"].detach() if not bg_finetuning else render_output["rendered_image_bg"].detach(),
            decoder_ft_mask,
            decoder_ft_mask_dilated,
        )
        loss.backward()
        optimizer.step()

    del optimizer


def finetune_depth_model(config, model, target_depth, epoch, mask_align=None, mask_cutoff=None, cutoff_depth=None):
    params = [{"params": model.depth_model.parameters(), "lr": config["depth_model_learning_rate"]}]
    optimizer = torch.optim.Adam(params)

    if mask_align is None:
        mask_align = target_depth > 0

    for _ in tqdm(range(config["num_finetune_depth_model_steps"]), leave=False):
        optimizer.zero_grad()

        loss = model.finetune_depth_model_step(
            target_depth,
            model.images[epoch],
            mask_align=mask_align,
            mask_cutoff=mask_cutoff,
            cutoff_depth=cutoff_depth,
        )
        try:
            loss.backward()
            optimizer.step()
        except RuntimeError:
            print('No valid pixels to compute depth fine-tuning loss. Skip this step.')
            return
        
def finetune_video_depth_model(config, model, target_depth, epoch, mask_align=None, mask_cutoff=None, cutoff_depth=None):
    params = [{"params": model.depth_model.parameters(), "lr": config["depth_model_learning_rate"]}]
    optimizer = torch.optim.Adam(params)

    if mask_align is None:
        mask_align = target_depth > 0
    n_frames = len(model.videos[epoch])
    for _ in tqdm(range(config["num_finetune_depth_model_steps"]), leave=False):
        optimizer.zero_grad()
        i = torch.randint(16, (1,)).item()
        loss = model.finetune_depth_model_step(
            target_depth[i:i+1],
            model.videos[epoch][i:i+1],
            mask_align=mask_align[i:i+1],
            mask_cutoff=mask_cutoff[i:i+1],
            cutoff_depth=cutoff_depth,
        )
        try:
            loss.backward()
            optimizer.step()
        except RuntimeError:
            print('No valid pixels to compute depth fine-tuning loss. Skip this step.')
            return
