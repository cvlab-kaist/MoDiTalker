import time
import sys

sys.path.extend([".", "src"])
import numpy as np
import torch
from utils import AverageMeter
from torchvision.utils import save_image, make_grid
from einops import rearrange
from losses.ddpm import DDPM

from evals.fvd.fvd import get_fvd_logits, frechet_distance
from evals.fvd.download import load_i3d_pretrained
import os

import torchvision
import PIL
import pdb
from tqdm import tqdm


def save_image_grid(img, fname, drange, grid_size, normalize=True):
    if normalize:
        lo, hi = drange
        img = np.asarray(img, dtype=np.float32)
        img = (img - lo) * (255 / (hi - lo))
        img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, T, H, W = img.shape
    img = img.reshape(gh, gw, C, T, H, W)
    img = img.transpose(3, 0, 4, 1, 5, 2)
    img = img.reshape(T, gh * H, gw * W, C)

    print(f"Saving Video with {T} frames, img shape {H}, {W}")

    assert C in [3]

    if C == 3:
        torchvision.io.write_video(f"{fname[:-3]}mp4", torch.from_numpy(img), fps=16)
        imgs = [PIL.Image.fromarray(img[i], "RGB") for i in range(len(img))]
        imgs[0].save(fname, quality=95, save_all=True, append_images=imgs[1:], duration=100, loop=0)

    return img


def test_psnr(rank, model, loader, it, logger=None, type="x"):
    device = torch.device("cuda", rank)

    losses = dict()
    losses["psnr"] = AverageMeter()
    check = time.time()

    model.eval()
    with torch.no_grad():
        for n_, (x_ref, x, x_l, masked_x, _) in enumerate(loader):
            if type == "x":
                x_l = x

            if n_ > 100:
                break
            batch_size = x_l.size(0)
            clip_length = x_l.size(1)
            x_l = x_l.to(device) / 127.5 - 1
            recon, _ = model(rearrange(x_l, "b t c h w -> b c t h w"))

            x_l = x_l.view(batch_size, -1)
            recon = recon.view(batch_size, -1)

            mse = ((x_l * 0.5 - recon * 0.5) ** 2).mean(dim=-1)
            psnr = (-10 * torch.log10(mse)).mean()

            losses["psnr"].update(psnr.item(), batch_size)

    model.train()
    return losses["psnr"].average


def test_ifvd(rank, model, loader, it, logger=None, type="x"):
    device = torch.device("cuda", rank)

    losses = dict()
    losses["fvd"] = AverageMeter()
    check = time.time()

    x_l_embeddings = []
    fake_embeddings = []
    fakes = []
    x_ls = []

    model.eval()
    i3d = load_i3d_pretrained(device)

    with torch.no_grad():
        for n_, (x_ref, x, x_l, masked_x, _) in enumerate(loader):
            if type == "x":
                x_l = x
            if n_ > 512:
                break
            batch_size = x_l.size(0)
            clip_length = x_l.size(1)
            x_l = x_l.to(device)
            fake, _ = model(rearrange(x_l / 127.5 - 1, "b t c h w -> b c t h w"))

            x_l = rearrange(x_l, "b t c h w -> b t h w c")  # videos
            fake = rearrange((fake.clamp(-1, 1) + 1) * 127.5, "(b t) c h w -> b t h w c", b=x_l.size(0))

            x_l = x_l.type(torch.uint8).cpu()
            fake = fake.type(torch.uint8)

            x_l_embeddings.append(get_fvd_logits(x_l.numpy(), i3d=i3d, device=device))
            fake_embeddings.append(get_fvd_logits(fake.cpu().numpy(), i3d=i3d, device=device))
            if len(fakes) < 16:
                x_ls.append(rearrange(x_l[0:1], "b t h w c -> b c t h w"))
                fakes.append(rearrange(fake[0:1], "b t h w c -> b c t h w"))

    model.train()

    x_ls = torch.cat(x_ls)
    fakes = torch.cat(fakes)

    if rank == 0:
        x_l_vid = save_image_grid(
            x_ls.cpu().numpy(),
            os.path.join(logger.logdir, "x_l.gif"),
            drange=[0, 255],
            grid_size=(4, 4),
        )
        fake_vid = save_image_grid(
            fakes.cpu().numpy(),
            os.path.join(logger.logdir, f"generated_{it}.gif"),
            drange=[0, 255],
            grid_size=(4, 4),
        )

        if it == 0:
            x_l_vid = np.expand_dims(x_l_vid, 0).transpose(0, 1, 4, 2, 3)
            logger.video_summary("x_l", x_l_vid, it)

        fake_vid = np.expand_dims(fake_vid, 0).transpose(0, 1, 4, 2, 3)
        logger.video_summary("recon", fake_vid, it)

    x_l_embeddings = torch.cat(x_l_embeddings)
    fake_embeddings = torch.cat(fake_embeddings)

    fvd = frechet_distance(fake_embeddings.clone().detach(), x_l_embeddings.clone().detach())
    return fvd.item()


def test_fvd_ddpm(rank, ema_model, decoder, loader, it, logger=None):
    device = torch.device("cuda", rank)
    decoder, decoder_ldmk = decoder

    losses = dict()
    losses["fvd"] = AverageMeter()
    check = time.time()

    cond_model = ema_model.diffusion_model.cond_model

    diffusion_model = DDPM(
        ema_model,
        channels=ema_model.diffusion_model.in_channels,
        image_size=ema_model.diffusion_model.image_size,
        sampling_timesteps=100,
        w=0.0,
    ).to(device)
    
    real_embeddings = []
    fake_embeddings = []
    pred_embeddings = []

    reals = []
    fakes = []
    masked_vid = []
    reals_ldmk = []
    predictions = []

    # i3d = load_i3d_pretrained(device)

    with torch.no_grad():
        for n, (x_ref, x, x_l, masked_x, _) in enumerate(loader):
            k = min(4, x.size(0))
            if n >= 4:
                break

            real = rearrange(x, "b t c h w -> b t h w c")
            real = real.type(torch.uint8).numpy()
            masked_z = decoder.extract(rearrange(masked_x[:k] / 127.5 - 1, "b t c h w -> b c t h w").to(device).detach())
            eval_ldmk = decoder_ldmk.extract(rearrange(x_l[:k] / 127.5 - 1, "b t c h w -> b c t h w").to(device).detach())
            cond = torch.cat([eval_ldmk, masked_z], dim=1)

            image_cond = decoder.extract(rearrange(x_ref[:k] / 127.5 - 1, "b t c h w -> b c t h w").to(device).detach())
            image_cond = image_cond[:, :, 0 : 32 * 32]

            z = diffusion_model.sample(batch_size=4, cond=cond, image_cond=image_cond, context=x_ref)
            fake = decoder.decode_from_sample(z).clamp(-1, 1).cpu()
            fake = (1 + rearrange(fake, "(b t) c h w -> b t h w c", b=4)) * 127.5
            fake = fake.type(torch.uint8)
            
            if len(fakes) < 4:
                reals.append(rearrange(x[:k].type(torch.uint8), "b t c h w -> b c t h w"))
                fakes.append(rearrange(fake, "b t h w c -> b c t h w"))
                reals_ldmk.append(rearrange(x_l[:k].type(torch.uint8), "b t c h w -> b c t h w"))
                masked_vid.append(rearrange(masked_x[:k].type(torch.uint8), "b t c h w -> b c t h w"))

    reals = torch.cat(reals)
    fakes = torch.cat(fakes)
    reals_ldmk = torch.cat(reals_ldmk)
    masked_vid = torch.cat(masked_vid)
    
    if rank == 0:
        real_vid = save_image_grid(
            reals.cpu().numpy(),
            os.path.join(logger.logdir, f"gt_{it}.gif"),
            drange=[0, 255],
            grid_size=(4, 4),
        )
        fake_vid = save_image_grid(
            fakes.cpu().numpy(),
            os.path.join(logger.logdir, f"generated_{it}.gif"),
            drange=[0, 255],
            grid_size=(4, 4),
        )
        ldmk_vid = save_image_grid(
            reals_ldmk.cpu().numpy(),
            os.path.join(logger.logdir, f"cond_motion_{it}.gif"),
            drange=[0, 255],
            grid_size=(4, 4),
        )
        masked_vid = save_image_grid(
            masked_vid.cpu().numpy(),
            os.path.join(logger.logdir, f"cond_pose_vid_{it}.gif"),
            drange=[0, 255],
            grid_size=(4, 4),
        )

        fake_vid = np.expand_dims(fake_vid, 0).transpose(0, 1, 4, 2, 3)
        logger.video_summary("unconditional", fake_vid, it)

    return 0