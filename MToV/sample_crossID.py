import sys

sys.path.extend(["."])
import os
import argparse
import torch
from omegaconf import OmegaConf
from exps.diffusion import diffusion
from exps.first_stage import first_stage
from utils import set_random_seed
import os
import json
from tools.trainer import latentDDPM
from tools.dataloader_sample_crossID import get_loaders
from tools.scheduler import LambdaLinearScheduler
from tools.data_utils import *
from models.autoencoder.autoencoder_vit import ViTAutoencoder
from models.ddpm.unet import UNetModel, DiffusionWrapper
from losses.ddpm import DDPM

import cv2
import natsort
import copy
from utils import file_name, Logger, download

from models.ema import LitEma

from einops import rearrange, repeat
import subprocess
import imageio

import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image

Img2Tensor = transforms.ToTensor()

import torchvision
import PIL

from evals.fvd.fvd import get_fvd_logits, frechet_distance
from evals.fvd.download import load_i3d_pretrained
import time
import random
import numpy as np
import torch

import pdb

from torchvision.utils import save_image

# SAVE MODULES


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
        # pdb.set_trace()
        # torchvision.io.write_video(f'{fname[:-3]}mp4', torch.from_numpy(img), fps=16)
        imgs = [PIL.Image.fromarray(img[i], "RGB") for i in range(len(img))]
        fname = fname.replace("generated", f"generated_gif")
        imgs[0].save(fname, quality=95, save_all=True, append_images=imgs[1:], duration=100, loop=0)


def save_image_at_folder(start_iter, img, fname, drange, grid_size, normalize=True):
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
        # torchvision.io.write_video(f'{fname[:-3]}mp4', torch.from_numpy(img), fps=16)
        imgs = [PIL.Image.fromarray(img[i], "RGB") for i in range(len(img))]
        cnt = 0
        for im in range(len(imgs)):
            save_name = os.path.join(fname, f"{start_iter+cnt}".zfill(4) + ".png")
            # if os.path.exists(save_name):
            #     prev_img = PIL.Image.open(save_name)
            #     avg = Image.blend(prev_img, imgs[im], 1.0 / float(2))
            #     avg.save(save_name)
            # else:
            imgs[im].save(save_name)
            cnt += 1

    # return img


def make_video(result_frames, audio_path, save_path_pasted_audio_pred, fps):
    save_path_pasted_pred = save_path_pasted_audio_pred.replace(".mp4", "no_audio.mp4")
    imageio.mimwrite(save_path_pasted_pred, result_frames, fps=fps, output_params=["-vf", f"fps={fps}"])
    subprocess.call(
        f"ffmpeg -y -i {save_path_pasted_pred} -i {audio_path} -map 0:v -map 1:a -c:v copy -shortest {save_path_pasted_audio_pred}",
        shell=True,
    )
    os.remove(save_path_pasted_pred)
    print(f"//////////////////////\n{save_path_pasted_audio_pred}\n//////////////////////")


def _file_ext(fname):
    return os.path.splitext(fname)[1].lower()


def _load_img_from_path(folder, fname):
    path = os.path.join(folder, fname)
    img = Img2Tensor(PIL.Image.open(path)).numpy() * 255

    return img  # c h w


# def get_ldmk_sequence(ldmk_id, num_frames):
#     ldmk_path = f"/media/data/HDTF_preprocessed/30_frame_re/HDTF_dlib_landmark/{ldmk_id}"
#     ldmk_list = os.listdir(ldmk_path)
#     ldmk_list.sort()

#     ldmk_list = [ldmk for ldmk in ldmk_list if ldmk.endswith(".png") and not "normed" in ldmk]

#     ldmk_vid = np.stack(
#         [
#             _load_img_from_path(
#                 folder=ldmk_path,
#                 fname=ldmk_list[i],
#             )
#             for i in range(num_frames)
#         ],
#         axis=0,
#     )

#     ldmk_vid = resize_crop(torch.from_numpy(ldmk_vid).float(), resolution=256)  # c t h w

#     if ldmk_vid.size(1) == num_frames // 2:
#         ldmk_vid = torch.cat([torch.zeros_like(ldmk_vid).to(ldmk_vid.device), ldmk_vid], dim=1)

#     ldmk = rearrange(ldmk_vid, "c t h w -> t c h w")

#     return ldmk


start = time.time()

seed = 2021
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42, help="random seed")
parser.add_argument("--id", type=str, default="main", help="experiment identifier")

""" Args about Data """
parser.add_argument("--data", type=str, default="SKY")
parser.add_argument("--ref_img", type=str, default=None)  # OOD test
parser.add_argument("--crossID", type=str, default=None)  # OOD test
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--ds", type=int, default=4)
parser.add_argument("--mouth_only", action="store_true")
parser.add_argument("--overlap", action="store_true")
parser.add_argument("--x_noisy_start", action="store_true")
parser.add_argument("--refvid_noisy_start", action="store_true")
parser.add_argument("--ratio_", type=float, default=0.75)
parser.add_argument("--fps", type=float, default=25)
parser.add_argument("--seconds", type=float, default=2)
parser.add_argument("--fix_noise", action="store_true")

""" Args about whose landmark to use """
parser.add_argument("--ldmk_owner_list", nargs="*")
parser.add_argument("--num_frames", type=int, default=304)
parser.add_argument("--eval_folder", type=str, default="eval_results")
parser.add_argument("--including_ldmk_video", action="store_true")
parser.add_argument("--use_last_as_reference", action="store_true")

""" Args about Model """
parser.add_argument("--pretrain_config", type=str, default="configs/autoencoder/base.yaml")
parser.add_argument("--diffusion_config", type=str, default="configs/latent-diffusion/base.yaml")

# for diffusion model path specification
parser.add_argument("--first_model", type=str, default="", help="the path of pretrained model")
parser.add_argument("--first_model_ldmk", type=str, default="", help="the path of pretrained model")
parser.add_argument(
    "--second_model",
    type=str,
    default="",
    help="the path of pretrained model",
)

args = parser.parse_args()
""" FIX THE RANDOMNESS """
set_random_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

args.n_gpus = torch.cuda.device_count()

device = torch.device("cuda")
# init and save configs
""" RUN THE EXP """
config = OmegaConf.load(args.diffusion_config)
first_stage_config = OmegaConf.load(args.pretrain_config)
args.unetconfig = config.model.params.unet_config
args.lr = config.model.base_learning_rate
args.scheduler = config.model.params.scheduler_config
args.res = first_stage_config.model.params.ddconfig.resolution
args.timesteps = first_stage_config.model.params.ddconfig.timesteps
args.skip = first_stage_config.model.params.ddconfig.skip
args.ddconfig = first_stage_config.model.params.ddconfig
args.embed_dim = first_stage_config.model.params.embed_dim
args.ddpmconfig = config.model.params
args.cond_model = config.model.cond_model

rank = 0

first_stage_model = ViTAutoencoder(args.embed_dim, args.ddconfig).to(device)
first_stage_model_ckpt = torch.load(args.first_model)
first_stage_model.load_state_dict(first_stage_model_ckpt)
# ----------------------------------------------------------------------------------------#
first_stage_model = ViTAutoencoder(args.embed_dim, args.ddconfig).to(device)
if rank == 0:
    first_stage_model_ckpt = torch.load(args.first_model)
    first_stage_model.load_state_dict(first_stage_model_ckpt)

first_stage_model_ldmk = ViTAutoencoder(args.embed_dim, args.ddconfig).to(device)
if rank == 0:
    first_stage_model_ldmk_ckpt = torch.load(args.first_model_ldmk)
    first_stage_model_ldmk.load_state_dict(first_stage_model_ldmk_ckpt)
# ----------------------------------------------------------------------------------------#

unet = UNetModel(**args.unetconfig)
model = DiffusionWrapper(unet).to(device)
# model = torch.nn.DataParallel(model, device_ids=[0])


# DDPM pre-trained 로드 어케함? --------------------------------------------------------------------------- #
ema_model = copy.deepcopy(model)
dir = args.second_model
ema_model_ckpt = torch.load(dir)
ema_model.load_state_dict(ema_model_ckpt)
ema_model.eval()
# ------------------------------------------------------------------------------------------------------- #

first_stage_model.eval()
first_stage_model_ldmk.eval()
model.eval()

cond_model = ema_model.diffusion_model.cond_model  # True
diffusion_model = DDPM(
    ema_model,
    channels=ema_model.diffusion_model.in_channels,
    image_size=ema_model.diffusion_model.image_size,
    sampling_timesteps=100,
    w=0.0,
).to(device)

if args.ldmk_owner_list:
    ldmk_owner_list = args.ldmk_owner_list
else:
    def load_list(path):
        with open(path, "r") as f:
            lines = f.readlines()
            eval_id_list = [line.replace("\n", "").replace(".mp4", "").strip() for line in lines]
        return eval_id_list

    audio_pth = "./text_folders/sample_cross_audio_hdtf.txt"
    audio_list = load_list(audio_pth)
    ref_pth = "./text_folders/sample_cross_id_hdtf.txt"
    ref_list = load_list(ref_pth)


for audio_id in audio_list:
    for ref_id in ref_list:
        test_dataset, total_vid = get_loaders(
            audio_id, 
            ref_id,
            rank,
            "HDTF",
            args.num_frames,
            args.res,
            args.timesteps,
            args.skip,
            args.batch_size,
            args.n_gpus,
            args.seed,
            args.cond_model,
            shuffle=False,
        )

        import os
        from torchvision.utils import save_image

        # Make Dirs
        epoch = dir.split("_")[-1].split(".")[0]
        os.makedirs(f"{args.eval_folder}_{args.second_model.split('/')[-1]}/{audio_id}", exist_ok=True)
        os.makedirs(f"{args.eval_folder}_{args.second_model.split('/')[-1]}/{audio_id}/{ref_id}", exist_ok=True)
        
        os.makedirs(f"{args.eval_folder}_{args.second_model.split('/')[-1]}/{audio_id}/{ref_id}/references", exist_ok=True)
        os.makedirs(f"{args.eval_folder}_{args.second_model.split('/')[-1]}/{audio_id}/{ref_id}/gif", exist_ok=True)
        os.makedirs(f"{args.eval_folder}_{args.second_model.split('/')[-1]}/{audio_id}/{ref_id}/frames", exist_ok=True)
        os.makedirs(f"{args.eval_folder}_{args.second_model.split('/')[-1]}/{audio_id}/{ref_id}/GT", exist_ok=True)

        from PIL import Image

        if args.ref_img:
            img = Image.open(args.ref_img)
            img = Img2Tensor(img).unsqueeze(0)
            img = img[:, :3, :, :]
            img = img * 2.0 - 1.0
            img = img.to(device=device)
            img = img.unsqueeze(2)
            image = [img[:, :, :, :, :] for _ in range(16)]

        for it, (x_ref, x, x_l, masked_x, _) in enumerate(total_vid):
            k = args.batch_size  # choose the number of result indetities
            r = 0  # ref_frame num

            # naming just for saving image
            if args.overlap:
                ldmk_srt = it * 8
                ldmk_end = ldmk_srt + 16
            else:
                ldmk_srt = it * 16
                ldmk_end = ldmk_srt + 16

            print("total num_frames: ", args.num_frames)
            print("generating ldmk_srt: ", ldmk_srt)
            if args.num_frames < ldmk_srt:
                break

            with torch.no_grad():
                # if args.ref_img is None:
                # pers_frames = x[:k, :, :, :, :]
                # save ref with it in its name
                # ref = rearrange(ref / 127.5 - 1, "b t c h w -> b c t h w").to(device).detach()
                # image = [ref] * 16

                # torch.Size([4, 3, 16, 256, 256])
                x_ref = x_ref.to(device)
                x = x.to(device)
                x_l = x_l.to(device)
                masked_x = masked_x.to(device)

                x_ref = rearrange(x_ref / 127.5 - 1, "b t c h w -> b c t h w")  # videos
                x = rearrange(x / 127.5 - 1, "b t c h w -> b c t h w")  # videos
                x_l = rearrange(x_l / 127.5 - 1, "b t c h w -> b c t h w")  # videos
                masked_x = rearrange(masked_x / 127.5 - 1, "b t c h w -> b c t h w")  # videos

                z_ = first_stage_model.extract(x).detach()
                image_cond = first_stage_model.extract(x_ref).detach()
                z_l = first_stage_model_ldmk.extract(x_l).detach()
                masked_z = first_stage_model.extract(masked_x).detach()
                
                

                save_image(
                    rearrange(x_l[0, :, :8, :, :], "c t h w -> t c h w"),
                    f"{args.eval_folder}_{args.second_model.split('/')[-1]}/{audio_id}/{ref_id}/references/kpts_{it}.png",
                    normalize=True,
                )

                ref_dir = f"{args.eval_folder}_{args.second_model.split('/')[-1]}/{audio_id}/{ref_id}/references/"

                """ Set up last frames as reference for generating """
                print("use_last_as_reference: ", args.use_last_as_reference)
                if args.use_last_as_reference:
                    last_frame_dir = os.path.join(ref_dir, str(ldmk_srt))
                    print("last_frame_dir name: ", last_frame_dir)
                    if os.path.exists(last_frame_dir):
                        frames = os.listdir(last_frame_dir)
                        frames.sort()
                        frames_list = []
                        for frame in frames:
                            img = Img2Tensor(Image.open(os.path.join(last_frame_dir, frame)))
                            img = img * 2.0 - 1.0
                            img = img.unsqueeze(0)
                            img = img.to(device=device)
                            image = [img for _ in range(16)]
                            image = torch.cat(image, dim=0)
                            frames_list.append(image)
                        frames_tensor = torch.stack(frames_list, dim=0)
                        frames_tensor = rearrange(frames_tensor, "b t c h w -> b c t h w")
                        image_cond = first_stage_model.extract(frames_tensor.to(device).detach())
                        image_cond = image_cond[:, :, 0 : 32 * 32]
                else:
                    pass

                """ 
                Landmark condition rearranging 
                """
                c = torch.cat([z_l, masked_z], dim=1)
            
                noised_start=None
                if args.x_noisy_start:
                    noised_start = image_cond.float()
                elif args.refvid_noisy_start:
                    noised_start = z_.float()
                    
                z = diffusion_model.sample(
                    batch_size=k,
                    cond=c.float(),
                    image_cond=image_cond.float(),
                    noised_start=noised_start,
                    ratio_=args.ratio_,
                    fix_noise=args.fix_noise,
                )
                fake = first_stage_model.decode_from_sample(z).clamp(-1, 1).cpu()
                fake = (1 + rearrange(fake, "(b t) c h w -> b t h w c", b=k)) * 127.5

                """ Save last frame for generating next 16 frames """
                reference_saved_folder = os.path.join(ref_dir, str(ldmk_end))
                os.makedirs(reference_saved_folder, exist_ok=True)
                last_frame = fake[:, -1, :, :, :]
                last_frames = [last_frame[i, :, :, :] for i in range(k)]
                for idx in range(len(last_frames)):
                    fname = os.path.join(reference_saved_folder, f"{idx}.png")
                    img = np.asarray(last_frames[idx], dtype=np.float32)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = np.rint(img).clip(0, 255).astype(np.uint8)
                    cv2.imwrite(fname, img)

                """ Visualization """
                fakes = []
                fake = fake.type(torch.uint8)
                fakes.append(rearrange(fake[:k, :, :, :, :], "b t h w c -> b c t h w"))
                fakes = torch.cat(fakes)
                if args.including_ldmk_video:
                    # gt_ldmk = rearrange(x_l[:k, :, :, :, :], "b t c h w -> b c t h w")
                    gt_ldmk = x_l[:k, :, :, :, :]
                    gt_ldmk = (gt_ldmk * 255 + 255) / 2
                    # check the gt_ldmk range!!!!!!!!!!!
                    fakes = torch.cat([gt_ldmk.to("cpu"), fakes])
                    grid_size = k + 1  # for expanding the columns number
                else:
                    grid_size = k

                save_image_grid(
                    img=fakes.cpu().numpy(),
                    fname=f"{args.eval_folder}_{args.second_model.split('/')[-1]}/{audio_id}/{ref_id}/gif/generated_{it}.gif",
                    drange=[0, 255],
                    grid_size=(grid_size, 1),
                )

                save_image_at_folder(
                    start_iter=ldmk_srt,
                    img=fakes.cpu().numpy(),
                    fname=f"{args.eval_folder}_{args.second_model.split('/')[-1]}/{audio_id}/{ref_id}/frames",
                    drange=[0, 255],
                    grid_size=(grid_size, 1),
                )

                if args.num_frames <= ldmk_end:
                    break

        print("🦖🦖🦖 LET'S MAKE VIDEO 🦖🦖🦖")
        visualization_root = f"{args.eval_folder}_{args.second_model.split('/')[-1]}/{audio_id}/{ref_id}/frames/*.png"
        result_frames = glob.glob(visualization_root)
        result_frames = natsort.natsorted(result_frames)
        frames = [Image.open(img) for img in result_frames]

        # Generate Video with Audio
        if args.crossID:
            audio_path = f"/media/data/HDTF/HDTF_30/{args.crossID}.mp4"
        else:
            audio_path = f"/media/data/HDTF/HDTF_30/{audio_id}.mp4"

        os.makedirs(f"{args.eval_folder}_{args.second_model.split('/')[-1]}/0_total-videos", exist_ok=True)
        save_path_pasted_audio_pred = f"{args.eval_folder}_{args.second_model.split('/')[-1]}/0_total-videos/audio_{audio_id}_id_{ref_id}.mp4"
        make_video(frames, audio_path, save_path_pasted_audio_pred, fps=args.fps)

        print("\n\n\nGenerating video is DonE!\n\n\n")

        print("time :", time.time() - start)
