import os
import sys
from os import path
import math
import random
import pickle
import warnings
import glob
import pdb
import imageio
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import torchvision.transforms as T
import torchvision.datasets as datasets
from torchvision.datasets.folder import make_dataset

sys.path.append(path.dirname(path.abspath(__file__)))
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from torchvision.io import read_video
import pdb
import cv2
import einops

# -------------------------------------------------------------------------- #

DATA_LOCATION = "path to your hdtf frames folder" # /data/HDTF/frames
landmark_dir = "../AToM/results/aligned1/self-recon/aligned_npy/"
HDTF_KPT_LOCATION = "path to your hdtf kpts folder" # /data/HDTF/keypoints/non-face-centric/posed

# -------------------------------------------------------------------------- #

# load text file and readline
def load_train_id_list(path):
    with open(path, "r") as f:
        lines = f.readlines()
        train_id_list = [line.replace("\n", "").replace(".mp4", "").strip() for line in lines]
    return train_id_list


train_id_txt = "text_folders/train_id.txt"
train_id_list = load_train_id_list(train_id_txt)
# -------------------------------------------------------------------------- #
from tools.data_utils import *

import av


class EvalDataset(Dataset):
    def __init__(
        self,
        path,  # Path to directory or zip.
        identity,
        total_frames=16,
        resolution=None,
        nframes=16,  # number of frames for each video.
        train=True,
        interpolate=False,
        loader=default_loader,  # loader for "sequence" of images
        return_vid=True,  # True for evaluating FVD
        cond=False,
        **super_kwargs,  # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self.apply_resize = True

        path_root = path
        identity_list = [identity]
        classes = sorted(identity_list)
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        imgs = make_imagefolder_hdtf_dataset(path, identity_list, nframes, class_to_idx, False)

        """ Set the length of the dataset for just one identity loading """
        print("identity_list", identity_list)
        print("path", path)
        # pdb.set_trace()
        path = imgs[0]
        imgs = natsorted(os.listdir(path[0]))

        if len(imgs) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + path + "\n" "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.imgs = imgs[:total_frames]
        self.ref = imgs[0]
        self.path = path
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.nframes = nframes
        self.loader = loader
        self.img_resolution = resolution
        self._path = path
        self._total_size = len(self.imgs)
        self._raw_shape = [self._total_size] + [3, resolution, resolution]
        self.xflip = False
        self.return_vid = return_vid
        self.shuffle_indices = [i for i in range(self._total_size)]
        self.to_tensor = transforms.ToTensor()
        random.shuffle(self.shuffle_indices)
        self._type = "dir"

    def _file_ext(self, fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == "zip"
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname):
        if self._type == "dir":
            return open(os.path.join(fname), "rb")
        if self._type == "zip":
            return self._get_zipfile().open(fname, "r")
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def _crop_lower_half(self, img, landmarks):
        h, w = img.shape[-2:]
        scaler = h / self.img_resolution
        
        mask = np.ones((h, w))
        mask[(landmarks[33][1]).astype(int) :, :] = 0.0
        mask = mask[None, ...]
        image_mask = (img * mask).astype(np.uint8)
        
        return image_mask

    def _load_img_from_path(self, folder, fname):
        path = os.path.join(folder, fname)
        with self._open_file(path) as f:
            img = self.to_tensor(PIL.Image.open(f)).numpy() * 255  # c h w
        return img

    def _load_np_from_path(self, folder, fname):
        path = os.path.join(folder, fname)
        # with self._open_file(path) as f:
        np_l = np.load(path)
        return np_l

    def _change_np_img_size(self, lm3D, WH=256, flip=True):
        B_ = lm3D.shape[0]
        if lm3D.shape[-1] == 3:
            # normed (0~1) ldmk
            lm3d = (lm3D * WH / 2 + WH / 2).astype(int)
            lm2d = lm3d[:B_, :, :2]
        else:
            # img-sized (0~1) ldmk
            lm2d = lm3D.astype(int)
        # img = np.ones([B_, WH, WH, 3], dtype=np.uint8) * 0
        img = np.ones([B_, 256, 256, 3], dtype=np.uint8) * 0
        for b in range(len(lm2d)):
            for i in range(len(lm2d[b])):
                x, y = lm2d[b][i]
                # img[b] = cv2.circle(img[b], center=(x, y), radius=6, color=(255, 255, 255), thickness=-1)
                _x, _y = int(x / WH * 256.0), int(y / WH * 256.0)
                img[b] = cv2.circle(img[b], center=(_x, _y), radius=3, color=(255, 255, 255), thickness=-1)

        if flip:
            img_ls = [cv2.flip(img[b], 0) for b in range(len(img))]
            img_concat = np.stack(img_ls, axis=0)

            img = img_concat
        else:
            pass

        return img

    def __getitem__(self, index):
        """Assign Periodic Frame Index"""

        srt_index = index * 16
        end_index = srt_index + 16
        clip = self.imgs[srt_index:end_index]

        print(srt_index, end_index)

        """ Load Images & Frames and stacking """
        ref = self.imgs[0]
        ref = np.stack([self._load_img_from_path(folder=self.path[0], fname=ref) for i in range(len(clip))], axis=0)
        vid = np.stack(
            [self._load_img_from_path(folder=self.path[0], fname=clip[i]) for i in range(len(clip))],
            axis=0,
        )  # shape: (16, 3, 634, 634)

        ldmk_pth = self.path[0].replace(DATA_LOCATION, landmark_dir)
        land_vid = np.stack(
            [
                self._load_np_from_path(
                    folder=ldmk_pth,
                    fname=clip[i].replace("jpg", "npy"),
                )
                for i in range(len(clip))
            ],
            axis=0,
        )

        masked_ref_vid_ls = []
        for i in range(len(clip)):
            masked_ref_vid_ls.append(
                self._crop_lower_half(
                    self._load_img_from_path(folder=self.path[0], fname=clip[i]),
                    self._load_np_from_path(
                        folder=self.path[0].replace(DATA_LOCATION, HDTF_KPT_LOCATION),
                        fname=clip[i].replace("jpg", "npy"),
                    ),
                )
            )
        masked_ref_vid = np.stack(masked_ref_vid_ls, axis=0)
 

        vid_size = vid.shape[-1]
        img_sized_ldmk = self._change_np_img_size(land_vid, WH=vid_size, flip=False)  # (16, 726, 726, 3)
        img_sized_ldmk = einops.rearrange(img_sized_ldmk, "t h w c -> t c h w")  # (16, 3, 726, 726)
 
        """ Resizing Frames """
        ref = resize_crop(torch.from_numpy(ref).float(), resolution=self.img_resolution)  # c t h w
        vid = resize_crop(torch.from_numpy(vid).float(), resolution=self.img_resolution)  # c t h w
        land_vid = torch.from_numpy(img_sized_ldmk).float()
        masked_ref_vid = resize_crop(torch.from_numpy(masked_ref_vid).float(), resolution=self.img_resolution)  # c t h w

        if vid.size(1) == self.nframes // 2:
            vid = torch.cat([torch.zeros_like(vid).to(vid.device), vid], dim=1)
        if land_vid.size(1) == self.nframes // 2:
            land_vid = torch.cat([torch.zeros_like(land_vid).to(land_vid.device), land_vid], dim=1)
        if masked_ref_vid.size(1) == self.nframes // 2:
            masked_ref_vid = torch.cat([torch.zeros_like(masked_ref_vid).to(masked_ref_vid.device), masked_ref_vid], dim=1)

        return (
            rearrange(ref, "c t h w -> t c h w"),
            rearrange(vid, "c t h w -> t c h w"),
            land_vid,
            rearrange(masked_ref_vid, "c t h w -> t c h w"),
            index,
        )

    def __len__(self):
        return self._total_size // 16


def get_loaders(
    rank,
    imgstr,
    num_frames,
    resolution,
    timesteps,
    skip,
    identity,
    batch_size=1,
    n_gpus=1,
    seed=42,
    cond=False,
    use_train_set=False,
    shuffle=True,
):
    """
    Load dataloaders for an image dataset, center-cropped to a resolution.
    """

    if imgstr == "HDTF":
        test_dir = DATA_LOCATION
        # if cond:
        #     print("here")
        #     timesteps *= 2  # for long generation
        # pdb.set_trace()
        testset = EvalDataset(
            test_dir,
            identity,
            total_frames=num_frames,
            train=False,
            resolution=resolution,
            nframes=timesteps,
            cond=cond,
        )
        print("testset length: ", len(testset))

    kwargs = {"pin_memory": True, "num_workers": 3}
    # len(testset)
    # testset_sampler = InfiniteSampler(testset, num_replicas=n_gpus, rank=rank, shuffle=shuffle, seed=seed)
    testloader = DataLoader(
        testset,
        batch_size=batch_size // n_gpus,
        pin_memory=False,
        num_workers=4,
        prefetch_factor=2,
    )

    return testset, testloader

