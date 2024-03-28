import os
import sys
from os import path
import math
import random
import pickle
import warnings
import glob
import face_alignment
import pdb
import imageio
import numpy as np
import cv2
import torch
import argparse
import torchvision.transforms as T
import torch.nn.functional as F
from tqdm import tqdm

sys.path.append(path.dirname(path.abspath(__file__)))
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from face3d_helper import Face3DHelper
from deep_3drecon.reconstructor import Reconstructor
from deep_3drecon.util.preprocess import align_img
from torchvision.utils import save_image
import einops
import PIL
from data_utils import *

# ========================================== #
HORIZON = 156
# ========================================== #
def load_idlist(path):
    with open(path, "r") as f:
        lines = f.readlines()
        id_list = [line.replace("\n", "").replace(".mp4", "").strip() for line in lines]
    return id_list


class Aligner_3dmm:
    def __init__(
        self,
        path,  # Path to directory or zip.
        ldmk_path,
        atom_identity,
        ps_drv_identity,
        resolution=256,
        nframes=16,  # number of frames for each video.
        **super_kwargs,  # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._zipfile = None
        self.apply_resize = True

        path_root = path

        self.fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, network_size=4, device="cuda", flip_input=False)
        self.face3d_helper = Face3DHelper("deep_3drecon/BFM")
        self.face_reconstructor = Reconstructor()
        land_vid = self._load_np_from_path(folder=ldmk_path, fname=f"{atom_identity}/atom_0.npy")
        if len(land_vid) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + ldmk_path + "\n" "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.land_vid = land_vid
        self.path = path
        self.atom_identity = atom_identity
        self.ps_drv_identity = ps_drv_identity

        self.nframes = nframes
        self.img_resolution = resolution
        self._path = path
        self._total_size = self.land_vid.shape[0]

        self._raw_shape = [self._total_size] + [3, resolution, resolution]
        self.xflip = False
        self.shuffle_indices = [i for i in range(self._total_size)]
        self.to_tensor = T.ToTensor()
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

    def rigid_transform(self, vs, rot, trans):
        vs_r = torch.matmul(vs, rot)
        vs_t = vs_r + trans.view(-1, 1, 3)
        return vs_t

    def compute_rotation_matrix(self, angles):
        n_b = angles.shape[0]
        sinx = torch.sin(angles[:, 0])
        siny = torch.sin(angles[:, 1])
        sinz = torch.sin(angles[:, 2])
        cosx = torch.cos(angles[:, 0])
        cosy = torch.cos(angles[:, 1])
        cosz = torch.cos(angles[:, 2])
        rotXYZ = torch.eye(3).view(1, 3, 3).repeat(n_b * 3, 1, 1).view(3, n_b, 3, 3).to(angles.device)
        rotXYZ[0, :, 1, 1] = cosx
        rotXYZ[0, :, 1, 2] = -sinx
        rotXYZ[0, :, 2, 1] = sinx
        rotXYZ[0, :, 2, 2] = cosx
        rotXYZ[1, :, 0, 0] = cosy
        rotXYZ[1, :, 0, 2] = siny
        rotXYZ[1, :, 2, 0] = -siny
        rotXYZ[1, :, 2, 2] = cosy
        rotXYZ[2, :, 0, 0] = cosz
        rotXYZ[2, :, 0, 1] = -sinz
        rotXYZ[2, :, 1, 0] = sinz
        rotXYZ[2, :, 1, 1] = cosz
        rotation = rotXYZ[2].bmm(rotXYZ[1]).bmm(rotXYZ[0])
        return rotation.permute(0, 2, 1)

    def _load_img_from_path(self, folder, fname):
        path = os.path.join(folder, fname)
        with self._open_file(path) as f:
            img = self.to_tensor(PIL.Image.open(f)).numpy() * 255  # c h w
        return img

    def lm68_2_lm5(self, in_lm):
        lm_idx = np.array([31, 37, 40, 43, 46, 49, 55]) - 1
        lm = np.stack(
            [
                in_lm[:, lm_idx[0], :],
                np.mean(in_lm[:, lm_idx[[1, 2]], :], 1),
                np.mean(in_lm[:, lm_idx[[3, 4]], :], 1),
                in_lm[:, lm_idx[5], :],
                in_lm[:, lm_idx[6], :],
            ],
            axis=1,
        )
        lm = lm[:, [1, 2, 0, 3, 4], :2]
        return lm

    def rgb_to_ldmk_16(self, pose_driving_rgb):
        lm68_lst = []
        input_lst = []

        for rgb in pose_driving_rgb:
            WH = rgb.shape[1]
            rgb_res = einops.rearrange(rgb, "c h w -> h w c")
            lm68 = self.fa.get_landmarks(rgb_res.to("cuda"))[0]  # [68,2]
            input_lst.append(rgb_res)
            lm68_lst.append(lm68)
        input = np.stack(input_lst)
        lm68 = np.stack(lm68_lst)
        lm5 = self.lm68_2_lm5(lm68)  # [5, 2]

        rgb_input = torch.from_numpy(input)
        lm68 = torch.from_numpy(lm68)
        lm5 = torch.from_numpy(lm5)
        return rgb_input, lm68, lm5

    def _load_np_from_path(self, folder, fname):
        path = os.path.join(folder, fname)
        np_l = np.load(path)
        return np_l

    def _change_np_img_size(self, lm3D, WH=256, flip=True):
        B_ = lm3D.shape[0]
        if lm3D.shape[-1] == 3:  # normed (0~1) ldmk
            lm3d = (lm3D * WH / 2 + WH / 2).astype(int)
            lm2d = lm3d[:B_, :, :2]
        else:  # img-sized ldmk
            lm2d = lm3D.astype(int)

        img = np.ones([B_, 256, 256, 3], dtype=np.uint8) * 0
       

        for b in range(len(lm2d)):
            for i in range(len(lm2d[b])):
                x, y = lm2d[b][i]
                _x, _y = int(x / WH * 256.0), int(y / WH * 256.0)
                img[b] = cv2.circle(img[b], center=(_x, _y), radius=3, color=(255, 255, 255), thickness=-1)

        if flip:
            img = np.stack([cv2.flip(im, 0) for im in img])

        return img

    def de_aligning_ldmk(self, img_size, lm68_224, t_, s_, temp_size=224.0):
        """
        Parameters:
            img_size: rgb frame size
            lm68_224: 68 landmarks adjusted to 3dmm template
            t: translation
            s: scaling

        Returns:
            lm68_po: 68 landmarks adjusted to rgb frame
        """
        B_ = lm68_224.shape[0]
        w0, h0 = img_size[:, 0], img_size[:, 1]
        w = (w0 * s_).astype(np.int32)
        h = (h0 * s_).astype(np.int32)

        s_ = s_[:, None, None]
        lm68_po = lm68_224 + np.reshape(np.array([(w / 2 - temp_size / 2), (h / 2 - temp_size / 2)]), [B_, 1, 2])
        lm68_po = (
            np.stack([lm68_po[:, :, 0] + t_[:, 0, :] - np.expand_dims(w0, 1) / 2, lm68_po[:, :, 1] + t_[:, 1, :] - np.expand_dims(h0, 1) / 2], axis=2)
            / s_
        )

        return lm68_po

    def align_3dmm_to_FA(self, index, NUM_OF_FRAME, args):
        srt_index = index * NUM_OF_FRAME
        end_index = srt_index + NUM_OF_FRAME

        """Assign Names"""
        print("assign names")
        rgb_root = f"{args.driv_video_path}/{self.atom_identity}"
        pose_drv_rgb_root = f"{args.driv_video_path}/{self.ps_drv_identity}"
        file_num_lst = [str(i).zfill(5) for i in range(srt_index, end_index)]

        """ Load vid & pos_drv_vid & lm68 of pose_driving vid """
        print("Load")
        vid = np.stack(
            [self._load_img_from_path(folder=rgb_root, fname=file_num + ".jpg") for file_num in file_num_lst],
            axis=0,
        )
        pos_drv_vid_np = np.stack(
            [self._load_img_from_path(folder=pose_drv_rgb_root, fname=file_num + ".jpg") for file_num in file_num_lst],
            axis=0,
        )
        print("lm 68 of pose driving vid")
        pos_drv_vid = torch.tensor(pos_drv_vid_np)
        _, lm68_fa, lm5_fa = self.rgb_to_ldmk_16(pos_drv_vid)

        """ GET the differences between lm68-ps_drv & 3dmm_mean """
        print("GET the differences between lm68-ps_drv & 3dmm_mean")
        lm68_atom = self.land_vid[srt_index:end_index, :, :]
        coeff, align_img, trans_params = self.face_reconstructor.recon_coeff(pos_drv_vid, lm5_fa, return_image=True)
        coeff = torch.from_numpy(coeff).float()
        angle = coeff[:, 224:227]
        translation = coeff[:, 254:257]  # [T_y, c=3]
        rotation = self.compute_rotation_matrix(angle)
        rotation = rotation.to("cuda")
        translation = translation.to("cuda")

        """ CHANGE the head rotation // scaling&translation not yet """
        print("CHANGE the head rotation // scaling&translation not yet")
        lm68_atom = torch.from_numpy(lm68_atom).float().to("cuda")
        lm68_atom_r = self.rigid_transform(lm68_atom, rotation, translation)

        """ CHANGE the scaling&translation """
        print("CHANGE the scaling&translation")
        img_size = pos_drv_vid.shape[2:]

        trans_params = self.face_reconstructor.pose_extract(img_size, lm5_fa, self.face_reconstructor.lm3d_std)
        trans_params_imgsize = np.array([trans_param.img_size for trans_param in trans_params])
        trans_params_t = np.array([trans_param.t for trans_param in trans_params])
        trans_params_s = np.array([trans_param.s for trans_param in trans_params])

        lm68_atom_r_224 = (lm68_atom_r.detach().cpu().numpy() * 224 / 2 + 224 / 2).astype(int)
        lm68_atom_r_s = self.de_aligning_ldmk(trans_params_imgsize, lm68_atom_r_224[:, :, :2], trans_params_t, trans_params_s)
        WH = trans_params_imgsize[0][0]
        """ FLIPPING & TRANSLATION """
        print(" FLIPPING & TRANSLATION ")
        lm68_fa_np = lm68_fa.detach().cpu().numpy()
        B_ = lm68_fa_np.shape[0]
        for b in range(B_):  # FLIPPING
            lm68_atom_r_s[b, :, 1] = trans_params_imgsize[b, 1] - 1 - lm68_atom_r_s[b, :, 1]
        """ Choose 5 points among 68 points """
        print(" Choose 5 points among 68 points ")
        lm5_fa = self.lm68_2_lm5(lm68_fa_np[:, :, :2])
        lm5_po_np = self.lm68_2_lm5(lm68_atom_r_s[:, :, :2])
        """ Fine mean point difference & Translation """
        mean_point_temp = np.mean(lm5_fa, axis=1)
        mean_point = np.mean(lm5_po_np, axis=1)
        lm68_atom_r_s_t = lm68_atom_r_s[:, :, :2] + np.expand_dims(mean_point_temp - mean_point, 1)
        """ Visualize """
        lm68_atom_r_s_t_img = self._change_np_img_size(lm68_atom_r_s_t, WH=int(trans_params_imgsize[0, 1]), flip=False)
        visu_ldmk = torch.tensor(lm68_atom_r_s_t_img[0, :, :, :]).permute(2, 0, 1)
        """ Resizing Frames """
        print(" Resizing Frames ")
        vid = resize_crop(torch.from_numpy(vid).float(), resolution=self.img_resolution)  # t c h w -> c t h w

        land_vid = torch.from_numpy(lm68_atom_r_s_t_img).permute(3, 0, 1, 2).float()
        
        if vid.size(1) == self.nframes // 2:
            vid = torch.cat([torch.zeros_like(vid).to(vid.device), vid], dim=1)
        if land_vid.size(1) == self.nframes // 2:
            land_vid = torch.cat([torch.zeros_like(land_vid).to(land_vid.device), land_vid], dim=1)

        return (
            rearrange(vid, "c t h w -> t c h w"),
            rearrange(land_vid, "c t h w -> t c h w"),
            lm68_atom_r_s_t[:, :, :2].astype(int),
            index,
        )


def main(args):
    NUM_OF_FRAME = 75  
    NP_SAVE = f"{args.save_dir}/self-recon/aligned_npy"
    DATA_LOCATION=args.driv_video_path
    OUTPUT_LOCATION = f"{args.save_dir}/self-recon/aligned_png"
    if args.id_list is None:
        id_list = os.listdir(args.driv_video_path)
    else:
        id_list = load_idlist(args.id_list)
        
    id_list = ['WRA_JoePitts_000'] # e.g
    for id_ in tqdm(id_list, leave=False):
        atom_id = id_
        ps_drv_id = id_

        face_aligner = Aligner_3dmm(
            DATA_LOCATION,
            args.ldmk_path,
            atom_id,
            ps_drv_id,
            nframes=NUM_OF_FRAME,  # you're gonna get 16 frames for each iteration below.
        )
        
        for index in range(HORIZON // NUM_OF_FRAME):
            str_index = index * NUM_OF_FRAME
            vid, land_vid, lm68_2d, index = face_aligner.align_3dmm_to_FA(index, NUM_OF_FRAME, args)
            os.makedirs(os.path.join(OUTPUT_LOCATION, ps_drv_id), exist_ok=True)
            os.makedirs(os.path.join(NP_SAVE, ps_drv_id), exist_ok=True)

            if atom_id not in os.listdir(OUTPUT_LOCATION):
                os.mkdir(os.path.join(OUTPUT_LOCATION, ps_drv_id))
            for land_fr in land_vid:
                save_image(land_fr / 255, os.path.join(OUTPUT_LOCATION, ps_drv_id, f"{str(str_index).zfill(5)}.png"))
                np.save(os.path.join(NP_SAVE, ps_drv_id, f"{str(str_index).zfill(5)}.npy"), lm68_2d[str_index - index * NUM_OF_FRAME])
                str_index += 1


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="")
    args.add_argument("--ldmk_path", type=str, default="../../AToM/results/frontalized1/frontalized_npy")
    # args.add_argument("--driv_video_path", type=str, default="data/train/HDTF/frames") # jpg frames root
    args.add_argument("--driv_video_path", type=str, default="/media/data/HDTF_preprocessed/30_frame_re/HDTF") # jpg frames root
    args.add_argument("--id_list", type=str, default=None)
    args.add_argument("--save_dir", type=str,
                      default="../../AToM/results/aligned1")
    args = args.parse_args()
    main(args)