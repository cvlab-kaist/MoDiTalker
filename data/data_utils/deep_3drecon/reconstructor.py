"""This script is the test script for Deep3DFaceRecon_pytorch
Pytorch Deep3D_Recon is 8x faster than TF-based, 16s/iter ==> 2s/iter
"""

import os
import pdb

# os.environ['PYTHONPATH'] = os.environ['PYTHONPATH'] + ":" + os.path.abspath("deep_3drecon")
import torch
import torch.nn as nn
from .deep_3drecon_models.facerecon_model import FaceReconModel
from .util.preprocess import align_img, trans_params_extract
from PIL import Image
import numpy as np
from .util.load_mats import load_lm3d
import torch
import pickle as pkl
from PIL import Image

from commons.tensor_utils import convert_to_tensor, convert_to_np
from torchvision.utils import save_image

# with open("aux_models/deep_3drecon/reconstructor_opt.pkl", "rb") as f:
with open("deep_3drecon/reconstructor_opt.pkl", "rb") as f:
    opt = pkl.load(f)


class Reconstructor(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = FaceReconModel(opt)
        self.model.setup(opt)
        
        self.model.device = "cuda"
        self.model.parallelize()
        self.model.eval()
        self.lm3d_std = load_lm3d(opt.bfm_folder)

    def preprocess_data(self, im, lm, lm3d_std):
        # to RGB
        _, W, H = im.shape
        lm = lm.reshape([-1, 2])
        lm[:, -1] = H - 1 - lm[:, -1]

        if im.shape[0] == 3:
            im = im.permute(1, 2, 0)
        trans_params, im, lm, _ = align_img(Image.fromarray(convert_to_np(im).astype(np.uint8)), convert_to_np(lm), convert_to_np(lm3d_std))
        im = torch.tensor(np.array(im) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
        return im, lm, trans_params

    def preprocess_data_reverse(self, im, lm68, lm5, lm3d_std):
        # to RGB
        H, W, _ = im.shape
        lm68 = lm68.reshape([-1, 3])
        lm5 = lm5.reshape([-1, 3])

        # (Pdb) H_ - 1 - lm[:, -2]
        # tensor([224.0000, 222.5000, 186.0000, 146.0000, 142.0000])
        # (Pdb) lm[:, -2]
        # tensor([165.0000, 166.5000, 203.0000, 243.0000, 247.0000])

        lm68[:, -2] = H - 1 - lm68[:, -2]
        lm5[:, -2] = H - 1 - lm5[:, -2]

        trans_params, im, lm, _ = align_img(Image.fromarray(convert_to_np(im)), convert_to_np(lm68), convert_to_np(lm5), convert_to_np(lm3d_std))
        im = torch.tensor(np.array(im) / 255.0, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
        lm = torch.tensor(lm).unsqueeze(0)
        return im, lm, trans_params

    def pose_extract(self, img_size_h_w, lm5, lm3d_std):
        # to RGB
        H = img_size_h_w[0]
        W = img_size_h_w[1]

        if lm5.shape[-1] == 3:
            lm5 = lm5.reshape([-1, 5, 3])
            lm5[:, :, -2] = H - 1 - lm5[:, :, -2]
        if lm5.shape[-1] == 2:
            lm5 = lm5.reshape([-1, 5, 2])
            lm5[:, :, -1] = H - 1 - lm5[:, :, -1]

        B_ = lm5.shape[0]  # 16
        trans_params_lst = []
        for i in range(B_):
            # pdb.set_trace()
            lm5_i = lm5[i]
            lm5_np = lm5_i.detach().cpu().numpy()
            trans_params = trans_params_extract((W, H), lm5_np, convert_to_np(lm3d_std))
            trans_params_lst.append(trans_params)
        # trans_params = trans_params_extract((W, H), convert_to_np(lm5_i), convert_to_np(lm3d_std))

        return trans_params_lst

    @torch.no_grad()
    def recon_coeff(self, batched_images, batched_lm5, return_image=True, batch_mode=True):
        bs = batched_images.shape[0]
        data_lst = []
        trans_param_lst = []
        for i in range(bs):
            img = batched_images[i]
            lm5 = batched_lm5[i]

            # image is aligned here -> align image to 3dmm size
            align_im, lm, trans_params = self.preprocess_data(img, lm5, self.lm3d_std)
            data = {"imgs": align_im, "lms": lm}
            data_lst.append(data)
            trans_param_lst.append(trans_params)

        if not batch_mode:
            coeff_lst = []
            align_lst = []
            for i in range(bs):
                data = data_lst

                self.model.set_input(data)  # unpack data from data loader
                self.model.forward()
                pred_coeff = self.model.output_coeff.cpu().numpy()
                align_im = (align_im.squeeze().permute(1, 2, 0) * 255).int().numpy().astype(np.uint8)

                coeff_lst.append(pred_coeff)
                align_lst.append(align_im)
            batch_coeff = np.concatenate(coeff_lst)
            batch_align_img = np.stack(align_lst)  # [B, 257]
        else:
            imgs = torch.cat([d["imgs"] for d in data_lst])
            # pdb.set_trace()  #
            lms = torch.cat([d["lms"] for d in data_lst])
            data = {"imgs": imgs, "lms": lms}
            self.model.set_input(data)  # unpack data from data loader
            self.model.forward()
            batch_coeff = self.model.output_coeff.cpu().numpy()
            batch_align_img = (imgs.permute(0, 2, 3, 1) * 255).int().numpy().astype(np.uint8)
            # pdb.set_trace()  #

        # pdb.set_trace()

        # trans_params
        # batch_coeff
        # self.model.pred_coeffs_dict
        # angle = batch_coeff[:, 224:227]
        # translation = batch_coeff[:, 254:257]

        return batch_coeff, batch_align_img, trans_param_lst

    # todo: batch-wise recon!

    def forward(self, batched_images, batched_lm5, return_image=True):
        return self.recon_coeff(batched_images, batched_lm5, return_image)
