import os, sys
import cv2
import numpy as np
import torch
import face_alignment
from os import path
from tqdm import tqdm
from PIL import Image
import pickle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
print(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))
sys.path.append(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))))

import deep_3drecon
from deep_3drecon.util.preprocess import align_img
from deep_3drecon.reconstructor import Reconstructor
from face3d_helper import Face3DHelper

from torchvision.utils import save_image

face3d_helper = Face3DHelper("deep_3drecon/BFM")

def chunk(iterable, chunk_size):
    final_ret = []
    cnt = 0
    ret = []
    for record in iterable:
        if cnt == 0:
            ret = []
        ret.append(record)
        cnt += 1
        if len(ret) == chunk_size:
            final_ret.append(ret)
            ret = []
    if len(final_ret[-1]) != chunk_size:
        final_ret.append(ret)
    return final_ret


def lm68_2_lm5(in_lm):
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


def compute_rotation_matrix(angles):
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


def save_lm_img(lm3D, out_path, WH=256, flip=True):
    if lm3D.shape[-1] == 3:
        lm3d = (lm3D * WH / 2 + WH / 2).astype(int)
        lm2d = lm3d[:, :2]
    else:
        lm2d = lm3D.astype(int)
    img = np.ones([WH, WH, 3], dtype=np.uint8) * 255
    for i in range(len(lm2d)):
        x, y = lm2d[i]
        img = cv2.circle(img, center=(x, y), radius=3, color=(0, 0, 0), thickness=-1)

    if flip:
        img = cv2.flip(img, 0)
    else:
        pass
    cv2.imwrite(out_path, img)


def de_aligning_ldmk(img_size, lm68_224, t_, s_, temp_size=224.0):
    """
    Parameters:
        img_size: rgb frame size
        lm68_224: 68 landmarks adjusted to 3dmm template
        t: translation
        s: scaling
        temp_size: 3dmm template size: 224

    Returns:
        lm68_po: 68 landmarks adjusted to rgb frame
    """
    B_ = lm68_224.shape[0]
    w0, h0 = img_size[:, 0], img_size[:, 1]
    w = (w0 * s_).astype(np.int32)
    h = (h0 * s_).astype(np.int32)

    s_ = s_[:, None, None]

    # frontalized lm68_224를 pose가 있도록 rotation.
    lm68_po = lm68_224 + np.reshape(np.array([(w / 2 - temp_size / 2), (h / 2 - temp_size / 2)]), [B_, 1, 2])
    lm68_po = (
        np.stack([lm68_po[:, :, 0] + t_[:, 0, :] - np.expand_dims(w0, 1) / 2, lm68_po[:, :, 1] + t_[:, 1, :] - np.expand_dims(h0, 1) / 2], axis=2)
        / s_
    )

    return lm68_po


def rigid_transform(vs, rot, trans):
    vs_r = torch.matmul(vs, rot)
    vs_t = vs_r + trans.view(-1, 1, 3)
    return vs_t


def get_lms(vs):
    kp_inds = face3d_helper.kp_inds
    lms = vs[:, kp_inds, :]
    return lms


def get_vs(id_coeff, exp_coeff):
    n_b = id_coeff.size(0)
    idBase = face3d_helper.id_base
    expBase = face3d_helper.exp_base
    meanshape = face3d_helper.meanshape
    face_shape = torch.einsum("ij,aj->ai", idBase, id_coeff) + torch.einsum("ij,aj->ai", expBase, exp_coeff) + meanshape
    face_shape = face_shape.view(n_b, -1, 3)
    face_shape = face_shape - meanshape.view(1, -1, 3).mean(dim=1, keepdim=True)
    return face_shape


def processing(frames_names_lst, input_stacked, lm68_stacked, face3d_helper, hdtf_frames_path, saving_pth):
    for frame_name in frames_names_lst:
        print("/".join(frame_name.split("/")[-2:]))

    """recon_coeff = lm5와 rgb image를 입력으로 받아서, 3DMM "크기"와 align되도록 crop 해주고, 3DMM coeff를 반환해주는 함수"""
    coeff, align_img, trans_params = face_reconstructor.recon_coeff(input_stacked, lm68_stacked, return_image=True)  # B, 257

    coeff = torch.from_numpy(coeff).float()  # tex = coeff[:, 144:224] # gamma = coeff[:, 227:254]
    identity = coeff[:, 0:80]
    exp = coeff[:, 80:144]
    angle = coeff[:, 224:227]
    translation = coeff[:, 254:257]  # [T_y, c=3]

    """ reconstruct_idexp_lm3d = id, exp coeff를 받아서, 그거를 base template에 따라 pose가 없도록 만들어 주는 함수 """
    idexp_lm3d = face3d_helper.reconstruct_idexp_lm3d(identity, exp).cpu().numpy()  # [B, 68, 3]
    B_ = idexp_lm3d.shape[0]

    """ visualize """
    idexp_lm3d = idexp_lm3d.reshape([B_, 1, -1])
    lm3d = idexp_lm3d / 10 + face3d_helper.key_mean_shape.unsqueeze(0).reshape([1, 1, -1]).cpu().numpy()
    lm3d = lm3d.reshape([-1, 68, 3])

    WH = 224
    # lm3d_224 = (lm3d * WH / 2 + WH / 2).astype(int)
    for i_img in range(len(lm3d)):
        input_pth = frames_names_lst[i_img]
        out_path_np = input_pth.replace(hdtf_frames_path, os.path.join(saving_pth, 'face-centric', 'unposed/')).replace("jpg", "npy")
        if not os.path.exists(os.path.dirname(out_path_np)):
            os.makedirs(os.path.dirname(out_path_np), exist_ok=True)
        np.save(out_path_np, lm3d[i_img])
        # save_lm_img(lm3d_224[i_img][:, :2].astype(int), 'lm3d_224.jpg', WH=WH, flip=True)

    # /media/data/HDTF_preprocessed/30_frame/pvdm/frontal/WRA_ShelleyMooreCapito0_000/03368.jpg
    # ------------------------------------------------------------------------------------------------------------ #
    rotation = compute_rotation_matrix(angle)
    lm3d_rotated = rigid_transform(torch.as_tensor(lm3d), rotation, translation)

    WH = 224
    lms_np_224 = (lm3d_rotated.detach().cpu().numpy() * WH / 2 + WH / 2).astype(int)
    for i_img in range(len(lm3d_rotated)):
        input_pth = frames_names_lst[i_img]
        out_path_np = input_pth.replace(hdtf_frames_path, os.path.join(saving_pth, 'face-centric', 'posed/')).replace("jpg", "npy")
        if not os.path.exists(os.path.dirname(out_path_np)):
            os.makedirs(os.path.dirname(out_path_np), exist_ok=True)
        np.save(out_path_np, lm3d_rotated[i_img].detach().cpu().numpy())
        # save_lm_img(lms_np_224[i_img][:, :2].astype(int), 'lms_t.jpg', WH=WH, flip=True)

    trans_params_imgsize = np.array([trans_param.img_size for trans_param in trans_params])
    trans_params_t = np.array([trans_param.t for trans_param in trans_params])
    trans_params_s = np.array([trans_param.s for trans_param in trans_params])

    lm3d_rotated_scaled = de_aligning_ldmk(trans_params_imgsize, lms_np_224[:, :, :2], trans_params_t, trans_params_s)
    # save_lm_img(lm3d_rotated_scaled[0, :, :2], "lm3d_rotated_scaled.jpg", WH=trans_params_imgsize[0][0], flip=True)

    lm68_fa = lm68_stacked[:, :, :2]
    lm68_fa_np = lm68_fa.detach().cpu().numpy()

    # FLIPPING... Which is SO Important
    B_ = lm68_fa_np.shape[0]
    for b in range(B_):
        lm68_fa_np[b, :, 1] = trans_params_imgsize[b, 1] - 1 - lm68_fa_np[b, :, 1]
        lm3d_rotated_scaled[b, :, 1] = trans_params_imgsize[b, 1] - 1 - lm3d_rotated_scaled[b, :, 1]
        # save_lm_img(lm3d_rotated_scaled[0, :, :2].astype(int), "fa.jpg", WH=int(trans_params_imgsize[0, 1]), flip=False)

    for b in range(B_):
        input_pth = frames_names_lst[b]
        # save_lm_img(lm[b, :, :2], f"lm3d_rotated_scaled_translated_{str(iter)}.jpg", WH=trans_params_imgsize[iter, 0], flip=True)
        out_path_np = input_pth.replace(hdtf_frames_path, os.path.join(saving_pth, 'fa/')).replace("jpg", "npy")
        if not os.path.exists(os.path.dirname(out_path_np)):
            os.makedirs(os.path.dirname(out_path_np), exist_ok=True)
        np.save(out_path_np, lm68_fa_np[b])

    # lm68 y축으로 flip 해주기 (3dmm 2d lmk는 y축이 반대로 되어있어서 그거 맞추기 위함)

    # save_lm_img(lm68_fa[:, :2], f"lm68_fa.jpg", WH=trans_params.img_size[0], flip=True)
    # save_image(input[0].permute(2, 0, 1) / 255, "input.jpg")

    lm5_fa = lm68_2_lm5(lm68_fa_np[:, :, :2])
    lm5_po_np = lm68_2_lm5(lm3d_rotated_scaled[:, :, :2])

    mean_point_temp = np.mean(lm5_fa, axis=1)
    mean_point = np.mean(lm5_po_np, axis=1)
    # save_lm_img(np.expand_dims(mean_point, 0), f"mean_point.jpg", WH=trans_params.img_size[0])
    lm = lm3d_rotated_scaled[:, :, :2] + np.expand_dims(mean_point_temp - mean_point, 1)
    for b in range(B_):
        input_pth = frames_names_lst[b]
        # save_lm_img(lm[iter, :, :2], f"lm3d_rotated_scaled_translated_{str(iter)}.jpg", WH=trans_params_imgsize[iter, 0], flip=True)
        out_path_np = input_pth.replace(hdtf_frames_path, os.path.join(saving_pth, 'non-face-centric', 'posed/')).replace("jpg", "npy")
        if not os.path.exists(os.path.dirname(out_path_np)):
            os.makedirs(os.path.dirname(out_path_np), exist_ok=True)
        np.save(out_path_np, lm[b])


if __name__ == "__main__":
    import random

    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--hdtf_frames_path", type=str, default="/media/data/HDTF_preprocessed/30_frame_re/HDTF/", help="")
    parser.add_argument("--saving_path", type=str, default="/media/data/HDTF_preprocessed/30_frame_re/keypoints/", help="")
    parser.add_argument("--process_id", type=int, default=0, help="")
    parser.add_argument("--total_process", type=int, default=1, help="")
    args = parser.parse_args()

    import os, glob

    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, network_size=4, device="cuda")
    face3d_helper = Face3DHelper("deep_3drecon/BFM")
    face_reconstructor = Reconstructor()
    
    hdtf_frames_path = args.hdtf_frames_path
    saving_pth = args.saving_path
    
    os.makedirs(saving_pth, exist_ok=True)
    os.makedirs(os.path.join(saving_pth, 'non-face-centric'), exist_ok=True)
    os.makedirs(os.path.join(saving_pth, 'face-centric'), exist_ok=True)
    
    iden_list = [iden for iden in os.listdir(hdtf_frames_path) if len(iden.split('_'))==3]
    
    frames_names = []
    for iden in iden_list:
        '''RESUME'''
        frames_pth = [file.split('/')[-1].replace('.jpg','') for file in glob.glob(os.path.join(hdtf_frames_path, iden, "*.jpg"))]
        frames_names += [os.path.join(hdtf_frames_path, iden, file+'.jpg') for file in frames_pth]

    tot_len = len(frames_names)
    frames_names.sort()
    print(tot_len)

    if args.total_process > 1:
        assert args.process_id <= args.total_process - 1
        num_samples_per_process = len(frames_names) // args.total_process
        if args.process_id == args.total_process - 1:
            frames_names = frames_names[args.process_id * num_samples_per_process :]
        else:
            frames_names = frames_names[args.process_id * num_samples_per_process : (args.process_id + 1) * num_samples_per_process]

    batched_frames_names_lst = chunk(frames_names, chunk_size=args.total_process)
    
    for batched_frames_names in tqdm(batched_frames_names_lst, desc="[ROOT]: extracting face mesh and 3DMM in batches...", position=0, leave=False):
        try:
            cnt = 0
            for frame_name in batched_frames_names:
                input = cv2.imread(frame_name)
                input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)

                lm68 = fa.get_landmarks(input)[0]  # [68,2]

                input = torch.from_numpy(input)
                lm68 = torch.from_numpy(lm68)

                # stacking
                if cnt == 0:
                    first_img_size = input.shape[0]
                    input_stacked = input.unsqueeze(0)
                    lm68_stacked = lm68.unsqueeze(0)
                else:
                    img_size = input.shape[0]
                    if first_img_size != img_size:
                        continue
                    input_stacked = torch.cat([input_stacked, input.unsqueeze(0)], dim=0)
                    lm68_stacked = torch.cat([lm68_stacked, lm68.unsqueeze(0)], dim=0)

                cnt += 1
            
            processing(batched_frames_names, input_stacked, lm68_stacked, face3d_helper, hdtf_frames_path, saving_pth)
        except Exception as e:
            print(e)
            continue
