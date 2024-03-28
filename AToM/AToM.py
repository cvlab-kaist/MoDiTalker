import multiprocessing
import os
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.state import AcceleratorState
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.atom_dataset import LRS3SeqDataset
from dataset.preprocess import increment_path
from model.adan import Adan
from model.diffusion import GaussianDiffusion
from model.model import MotionDecoder
import cv2

from data_util.face3d_helper import Face3DHelper

def wrap(x):
    return {f"module.{key}": value for key, value in x.items()}


def maybe_wrap(x, num):
    return x if num == 1 else wrap(x)


class AToM:
    def __init__(
        self,
        feature_type,
        checkpoint_path="",
        normalizer=None,
        EMA=True,
        learning_rate=4e-4,
        weight_decay=0.02,
    ):
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        state = AcceleratorState()
        num_processes = state.num_processes
        use_baseline_feats = feature_type == "baseline"
        self.checkpoint_num = checkpoint_path.split("/")[-1].split(".")[0].split("-")[-1] if checkpoint_path is not None else ""  
        repr_dim = 204
        self.repr_dim = 204
        feature_dim = 1024
        self.horizon = horizon = 156

        self.accelerator.wait_for_everyone()

        if checkpoint_path != "":
            checkpoint = torch.load(checkpoint_path, map_location=self.accelerator.device)

        model = MotionDecoder(
            nfeats=repr_dim,
            seq_len=horizon,
            latent_dim=512,
            ff_size=1024,
            num_layers=8,
            num_heads=8,
            dropout=0.1,
            cond_feature_dim=feature_dim,
            activation=F.gelu,
        )

        diffusion = GaussianDiffusion(
            model,
            horizon,
            repr_dim,
            schedule="cosine",
            n_timestep=1000,
            predict_epsilon=False,
            loss_type="l2",
            use_p2=False,
            cond_drop_prob=0.25,
            guidance_weight=2,
        )

        self.face3d_helper = Face3DHelper("../data/data_utils/deep_3drecon/BFM")

        print("Model has {} parameters".format(sum(y.numel() for y in model.parameters())))

        self.model = self.accelerator.prepare(model)
        self.diffusion = diffusion.to(self.accelerator.device)
        optim = Adan(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        self.optim = self.accelerator.prepare(optim)

        if checkpoint_path != "":
            self.model.load_state_dict(
                maybe_wrap(
                    checkpoint["ema_state_dict" if EMA else "model_state_dict"],
                    num_processes,
                )
            )

    def eval(self):
        self.diffusion.eval()

    def train(self):
        self.diffusion.train()

    def prepare(self, objects):
        return self.accelerator.prepare(*objects)

    def train_loop(self, opt):
        train_data_loader = LRS3SeqDataset("train").get_dataloader(opt.batch_size)
        test_data_loader = LRS3SeqDataset("val").get_dataloader(opt.batch_size)
       
        train_data_loader = self.accelerator.prepare(train_data_loader)
        load_loop = partial(tqdm, position=1, desc="Batch") if self.accelerator.is_main_process else lambda x: x
        if self.accelerator.is_main_process:
            save_dir = str(increment_path(Path(opt.project) / opt.exp_name))
            opt.exp_name = save_dir.split("/")[-1]
            save_dir = Path(save_dir)
            wdir = save_dir / "weights"
            wdir.mkdir(parents=True, exist_ok=True)

        device = self.accelerator.device
        self.accelerator.wait_for_everyone()
        for epoch in range(1, opt.epochs + 1):
            avg_loss = 0
            avg_vloss = 0
         
            self.train()

            """
            item_id :        filename
            mel              [B, 312, 80]
            hubert           [B, 312, 1024]
            x_mask           [B, 312]
            exp              [B, 156, 64]
            pose             [B, 156, 7]
            y_mask           [B, 156]
            idexp_lm3d       [B, 156, 204]     
            ref_mean_lm3d    [B, 204]         -1~1?
            mouth_idexp_lm3d [B, 156, 60]
            f0               [B, 312]
            """
            for step, (batch) in enumerate(load_loop(train_data_loader)):
                
                if batch["hubert"].shape[1] == 304:
                    continue

                for k, v in batch.items():
                    if k == "item_id":
                        continue
                    batch[k] = v.to(device)

                x = batch["idexp_lm3d"]
                x_pos = batch["pose"]
                batch_size = x.shape[0]
                # ------------------------------------------------------------------------------------------------ #
                cond_keypoint = x[:, 0:1, :]
                cond_keypoint = torch.cat([cond_keypoint for _ in range(self.horizon)], dim=1) 
                cond = batch["hubert"]
                # ------------------------------------------------------------------------------------------------ #
                x_ldmk = x.view(batch_size, self.horizon, -1, 3)  
                cond_keypoint_ldmk = cond_keypoint.view(batch_size, self.horizon, -1, 3) 
                residual_ldmk = x_ldmk - cond_keypoint_ldmk 
                residual = residual_ldmk.view(batch_size, self.horizon, -1)

                total_loss, (loss, v_loss) = self.diffusion(residual, x_pos, cond_keypoint, cond, t_override=None)
                self.optim.zero_grad()
                self.accelerator.backward(total_loss)
                self.optim.step()

                if self.accelerator.is_main_process:
                    avg_loss += loss.detach().cpu().numpy()
                    avg_vloss += v_loss.detach().cpu().numpy()
                    if step % opt.ema_interval == 0:
                        self.diffusion.ema.update_model_average(self.diffusion.master_model, self.diffusion.model)

                if step % 2000 == 0:
                    if (epoch % opt.save_interval) == 0:
                        self.accelerator.wait_for_everyone()
                        if self.accelerator.is_main_process:
                            self.eval()
                            avg_loss /= len(train_data_loader)
                            avg_vloss /= len(train_data_loader)
                            log_dict = {
                                "Train Loss": avg_loss,
                                "V Loss": avg_vloss,
                            }
                            ckpt = {
                                "ema_state_dict": self.diffusion.master_model.state_dict(),
                                "model_state_dict": self.accelerator.unwrap_model(self.model).state_dict(),
                                "optimizer_state_dict": self.optim.state_dict(),
                            }
                            if self.checkpoint_num == "":
                                torch.save(ckpt, os.path.join(wdir, f"train-{epoch}.pt"))
                            else:
                                torch.save(ckpt, os.path.join(wdir, f"train-{epoch + int(self.checkpoint_num)}.pt"))

                            (batch) = next(iter(test_data_loader))
                            render_count = batch["hubert"].shape[0]
                            shape = (render_count, self.horizon, self.repr_dim)
                            print("Generating Sample")

                            if batch["hubert"].shape[1] == 304:
                                continue
                            for k, v in batch.items():
                                if k == "item_id":
                                    continue
                                batch[k] = v.to(device)

                            x = batch["idexp_lm3d"][:render_count]
                            x_pos = batch["pose"][:render_count]

                            cond_keypoint = x[:, 0:1, :][:render_count]
                            cond_keypoint = torch.cat([cond_keypoint for _ in range(self.horizon)], dim=1)  
                            cond = batch["hubert"][:render_count] 
                            batch_size = x.shape[0]
                            x_ldmk = x.view(batch_size, self.horizon, -1, 3) 
                            cond_keypoint_ldmk = cond_keypoint.view(batch_size, self.horizon, -1, 3)  
                            residual_ldmk = x_ldmk - cond_keypoint_ldmk 
                            residual = residual_ldmk.view(batch_size, self.horizon, -1)

                            _ = self.diffusion.render_sample(
                                self.face3d_helper,
                                shape,
                                cond_keypoint,
                                residual,  
                                x_pos,
                                residual, 
                                cond,
                                epoch,
                                os.path.join(opt.render_dir, "train_" + opt.exp_name),
                                name=batch["item_id"][:render_count],
                                sound=True,
                            )

                    print(f"[MODEL SAVED at Epoch {epoch}]")

    def render_sample(self, data_tuple, label, render_dir, render_count=-1, fk_out=None, render=True):
        _, cond, wavname = data_tuple
        assert len(cond.shape) == 3
        if render_count < 0:
            render_count = len(cond)
        shape = (render_count, self.horizon, self.repr_dim)
        cond = cond.to(self.accelerator.device)
        self.diffusion.render_sample(
            shape,
            cond[:render_count],
            self.normalizer,
            label,
            render_dir,
            name=wavname[:render_count],
            sound=True,
            mode="long",
            fk_out=fk_out,
            render=render,
        )
