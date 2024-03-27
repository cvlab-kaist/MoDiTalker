import os
import random
import numpy as np
import sys

sys.path.extend([sys.path[0][:-4], "/app"])
import pdb
import time
from tqdm import tqdm
import copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from utils import AverageMeter
from evals.eval import test_psnr, test_ifvd, test_fvd_ddpm
from models.ema import LitEma
from einops import rearrange
from torch.optim.lr_scheduler import LambdaLR
from torchvision import transforms

def latentDDPM(
    rank,
    first_stage_model,
    model,
    opt,
    criterion,
    train_loader,
    test_loader,
    scheduler,
    ema_model=None,
    cond_prob=0.3,
    logger=None,
):
    scaler = GradScaler()

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    if rank == 0:
        rootdir = logger.logdir

    device = torch.device("cuda", rank)

    losses = dict()
    losses["diffusion_loss"] = AverageMeter()
    check = time.time()

    
    if ema_model == None:
        ema_model = copy.deepcopy(model)
        ema = LitEma(ema_model)
        ema_model.eval()
    else:
        ema = LitEma(ema_model)
        ema.num_updates = torch.tensor(11200, dtype=torch.int)
        ema_model.eval()

    first_stage_model, first_stage_model_ldmk = first_stage_model
    first_stage_model.eval()
    first_stage_model_ldmk.eval()
    model.train()
    
        
    for it, (x_ref, x, x_l, masked_x, _) in tqdm(enumerate(train_loader)):
        """
        ref; ref frame (.jpg rgb file)
        x; video (.jpg rgb file)
        l; landmark (.png rgb file)
        """
        x_ref = x_ref.to(device)
        x = x.to(device)
        x_l = x_l.to(device)
        masked_x = masked_x.to(device)
        
        x_ref = rearrange(x_ref / 127.5 - 1, "b t c h w -> b c t h w")  # videos
        x = rearrange(x / 127.5 - 1, "b t c h w -> b c t h w")  # videos
        x_l = rearrange(x_l / 127.5 - 1, "b t c h w -> b c t h w")  # videos
        masked_x = rearrange(masked_x / 127.5 - 1, "b t c h w -> b c t h w")  # videos

        c = None
        # conditional free guidance training
        model.zero_grad()

        with autocast():
            with torch.no_grad():
                image_cond = first_stage_model.extract(x_ref).detach()  # torch.Size([4, 4, 2048])
                image_cond = image_cond[:, :, 0 : 32 * 32]  # torch.Size([4, 4, 1024])
                z = first_stage_model.extract(x).detach()
                z_l = first_stage_model_ldmk.extract(x_l).detach()
                masked_z = first_stage_model.extract(masked_x).detach()

        c = torch.cat([z_l, masked_z], dim=1)
        (loss, t), loss_dict = criterion(x=z.float(), cond=c.float(), image_cond=image_cond.float())
                
        """
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        """
        
        loss.backward()
        opt.step()

        losses["diffusion_loss"].update(loss.item(), 1)

        # ema model
        if it % 25 == 0 and it > 0:
            ema(model)
        if it % 500 == 0:
            if logger is not None and rank == 0:
                logger.scalar_summary("train/diffusion_loss", losses["diffusion_loss"].average, it)

                log_("[Time %.3f] [Diffusion %f]" % (time.time() - check, losses["diffusion_loss"].average))

            losses = dict()
            losses["diffusion_loss"] = AverageMeter()

        if it % 1000 == 0 and rank == 0:
            ema.copy_to(ema_model)
            torch.save(ema_model.state_dict(), rootdir + f"/ema_model_{it}.pth")
            fvd = test_fvd_ddpm(rank, ema_model, [first_stage_model, first_stage_model_ldmk], test_loader, it, logger)

            if logger is not None and rank == 0:
                logger.scalar_summary("test/fvd", fvd, it)

                log_("[Time %.3f] [FVD %f]" % (time.time() - check, fvd))



def first_stage_train(rank, model, opt, d_opt, criterion, train_loader, test_loader, first_model, fp, logger=None):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log
    if rank == 0:
        rootdir = logger.logdir

    device = torch.device("cuda", rank)

    losses = dict()
    losses["ae_loss"] = AverageMeter()
    losses["d_loss"] = AverageMeter()
    check = time.time()

    accum_iter = 3
    disc_opt = False

    if fp:
        scaler = GradScaler()
        scaler_d = GradScaler()

        try:
            scaler.load_state_dict(torch.load(os.path.join(first_model, "scaler.pth")))
            scaler_d.load_state_dict(torch.load(os.path.join(first_model, "scaler_d.pth")))
        except:
            print("Fail to load scalers. Start from initial point.")

    model.train()
    disc_start = criterion.discriminator_iter_start

    for it, (x_ref, x, x_l, masked_x, _) in tqdm(enumerate(train_loader)):
        if type == "x":
            x_l = x

        if it > 1000000:
            break
        batch_size = x.size(0)
        x = x.to(device)
        x = rearrange(x / 127.5 - 1, "b t c h w -> b c t h w")  # videos
        

        if not disc_opt:  # without Gan loss
            with autocast():  
                x_tilde, vq_loss = model(x)

                if it % accum_iter == 0:
                    model.zero_grad()
                ae_loss = criterion(
                    vq_loss,
                    x,
                    rearrange(x_tilde, "(b t) c h w -> b c t h w", b=batch_size),
                    optimizer_idx=0,
                    global_step=it,
                )
                ae_loss = ae_loss / accum_iter

            scaler.scale(ae_loss).backward()
            if it % accum_iter == accum_iter - 1:
                scaler.step(opt)
                scaler.update()

            losses["ae_loss"].update(ae_loss.item(), 1)

        else:
            if it % accum_iter == 0:
                criterion.zero_grad()

            with autocast():
                with torch.no_grad():
                    x_tilde, vq_loss = model(x)
                d_loss = criterion(
                    vq_loss,
                    x,
                    rearrange(x_tilde, "(b t) c h w -> b c t h w", b=batch_size),
                    optimizer_idx=1,
                    global_step=it,
                )
                d_loss = d_loss / accum_iter

            scaler_d.scale(d_loss).backward()

            if it % accum_iter == accum_iter - 1:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler_d.unscale_(d_opt)
                torch.nn.utils.clip_grad_norm_(criterion.discriminator_2d.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(criterion.discriminator_3d.parameters(), 1.0)
                scaler_d.step(d_opt)
                scaler_d.update()

            losses["d_loss"].update(d_loss.item() * 3, 1)

        if it % accum_iter == accum_iter - 1 and it // accum_iter >= disc_start:
            if disc_opt:
                disc_opt = False
            else:
                disc_opt = True

        if it % 2000 == 0:
            fvd = test_ifvd(rank, model, test_loader, it, logger, "x")
            psnr = test_psnr(rank, model, test_loader, it, logger, "x")
            if logger is not None and rank == 0:
                logger.scalar_summary("train/ae_loss", losses["ae_loss"].average, it)
                logger.scalar_summary("train/d_loss", losses["d_loss"].average, it)
                logger.scalar_summary("test/psnr", psnr, it)
                logger.scalar_summary("test/fvd", fvd, it)

                log_(
                    "[Time %.3f] [AELoss %f] [DLoss %f] [PSNR %f]"
                    % (
                        time.time() - check,
                        losses["ae_loss"].average,
                        losses["d_loss"].average,
                        psnr,
                    )
                )

                torch.save(model.state_dict(), rootdir + f"model_last.pth")
                # torch.save(criterion.module.state_dict(), rootdir + f"loss_last.pth")
                torch.save(criterion.state_dict(), rootdir + f"loss_last.pth")
                torch.save(opt.state_dict(), rootdir + f"opt.pth")
                torch.save(d_opt.state_dict(), rootdir + f"d_opt.pth")
                torch.save(scaler.state_dict(), rootdir + f"scaler.pth")
                torch.save(scaler_d.state_dict(), rootdir + f"scaler_d.pth")

            losses = dict()
            losses["ae_loss"] = AverageMeter()
            losses["d_loss"] = AverageMeter()  # If you do not use discriminator loss, nothing will appear


        # if it % 2000 == 0 and rank == 0:
        # torch.save(model.state_dict(), rootdir + f"model_{it}.pth")


def first_stage_x_l_train(rank, model, opt, d_opt, criterion, train_loader, test_loader, first_model, fp, logger=None):
    if logger is None:
        log_ = print
    else:
        log_ = logger.log
    if rank == 0:
        rootdir = logger.logdir

    device = torch.device("cuda", rank)

    losses = dict()
    losses["ae_loss"] = AverageMeter()
    losses["d_loss"] = AverageMeter()
    check = time.time()

    accum_iter = 3
    disc_opt = False

    if fp:
        scaler = GradScaler()
        scaler_d = GradScaler()

        try:
            scaler.load_state_dict(torch.load(os.path.join(first_model, "scaler.pth")))
            scaler_d.load_state_dict(torch.load(os.path.join(first_model, "scaler_d.pth")))
        except:
            print("Fail to load scalers. Start from initial point.")

    # 230425
    # model = torch.nn.DataParallel(model, output_device=7)

    model.train()
    lst = set([name.split('.')[0] for name, param in model.named_parameters()])
    

    for name, param in model.named_parameters():
        if "decoder" in name.split("."):
            param.requires_grad = False
            print(name, param.requires_grad)
        else:
            param.requires_grad = True
            print(name, param.requires_grad)
    # disc_start = criterion.module.discriminator_iter_start
    disc_start = criterion.discriminator_iter_start

    for it, (x_ref, x, x_l, masked_x, _) in enumerate(train_loader):
        print("it; ", it)
        if it > 1000000:
            break
        batch_size = x.size(0)
        x_l = x_l.to(device)
        x_l = rearrange(x_l / 127.5 - 1, "b t c h w -> b c t h w")  # videos

        if not disc_opt:  # without Gan loss
            with autocast():  # fp16으로 바꿔줌
                x_tilde, vq_loss = model(x_l)
                if it % accum_iter == 0:
                    model.zero_grad()
                ae_loss = criterion(
                    vq_loss,
                    x_l,
                    rearrange(x_tilde, "(b t) c h w -> b c t h w", b=batch_size),
                    optimizer_idx=0,
                    global_step=it,
                )
                ae_loss = ae_loss / accum_iter

            scaler.scale(ae_loss).backward()
            if it % accum_iter == accum_iter - 1:
                scaler.step(opt)
                scaler.update()

            losses["ae_loss"].update(ae_loss.item(), 1)

        else:
            if it % accum_iter == 0:
                criterion.zero_grad()

            with autocast():
                with torch.no_grad():
                    x_tilde, vq_loss = model(x)
                d_loss = criterion(
                    vq_loss,
                    x_l,
                    rearrange(x_tilde, "(b t) c h w -> b c t h w", b=batch_size),
                    optimizer_idx=1,
                    global_step=it,
                )
                d_loss = d_loss / accum_iter

            scaler_d.scale(d_loss).backward()

            if it % accum_iter == accum_iter - 1:
                # Unscales the gradients of optimizer's assigned params in-place
                scaler_d.unscale_(d_opt)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(criterion.module.discriminator_2d.parameters(), 1.0)
                torch.nn.utils.clip_grad_norm_(criterion.module.discriminator_3d.parameters(), 1.0)

                scaler_d.step(d_opt)
                scaler_d.update()

            losses["d_loss"].update(d_loss.item() * 3, 1)

        if it % accum_iter == accum_iter - 1 and it // accum_iter >= disc_start:
            if disc_opt:
                disc_opt = False
            else:
                disc_opt = True

        if it % 2000 == 0:
            fvd = test_ifvd(rank, model, test_loader, it, logger, "xl")
            psnr = test_psnr(rank, model, test_loader, it, logger, "xl")
            if logger is not None and rank == 0:
                logger.scalar_summary("train/ae_loss", losses["ae_loss"].average, it)
                logger.scalar_summary("train/d_loss", losses["d_loss"].average, it)
                logger.scalar_summary("test/psnr", psnr, it)
                logger.scalar_summary("test/fvd", fvd, it)

                log_(
                    "[Time %.3f] [AELoss %f] [DLoss %f] [PSNR %f]"
                    % (
                        time.time() - check,
                        losses["ae_loss"].average,
                        losses["d_loss"].average,
                        psnr,
                    )
                )

                torch.save(model.state_dict(), rootdir + f"/model_last_{it}.pth")
                torch.save(criterion.state_dict(), rootdir + f"/loss_last_{it}.pth")

            losses = dict()
            losses["ae_loss"] = AverageMeter()
            losses["d_loss"] = AverageMeter() 

        # if it % 2000 == 0 and rank == 0:
        # torch.save(model.state_dict(), rootdir + f"model_{it}.pth")
