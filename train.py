import os
from posixpath import join

import argparse
import logging
import torch
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
import random
from datetime import datetime

from torch.utils.data.dataloader import DataLoader
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from utils import AverageMeter, calc_psnr, calc_ssim, mixing_noise, requires_grad
from dataset import Dataset
from PIL import Image
from models.loss import d_logistic_loss, g_nonsaturating_loss
from models.model import Discriminator, Generator

""" DDP (Distributed data parallel) """
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


""" 로그 설정 """
logger = logging.getLogger(__name__)
logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)


def setup(rank, world_size):
    """DDP 디바이스 설정"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    # initialize the process group
    dist.init_process_group("NCCL", rank=rank, world_size=world_size)


def cleanup():
    """Kill DDP process group"""
    dist.destroy_process_group()


def gan_trainer(
    train_dataloader,
    eval_dataloader,
    generator,
    discriminator,
    generator_optimizer,
    discriminator_optimizer,
    epoch,
    best_ssim,
    device,
    writer,
    args,
):
    generator.train()
    discriminator.train()

    """ latent z 랜덤 생성 """
    sample_z = torch.randn(args.batch_size, args.style_dims, device=device)

    """ Losses average meter 설정 """
    d_losses = AverageMeter(name="D Loss", fmt=":.6f")
    g_losses = AverageMeter(name="G Loss", fmt=":.6f")

    """ 모델 평가 measurements 설정 """
    psnr = AverageMeter(name="PSNR", fmt=":.6f")
    ssim = AverageMeter(name="SSIM", fmt=":.6f")
    fake_score = AverageMeter(name="fake_score", fmt=":.6f")
    real_score = AverageMeter(name="real_score", fmt=":.6f")
    psnr = AverageMeter(name="PSNR", fmt=":.6f")

    start = datetime.now()

    """  트레이닝 Epoch 시작 """
    for i, hr in enumerate(train_dataloader):
        """LR & HR 디바이스 설정"""
        real_img = hr.to(device)

        """============= 식별자 학습 ============="""
        # requires_grad(generator, False)
        # requires_grad(discriminator, True)

        """ 식별자 최적화 초기화 """
        discriminator_optimizer.zero_grad()

        """추론"""
        noise = mixing_noise(args.batch_size, args.style_dims, args.mixing, device)
        fake_img, _ = generator(noise)

        """ 식별자 통과 후 loss 계산 """
        real_output = discriminator(real_img)
        fake_output = discriminator(fake_img.detach())
        d_loss = d_logistic_loss(real_output, fake_output)
        real_score.update(real_output.mean(), hr.size(0))
        fake_score.update(fake_output.mean(), hr.size(0))

        """ 가중치 업데이트 """
        d_loss.backward()
        discriminator_optimizer.step()

        """============= 생성자 학습 ============="""
        # requires_grad(generator, True)
        # requires_grad(discriminator, False)

        """ 생성자 최적화 초기화 """
        generator_optimizer.zero_grad()

        """추론"""
        noise = mixing_noise(args.batch_size, args.style_dims, args.mixing, device)

        """ 식별자 통과 후 loss 계산 """
        fake_output, _ = generator(noise)
        fake_pred = discriminator(fake_output)
        g_loss = g_nonsaturating_loss(fake_pred)

        """ 가중치 업데이트 """
        g_loss.backward()
        generator_optimizer.step()

        """ 생성자 초기화 """
        generator.zero_grad()

        """ loss 업데이트 """
        d_losses.update(d_loss.item(), hr.size(0))
        g_losses.update(g_loss.item(), hr.size(0))

    """  테스트 Epoch 시작 """
    generator.eval()
    with torch.no_grad():
        for i, hr in enumerate(eval_dataloader):
            hr = hr.to(device)
            preds, _ = generator([sample_z])

            print(f'hr : {hr.shape}, preds : {preds.shape}')
            """ 1 epoch 마다 테스트 이미지 확인 """
            if i == 0:
                vutils.save_image(
                    hr.detach(), os.path.join(args.outputs_dir, f"HR_{epoch}.jpg")
                )
                vutils.save_image(
                    preds.detach(), os.path.join(args.outputs_dir, f"preds_{epoch}.jpg")
                )
            psnr.update(calc_psnr(preds, hr), len(hr))
            ssim.update(calc_ssim(preds, hr).mean(), len(hr))

    if args.distributed and device == 0:
        """Generator 모델 저장"""
        if ssim.avg > best_ssim:
            best_ssim = ssim.avg
            torch.save(
                generator.module.state_dict(),
                os.path.join(args.outputs_dir, "best_g.pth"),
            )
        if epoch % 1 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": generator.module.state_dict(),
                    "optimizer_state_dict": generator_optimizer.state_dict(),
                    "best_ssim": best_ssim,
                },
                os.path.join(args.outputs_dir, "g_epoch_{}.pth".format(epoch)),
            )

    if not args.distributed:
        """Generator 모델 저장"""
        if ssim.avg > best_ssim:
            best_ssim = ssim.avg
            torch.save(
                generator.state_dict(),
                os.path.join(args.outputs_dir, "best_g.pth"),
            )

        if epoch % 1 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": generator.state_dict(),
                    "optimizer_state_dict": generator_optimizer.state_dict(),
                    "best_ssim": best_ssim,
                },
                os.path.join(args.outputs_dir, "g_epoch_{}.pth".format(epoch)),
            )

    """Epoch 1번에 1번 저장"""
    if epoch % 1 == 0:
        """Discriminator 모델 저장"""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": discriminator.state_dict(),
                "optimizer_state_dict": discriminator_optimizer.state_dict(),
            },
            os.path.join(args.outputs_dir, "d_epoch_{}.pth".format(epoch)),
        )

    if device == 0:
        """1 epoch 마다 텐서보드 업데이트"""
        writer.add_scalar("d_Loss/train", d_losses.avg, epoch)
        writer.add_scalar("g_Loss/train", g_losses.avg, epoch)

        writer.add_scalar("real_score/train", real_score.avg, epoch)
        writer.add_scalar("fake_score/train", fake_score.avg, epoch)

        """ 1 epoch 마다 텐서보드 업데이트 """
        writer.add_scalar("psnr/test", psnr.avg, epoch)
        writer.add_scalar("ssim/test", ssim.avg, epoch)

        print("Training complete in: " + str(datetime.now() - start))


def main_worker(gpu, args):
    if args.distributed:
        args.rank = args.nr * args.gpus + gpu
        setup(args.rank, args.world_size)

    """ GPEN 모델 설정 """
    generator = Generator(
        size=args.patch_size,
        style_dim=args.style_dims,
        n_mlp=args.mlp,
        channel_multiplier=args.channel_multiplier,
        narrow=args.narrows,
        # isconcat=args.is_concat
    ).to(gpu)
    discriminator = Discriminator(
        args.patch_size, channel_multiplier=args.channel_multiplier, narrow=args.narrows
    ).to(gpu)

    """ regularzation ratio 설정 """
    d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)
    g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)

    """ Optimizer 설정 """
    generator_optimizer = torch.optim.Adam(
        generator.parameters(),
        lr=args.lr * g_reg_ratio,
        betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
    )

    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(),
        lr=args.lr * d_reg_ratio,
        betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
    )

    """ epoch & PSNR 설정 """
    g_epoch = 0
    d_epoch = 0
    best_ssim = 0

    """ 체크포인트 weight 불러오기 """
    if os.path.exists(args.resume_g):
        checkpoint_g = torch.load(args.resume_g)
        generator.load_state_dict(checkpoint_g["model_state_dict"])
        g_epoch = checkpoint_g["epoch"] + 1
        generator_optimizer.load_state_dict(checkpoint_g["optimizer_state_dict"])
    if os.path.exists(args.resume_d):
        """resume discriminator"""
        checkpoint_d = torch.load(args.resume_d)
        discriminator.load_state_dict(checkpoint_d["model_state_dict"])
        discriminator_optimizer.load_state_dict(checkpoint_d["optimizer_state_dict"])

    """ 데이터셋 설정 """
    train_dataset = Dataset(args.train_dir, args.patch_size)
    eval_dataset = Dataset(args.eval_dir, args.patch_size)
    train_sampler = None

    if args.distributed:
        generator = DDP(generator, device_ids=[gpu])
        """ 데이터셋 & 데이터셋 설정 """
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=args.rank
        )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=True,
        sampler=train_sampler,
    )

    eval_dataloader = DataLoader(
        dataset=eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )

    if gpu == 0 or not args.distributed:
        """로그 인포 프린트 하기"""
        logger.info(
            f"GPEN MODEL INFO:\n"
            f"GPEN TRAINING INFO:\n"
            f"\tTotal Epoch:                   {args.num_epochs}\n"
            f"\tStart generator Epoch:         {g_epoch}\n"
            f"\tStart discrimnator Epoch:      {d_epoch}\n"
            f"\tTrain directory path:          {args.train_dir}\n"
            f"\tTest directory path:           {args.eval_dir}\n"
            f"\tOutput weights directory path: {args.outputs_dir}\n"
            f"\tGAN learning rate:             {args.lr}\n"
            f"\tPatch size:                    {args.patch_size}\n"
            f"\tBatch size:                    {args.batch_size}\n"
        )

    """텐서보드 설정"""
    writer = SummaryWriter(args.outputs_dir)

    """GAN Training"""
    for epoch in range(g_epoch, args.num_epochs):
        gan_trainer(
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            generator=generator,
            discriminator=discriminator,
            generator_optimizer=generator_optimizer,
            discriminator_optimizer=discriminator_optimizer,
            epoch=epoch,
            best_ssim=best_ssim,
            device=gpu,
            writer=writer,
            args=args,
        )


# 616210 CUDA_VISIBLE_DEVICES=2 nohup python3 train.py --train-dir /dataset/FFHQ --eval-dir /dataset/FFHQ_test/ --outputs-dir weights_stylegan2 --batch-size 12 --patch-size 256 --num-epoch 20 --is-concat &
if __name__ == "__main__":
    """로그 설정"""
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO)

    """ Argparse 설정 """
    parser = argparse.ArgumentParser()

    """data args setup"""
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--eval-dir", type=str, required=True)
    parser.add_argument("--outputs-dir", type=str, required=True)

    """model args setup"""
    parser.add_argument(
        "--n-sample",
        type=int,
        default=64,
        help="number of the samples generated during training",
    )
    parser.add_argument(
        "--d-reg-every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g-reg-every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument("--is-concat", action="store_true")
    parser.add_argument("--style-dims", type=int, default=512)
    parser.add_argument("--mlp", type=int, default=8)
    parser.add_argument("--channel-multiplier", type=int, default=1)
    parser.add_argument("--narrows", type=float, default=0.5)

    """Training details args setup"""
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-epochs", type=int, default=1000)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--resume-g", type=str, default="generator.pth")
    parser.add_argument("--resume-d", type=str, default="discriminator.pth")
    parser.add_argument("--patch-size", type=int, default=256)

    """ Distributed data parallel setup"""
    parser.add_argument("-n", "--nodes", default=1, type=int, metavar="N")
    parser.add_argument(
        "-g",
        "--gpus",
        default=0,
        type=int,
        help="if DDP, number of gpus per node or if not ddp, gpu number",
    )
    parser.add_argument(
        "-nr", "--nr", default=0, type=int, help="ranking within the nodes"
    )
    parser.add_argument("--distributed", action="store_true")
    args = parser.parse_args()

    """ weight를 저장 할 경로 설정 """
    args.outputs_dir = os.path.join(args.outputs_dir, f"StyleGAN2")
    if not os.path.exists(args.outputs_dir):
        os.makedirs(args.outputs_dir)

    """ Seed 설정 """
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    """ GPU 디바이스 설정 """
    cudnn.benchmark = True
    cudnn.deterministic = True

    if args.distributed:
        gpus = gpus = torch.cuda.device_count()
        args.world_size = gpus * args.nodes
        mp.spawn(main_worker, nprocs=gpus, args=(args,), join=True)
    else:
        main_worker(args.gpus, args)
