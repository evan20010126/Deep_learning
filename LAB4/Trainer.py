import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader

from modules import Generator, Gaussian_Predictor, Decoder_Fusion, Label_Encoder, RGB_Encoder

from dataloader import Dataset_Dance
from torchvision.utils import save_image
import random
import torch.optim as optim
from torch import stack

from tqdm import tqdm
import imageio

import matplotlib.pyplot as plt
from math import log10


def write_record(root, line, changeLine=True):
    with open(root+"/record.txt", 'a') as f:
        if changeLine:
            f.write(line + '\n')
        else:
            f.write(line + ' ')


def Generate_PSNR(imgs1, imgs2, data_range=1.):
    """PSNR for torch tensor"""
    mse = nn.functional.mse_loss(
        imgs1, imgs2)  # wrong computation for batch size > 1
    psnr = 20 * log10(data_range) - 10 * torch.log10(mse)
    return psnr


def kl_criterion(mu, logvar, batch_size):
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    KLD /= batch_size
    return KLD


class kl_annealing():
    def __init__(self, args, current_epoch=0):
        # TODO
        self.args = args
        self.current_epoch = current_epoch
        self.kl_anneal_type = args.kl_anneal_type
        self.kl_anneal_cycle = args.kl_anneal_cycle
        self.kl_anneal_ratio = args.kl_anneal_ratio

        self.kl_anneal_percent = args.kl_anneal_percent
        self.ratio_values = self.make_ratio_list(self.kl_anneal_type,
                                                 self.kl_anneal_ratio, 1.0, self.kl_anneal_cycle, self.kl_anneal_percent)

    def make_ratio_list(self, kl_anneal_type, init_ratio, stop_ratio, epochs_per_cycle, percent):
        increase_step = (stop_ratio - init_ratio) / \
            (epochs_per_cycle * percent)
        values = [init_ratio]
        tmp_ratio = init_ratio
        while (values[-1] < stop_ratio):
            tmp_ratio += increase_step
            tmp_ratio = min(max(init_ratio, tmp_ratio), stop_ratio)
            values.append(tmp_ratio)
            print(tmp_ratio)
        for i in range(int(epochs_per_cycle * (1 - percent))):
            print(values[-1])
            values.append(values[-1])
        return values

    def update(self):
        # TODO
        if self.kl_anneal_type == 'Cyclical':
            self.kl_anneal_ratio = self.ratio_values[self.current_epoch % len(
                self.ratio_values)]
        elif self.kl_anneal_type == 'Monotonic':
            if len(self.ratio_values) <= self.current_epoch:
                self.kl_anneal_ratio = 1.0
            else:
                self.kl_anneal_ratio = self.ratio_values[self.current_epoch]
        elif self.kl_anneal_type == 'Without':
            self.kl_anneal_ratio = 1.0

        self.current_epoch += 1

    def get_beta(self):
        # TODO
        return self.kl_anneal_ratio

    def frange_cycle_linear(self, n_iter, start=0.0, stop=1.0,  n_cycle=1, ratio=1):
        # TODO
        # n_iter: 迭代的次數
        # start: 數值的初始值
        # stop: 數值的終止值
        # n_cycle: 循環的周期數
        # ratio: 數值的變化速率
        num_values = n_cycle * n_iter
        values = []
        for i in range(num_values):
            cycle_progress = i / n_iter  # 循環內的進度 (0.0 到 1.0)
            linear_interpolation = start + (stop - start) * cycle_progress
            values.append(linear_interpolation)
        values = np.array(values)
        values = np.clip(values, start, stop)  # 確保數值在指定範圍內
        values = values * ratio  # 按照比例縮放數值，這可能是相關變量的倍數

        return values


class VAE_Model(nn.Module):
    def __init__(self, args):
        super(VAE_Model, self).__init__()
        self.args = args

        # Modules to transform image from RGB-domain to feature-domain
        self.frame_transformation = RGB_Encoder(3, args.F_dim)
        self.label_transformation = Label_Encoder(3, args.L_dim)

        # Conduct Posterior prediction in Encoder
        self.Gaussian_Predictor = Gaussian_Predictor(
            args.F_dim + args.L_dim, args.N_dim)
        self.Decoder_Fusion = Decoder_Fusion(
            args.F_dim + args.L_dim + args.N_dim, args.D_out_dim)

        # Generative model
        self.Generator = Generator(input_nc=args.D_out_dim, output_nc=3)

        self.optim = optim.Adam(self.parameters(), lr=self.args.lr)
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optim, milestones=[2, 5], gamma=0.1)
        self.kl_annealing = kl_annealing(args, current_epoch=0)
        self.mse_criterion = nn.MSELoss()
        self.current_epoch = 0

        # Teacher forcing arguments
        self.tfr = args.tfr
        self.tfr_d_step = args.tfr_d_step
        self.tfr_sde = args.tfr_sde

        self.train_vi_len = args.train_vi_len
        self.val_vi_len = args.val_vi_len
        self.batch_size = args.batch_size

    def forward(self, img, label):
        pass

    def training_stage(self):
        for i in range(self.args.num_epoch):
            train_loader = self.train_dataloader()
            adapt_TeacherForcing = True if random.random() < self.tfr else False

            loss = 0
            for (img, label) in (pbar := tqdm(train_loader, ncols=120)):
                img = img.to(self.args.device)
                label = label.to(self.args.device)
                loss += self.training_one_step(img,
                                               label, adapt_TeacherForcing)

                beta = self.kl_annealing.get_beta()
                if adapt_TeacherForcing:
                    self.tqdm_bar('train [TeacherForcing: ON, {:.1f}], beta: {}'.format(
                        self.tfr, beta), pbar, loss, lr=self.scheduler.get_last_lr()[0])

                else:
                    self.tqdm_bar('train [TeacherForcing: OFF, {:.1f}], beta: {}'.format(
                        self.tfr, beta), pbar, loss, lr=self.scheduler.get_last_lr()[0])

            if self.current_epoch % self.args.per_save == 0:
                write_record(self.args.save_root, "Epoch: {}, batch_size: {}, [TeacherForcing: {}, {:.1f}], [anneal_type: {}, beta: {}, cycle: {}, percent: {}], lr: {}, train_loss: {}".format(
                    i, self.args.batch_size, adapt_TeacherForcing, self.tfr, args.kl_anneal_type, beta, self.args.kl_anneal_cycle, self.args.kl_anneal_percent, self.scheduler.get_last_lr()[0], loss/len(train_loader)), False)
                self.save(os.path.join(self.args.save_root,
                          f"epoch={self.current_epoch}.ckpt"))

            self.eval()
            self.current_epoch += 1
            self.scheduler.step()
            self.teacher_forcing_ratio_update()
            self.kl_annealing.update()

    @torch.no_grad()
    def eval(self):
        val_loader = self.val_dataloader()
        loss = 0
        for (img, label) in (pbar := tqdm(val_loader, ncols=120)):
            img = img.to(self.args.device)
            label = label.to(self.args.device)
            loss += self.val_one_step(img, label)
            self.tqdm_bar('val', pbar, loss,
                          lr=self.scheduler.get_last_lr()[0])
        write_record(self.args.save_root,
                     f"valid_loss: {loss/len(val_loader)}")

    def training_one_step(self, img, label, adapt_TeacherForcing):
        # TODO
        self.optim.zero_grad()
        total_loss = 0.0
        tmp = img[:, 0, :, :, :].clone().unsqueeze(1) .detach()
        tmp.requires_grad = False
        batch_g = tmp.clone()  # +每個的第0幀

        t_minus_1_img = img[:, :-1, :, :, :].clone().detach()
        t_minus_1_img.requires_grad = False

        for frame_num in range(1, label.size(1)):
            # ----------------frame_num: 要產生的照片idx

            # Conduct Posterior prediction in Encoder
            # reconstructed_img, mu, logvar = self.forward(img, label)
            en_frame = self.frame_transformation(
                img[:, frame_num, :, :, :].squeeze())  # frame
            en_pos = self.label_transformation(
                label[:, frame_num, :, :, :].squeeze())  # label
            z, mu, logvar = self.Gaussian_Predictor(
                en_frame, en_pos)

            # print(z.size())  # 2,12,32,64
            # print(t_minus_1_img.size())  # 2,15,3,32,64
            # print(label.size())  # 2,16,3,32,64
            # ==================================== #

            de_frame = None
            de_pos = self.label_transformation(
                label[:, frame_num, :, :, :].squeeze())

            if (adapt_TeacherForcing == True):  # not first的原因是因為一開始有加第一章照片上去batch_g了
                # use ground-truth img
                de_frame = self.frame_transformation(
                    t_minus_1_img[:, frame_num - 1, :, :, :].squeeze())
            else:
                # use generated img
                # It is no problem on X1 because X0 have put into batch_g
                de_frame = self.frame_transformation(
                    batch_g[:, -1, :, :, :].squeeze())

            parm = self.Decoder_Fusion(de_frame, de_pos, z)
            g_out = self.Generator(parm)  # next frame # 2,3,32,64 少1dim
            g_out = g_out.unsqueeze(1)

            tmp = g_out.detach()
            tmp.requires_grad = False
            batch_g = torch.cat([batch_g, tmp], dim=1)

            # 計算重構損失和 KL 散度
            reconstruction_loss = self.mse_criterion(
                g_out, img[:, frame_num, :, :, :].unsqueeze(1))
            kl_divergence = kl_criterion(mu, logvar, img.size(0))

            # 總損失
            loss = reconstruction_loss + self.kl_annealing.get_beta() * kl_divergence

            total_loss += loss
        # 反向傳播和參數更新
        total_loss.backward()
        self.optimizer_step()

        return total_loss.item()

    def val_one_step(self, img, label):
        # TODO
        with torch.no_grad():
            total_loss = 0
            batch_g = img[:, 0, :, :, :].clone().unsqueeze(1)  # +每個的第0幀

            for frame_num in range(1, label.size(1)):
                # Conduct Posterior prediction in Encoder
                z = torch.cuda.FloatTensor(
                    1, self.args.N_dim, self.args.frame_H, self.args.frame_W).normal_()

                de_pos = self.label_transformation(
                    label[:, frame_num, :, :, :].squeeze(1))  # label
                de_frame = self.frame_transformation(
                    batch_g[:, -1, :, :, :].squeeze(1))

                # Use previous generated frame
                parm = self.Decoder_Fusion(de_frame, de_pos, z)
                g_out = self.Generator(parm)  # next frame
                tmp = g_out.clone().unsqueeze(1)
                batch_g = torch.cat([batch_g, tmp], dim=1)

                total_loss += self.mse_criterion(
                    batch_g[:, frame_num, :, :, :], img[:, frame_num, :, :, :]).item()
        return total_loss

    def make_gif(self, images_list, img_name):
        new_list = []
        for img in images_list:
            new_list.append(transforms.ToPILImage()(img))

        new_list[0].save(img_name, format="GIF", append_images=new_list,
                         save_all=True, duration=40, loop=0)

    def train_dataloader(self):
        # TODO
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])

        dataset = Dataset_Dance(root=self.args.DR, transform=transform, mode='train', video_len=self.train_vi_len,
                                partial=args.fast_partial if self.args.fast_train else args.partial)
        if self.current_epoch > self.args.fast_train_epoch:
            self.args.fast_train = False

        train_loader = DataLoader(dataset,
                                  batch_size=self.batch_size,
                                  num_workers=self.args.num_workers,
                                  drop_last=True,
                                  shuffle=False)
        return train_loader

    def val_dataloader(self):
        transform = transforms.Compose([
            transforms.Resize((self.args.frame_H, self.args.frame_W)),
            transforms.ToTensor()
        ])
        dataset = Dataset_Dance(root=self.args.DR, transform=transform,
                                mode='val', video_len=self.val_vi_len, partial=1.0)
        val_loader = DataLoader(dataset,
                                batch_size=1,
                                num_workers=self.args.num_workers,
                                drop_last=True,
                                shuffle=False)
        return val_loader

    def teacher_forcing_ratio_update(self):
        if self.current_epoch > self.args.tfr_sde:
            if self.tfr-self.args.tfr_d_step < self.args.teacher_forcing_bounder:
                self.tfr = self.args.teacher_forcing_bounder
            else:
                self.tfr -= self.args.tfr_d_step

    def tqdm_bar(self, mode, pbar, loss, lr):
        pbar.set_description(
            f"({mode}) Epoch {self.current_epoch}, lr:{lr}", refresh=False)
        pbar.set_postfix(loss=float(loss), refresh=False)
        pbar.refresh()

    def save(self, path):
        torch.save({
            "state_dict": self.state_dict(),
            "optimizer": self.state_dict(),
            "lr": self.scheduler.get_last_lr()[0],
            "tfr":   self.tfr,
            "last_epoch": self.current_epoch
        }, path)
        print(f"save ckpt to {path}")

    def load_checkpoint(self):
        if self.args.ckpt_path != None:
            checkpoint = torch.load(self.args.ckpt_path)
            self.load_state_dict(checkpoint['state_dict'], strict=True)
            self.args.lr = checkpoint['lr']
            self.tfr = checkpoint['tfr']

            self.optim = optim.Adam(self.parameters(), lr=self.args.lr)
            self.scheduler = optim.lr_scheduler.MultiStepLR(
                self.optim, milestones=[2, 4], gamma=0.1)
            self.kl_annealing = kl_annealing(
                self.args, current_epoch=checkpoint['last_epoch'])
            self.current_epoch = checkpoint['last_epoch']

    def optimizer_step(self):
        nn.utils.clip_grad_norm_(self.parameters(), 1.)
        self.optim.step()


def main(args):

    os.makedirs(args.save_root, exist_ok=True)
    model = VAE_Model(args).to(args.device)
    model.load_checkpoint()
    if args.test:
        model.eval()
    else:
        model.training_stage()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('--batch_size',    type=int,    default=2)
    parser.add_argument('--lr',            type=float,
                        default=0.001,     help="initial learning rate")
    parser.add_argument('--device',        type=str,
                        choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--optim',         type=str,
                        choices=["Adam", "AdamW"], default="Adam")
    parser.add_argument('--gpu',           type=int, default=1)
    parser.add_argument('--test',          action='store_true')
    parser.add_argument('--store_visualization',      action='store_true',
                        help="If you want to see the result while training")
    parser.add_argument('--DR',            type=str,
                        required=True,  help="Your Dataset Path")
    parser.add_argument('--save_root',     type=str,
                        required=True,  help="The path to save your data")
    parser.add_argument('--num_workers',   type=int, default=4)
    parser.add_argument('--num_epoch',     type=int,
                        default=90,     help="number of total epoch")
    parser.add_argument('--per_save',      type=int, default=1,
                        help="Save checkpoint every seted epoch")
    parser.add_argument('--partial',       type=float, default=1.0,
                        help="Part of the training dataset to be trained")
    parser.add_argument('--train_vi_len',  type=int,
                        default=16,     help="Training video length")
    parser.add_argument('--val_vi_len',    type=int,
                        default=630,    help="valdation video length")
    parser.add_argument('--frame_H',       type=int, default=32,
                        help="Height input image to be resize")
    parser.add_argument('--frame_W',       type=int, default=64,
                        help="Width input image to be resize")

    # Module parameters setting
    parser.add_argument('--F_dim',         type=int, default=128,
                        help="Dimension of feature human frame")
    parser.add_argument('--L_dim',         type=int, default=32,
                        help="Dimension of feature label frame")
    parser.add_argument('--N_dim',         type=int,
                        default=12,     help="Dimension of the Noise")
    parser.add_argument('--D_out_dim',     type=int, default=192,
                        help="Dimension of the output in Decoder_Fusion")

    # Teacher Forcing strategy
    parser.add_argument('--tfr',           type=float,
                        default=1.0,  help="The initial teacher forcing ratio")
    parser.add_argument('--tfr_sde',       type=int,   default=10,
                        help="The epoch that teacher forcing ratio start to decay")
    parser.add_argument('--tfr_d_step',    type=float, default=0.1,
                        help="Decay step that teacher forcing ratio adopted")
    parser.add_argument('--ckpt_path',     type=str,
                        default=None, help="The path of your checkpoints")
    parser.add_argument('--teacher_forcing_bounder', type=float,
                        default=0.0, help="MIN teacher_forcing probability")

    # Training Strategy
    parser.add_argument('--fast_train',         action='store_true')
    parser.add_argument('--fast_partial',       type=float, default=0.4,
                        help="Use part of the training data to fasten the convergence")
    parser.add_argument('--fast_train_epoch',   type=int, default=6,
                        help="Number of epoch to use fast train mode")

    # Kl annealing stratedy arguments
    parser.add_argument('--kl_anneal_type', choices=["Cyclical", "Monotonic", "Without"], type=str,
                        default='Cyclical',       help="")
    parser.add_argument('--kl_anneal_cycle',    type=int,
                        default=10,               help="Number of epochs in a single annealing cycle.")
    parser.add_argument('--kl_anneal_percent',
                        type=float, default=0.5)  # myself
    parser.add_argument('--kl_anneal_ratio',    type=float,
                        default=1,              help="Initial annealing ratio for KL divergence loss.")

    args = parser.parse_args()

    main(args)
