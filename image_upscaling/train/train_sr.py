import torch

from torch import nn
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader

from config import SRGAN_CONFIG as CONFIG
from utils import load_checkpoint, save_checkpoint
from losses.vgg_loss import VGGLoss
from models.srgan import Generator, Discriminator
from dataset import ImageDataset


def pretrain_fn(loader, gen, opt_gen, mse):
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(CONFIG['device'])
        low_res = low_res.to(CONFIG['device'])

        fake = gen(low_res)
        loss = mse(fake, high_res)
        opt_gen.zero_grad()
        loss.backwards()
        opt_gen.step()


def train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss):
    loop = tqdm(loader, leave=True)

    for idx, (low_res, high_res) in enumerate(loop):
        high_res = high_res.to(CONFIG['device'])
        low_res = low_res.to(CONFIG['device'])

        fake = gen(low_res)
        disc_real = disc(high_res)
        disc_fake = disc(fake.detach())
        disc_loss_real = bce(
            disc_real, torch.ones_like(disc_real) - 0.1 * torch.rand_like(disc_real)
        )  # label smoothing
        disc_loss_fake = bce(disc_fake, torch.zeros_like(disc_fake))
        disc_loss = disc_loss_real + disc_loss_fake

        opt_disc.zero_grad()
        disc_loss.backward()
        opt_disc.step()

        disc_fake = disc(fake)
        adversarial_loss = 1e-3 * bce(disc_fake, torch.ones_like(disc_fake))
        vgg_act_loss = 6e-3 * vgg_loss(fake, high_res)
        gen_loss = vgg_act_loss + adversarial_loss

        opt_gen.zero_grad()
        gen_loss.backward()
        opt_gen.step()

def main():
    dataset = ImageDataset(root_dir='flickr2k/')
    loader = DataLoader(
        dataset,
        CONFIG['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=CONFIG['num_workers']
    )

    gen = Generator(in_channels=3).to(CONFIG['device'])
    disc = Discriminator(in_channels=3).to(CONFIG['device'])
    opt_gen = optim.Adam(gen.parameters(), lr=CONFIG['learning_rate'], betas=(0.9, 0.999))
    opt_disc = optim.Adam(gen.parameters(), lr=CONFIG['learning_rate'], betas=(0.9, 0.999))
    mse = nn.MSELoss()
    bce = nn.BCEWithLogitsLoss()
    vgg_loss = VGGLoss()

    if CONFIG['load_model']:
        load_checkpoint(
            CONFIG['checkpoint_gen'],
            gen,
            opt_gen,
            CONFIG['learning_rate']
        )
        load_checkpoint(
            CONFIG['checkpoint_disc'],
            disc,
            opt_disc,
            CONFIG['learning_rate']
        )

    for epoch in range(CONFIG['num_pretrain_epochs']):
        pretrain_fn(loader, gen, opt_gen, mse)

    for epoch in range(CONFIG['num_epochs']):
        train_fn(loader, disc, gen, opt_gen, opt_disc, mse, bce, vgg_loss)

        if CONFIG['save_model']:
            save_checkpoint(gen, opt_gen, filename=CONFIG['checkpoint_gen'])
            save_checkpoint(disc, opt_disc, filename=CONFIG['checkpoint_disc'])


if __name__ == '__main__':
    main()