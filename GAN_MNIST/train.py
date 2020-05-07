import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import argparse
import numpy as np

from pathlib import Path
from collections import namedtuple
from torch.autograd import Variable
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

from .model import *
from .loss import *
from .dataloader import *

def train(opt):
    # os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'

    generator = GAN_MNIST_GENERATOR().cuda() if opt.gpu else GAN_MNIST_GENERATOR()
    discriminator = GAN_MNIST_DISCRIMINATOR().cuda() if opt.gpu else GAN_MNIST_DISCRIMINATOR()

    checkpoint = Path().absolute().parent / 'checkpoint_generator.pt'
    if checkpoint.exists():
        print("loading generator checkpoint")
        generator.load_state_dict(torch.load('checkpoint_generator.pt'))
    
    checkpoint = Path().absolute().parent / 'checkpoint_discriminator.pt'
    if checkpoint.exists():
        print("loading discriminator checkpoint")
        generator.load_state_dict(torch.load('checkpoint_discriminator.pt'))
    
    generator.train()
    discriminator.train()

    train_dataset = MNISTDataset('train')
    train_data_loader = MNISTDataloader(train_dataset, opt)

    test_dataset = MNISTDataset('test')
    test_data_loader = MNISTDataloader(test_dataset, opt)

    criterion = Loss()

    optim_gen = torch.optim.Adam(generator.parameters(), lr=0.0001)
    optim_dsc = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

    writer = SummaryWriter()

    print("ready")
    
    for epoch in range(opt.epoch):
        for n, (images, labels) in enumerate(train_data_loader.data_loader):
            step = epoch * len(train_data_loader.data_loader) + n + 1
            # load real image
            real_imgs = images.cuda() if opt.gpu else images
            # 숫자의 label -> CGAN에서 사용
            # real_label = labels.cuda() if opt.gpu else labels
            _label = torch.ones(images.size(0))
            _label = _label.cuda() if opt.gpu else _label
            real_dsc_labels = Variable(_label)

            # create fake image
            latent_vector = torch.randn(images.size(0), 100)
            latent_vector = latent_vector.cuda() if opt.gpu else latent_vector
            fake_imgs = generator(latent_vector)
            # create된 image의 label -> CGAN에서 사용
            # fake_label = Variable(torch.randn(images.size(0)).cuda())
            _label = torch.zeros(images.size(0))
            _label = _label.cuda() if opt.gpu else _label
            fake_dsc_labels = Variable(_label)

            # train discriminator
            discriminator.zero_grad()
            real_result = discriminator(real_imgs)
            loss_real = criterion(real_result, real_dsc_labels)
            fake_result = discriminator(fake_imgs)
            loss_fake = criterion(fake_result, fake_dsc_labels)
            loss_total = loss_real + loss_fake
            loss_total.backward(retain_graph=True)
            optim_dsc.step()

            # train generator
            generator.zero_grad()
            fake_result = discriminator(fake_imgs)
            loss_gen = criterion(fake_result, real_dsc_labels)
            loss_gen.backward(retain_graph=True)
            optim_gen.step()

            if step % opt.display_step == 0:
                print('[Epoch {}]'.format(epoch))

                # discriminator - real image
                total_train = labels.size(0)
                correct_train = (real_result == 1).sum().item()
                total_test = 0
                correct_test = 0
                for _, (images, labels) in enumerate(test_data_loader.data_loader):
                    real_imgs = images.cuda() if opt.gpu else images
                    result = discriminator(real_imgs)
                    total_test += labels.size(0)
                    correct_test += (result == 1).sum().item()
                print('1> discriminator - on real image')
                print('Loss : {:.2}, train_acc : {:.2}, test_acc : {:.2}'.format(loss_real, correct_train/total_train, correct_test/total_test))                

                # generator - fake image
                total_train = labels.size(0)
                correct_train = (fake_result == 1).sum().item()
                print('2> generator - creating fake image')
                print('Loss : {:.2}, acc : {:.2}'.format(loss_gen, correct_train/total_train))

        torch.save(generator.state_dict(), Path().absolute().parent / 'checkpoint_generator.pt')
        torch.save(discriminator.state_dict(), Path().absolute().parent / 'checkpoint_discriminator.pt')

def get_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gpu', type=int, default=1)
    parser.add_argument('-n', '--num-workers', type=int, default=4)
    parser.add_argument('-e', '--epoch', type=int, default=100)
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-t', '--test-batch-size', type=int, default=1)
    parser.add_argument('-d', '--display-step', type=int, default=600)
    opt = parser.parse_args()
    print(opt)
    return opt

def run():
    Opts = namedtuple("Opts", 'gpu num_workers epoch batch_size test_batch_size display_step')
    # debug setup on non GPU
    opt = Opts(gpu=0, num_workers=0, epoch=100, batch_size=128, test_batch_size=1, display_step=100)
    train(opt)

if __name__ == '__main__':
    opt = get_opt()
    train(opt)
