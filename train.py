import os
import traceback

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.autograd import Variable

from torchvision import transforms, datasets
from torchvision import utils
from torchvision.utils import save_image
import torchvision.utils as vutils

from net import Generator, Discriminator

image_size = 64
device = 'cuda'
nz = 100 # z vector size
ngf = 64
ndf = 64
nc = 3 # Channel

def create_transform():
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    return  transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])


def set_dataloader(dataset_path, transform, batch_size):
    dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                       shuffle=True, num_workers=4)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def training_with_real(D, criterion, label, real_image):
    # 勾配の更新
    D.zero_grad()
    real_image_output = D(real_image)

    error_real_image = criterion(real_image_output,
                                 label)
    error_real_image.backward()
    D_x = real_image_output.mean().item()

    return error_real_image, D_x

def training_with_fake(G, D, criterion, batch_size, label):
    fake_label = 0
    noise = torch.randn(batch_size, nz, 1, 1, device=device)
    fake = G(noise)
    label.fill_(fake_label)
    fake_image = D(fake.detach())
    error_discriminator = criterion(fake_image, label)
    error_discriminator.backward()
    D_G_z1 = fake_image.mean().item()

    return error_discriminator, fake, D_G_z1

def update_network(D, G, criterion, fake, real_label, label):
    # Update Network
    G.zero_grad()
    label.fill_(real_label) 
    output = D(fake) # 鑑定を行う
    errorG = criterion(output, label)
    errorG.backward()
    D_G_z2 = output.mean().item()

    return errorG, D_G_z2


def run_train(netD, netG, dataloader, options):
    netG.train()
    netG.apply(weights_init)
    netD.train()
    netD.apply(weights_init)

    criterion = nn.BCELoss()

    print(netG)
    print(netD)
    
    fixed_noise = torch.randn(options["batch_size"], nz, 1, 1, device=device)

    # Setup Optimizer
    optimizerD = optim.Adam(netD.parameters(),
                             lr=options["lr"],
                             betas=(0.5, 0.999))

    optimizerG = optim.Adam(netG.parameters(),
                             lr=options["lr"],
                             betas=(0.5, 0.999))

    real_label = 1
    errorG = 0

    for epoch in range(options['epoch']):
        print(f'{epoch + 1}')
        for i, data in enumerate(dataloader):
            raw_image, raw_label = data
            real_image = raw_image.to(device)
            batch_size = real_image.size(0)
            label = torch.full((batch_size,), real_label, device=device)

            # Train with Real
            error_real_image, D_x = training_with_real(netD, criterion,
                                                       label, real_image)

            # # train with fake
            error_discriminator, fake, D_G_z1 = training_with_fake(netG, netD, criterion, 
                                                                   batch_size, label)
            optimizerD.step()

            error_discriminator = error_real_image + error_discriminator

            errorG, D_G_z2 = update_network(netD, netG, criterion,
                                            fake, real_label, label)
            optimizerG.step()

            print(f'[{epoch}/{options["epoch"]}][{i}/{len(dataloader)}] \
                     Loss_D: {error_discriminator.item()} Loss_G: {errorG.item()} \
                     D(x): {D_x} D(G(z)): {D_G_z1} / {D_G_z2}')

            print("Save")
            vutils.save_image(real_image,
                              f'result_image/real_{epoch}_samples.png',
                              normalize=False)
            
            fake_image = netG(fixed_noise)
            vutils.save_image(fake_image.detach(),
                                f'result_image/fake_{epoch}_samples.png',
                                normalize=True)
            print("END Save")

            torch.save(netD.state_dict(), f'result_pth/real_{epoch}.pth')
            torch.save(netG.state_dict(), f'result_pth/fake_{epoch}.pth')


def main():
    train_dataset_path = 'D:\project\dcgan2\dataset'
    # train_dataset_path = 'D:\project\dataset\\food'
    options = {
        'batch_size': 128,
        'epoch': 1500,
        'lr': 1e-5
    }

    data_transform = create_transform()
    data_loader = set_dataloader(train_dataset_path, data_transform,
                                options['batch_size'])

    G = Generator().to('cuda')
    D = Discriminator().to('cuda')

    run_train(D, G, data_loader, options)

main() if __name__ == '__main__' else None