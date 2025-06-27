#DCGAN - Deep Convolutional GAN

#D - stride convolutions G-Fractional Stride Convolution
import torch
import torch.nn as nn
import  torchvision
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.nn import BatchNorm2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter




###Discriminator###

class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            #Input : N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ), # 32 x 32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d*2, 4, 2,1), # 16x 16
            self._block(features_d*2, features_d * 4, 4, 2, 1), #  8 x 8
            self._block(features_d*4, features_d * 8, 4, 2, 1),# 4 x 4
            nn.Conv2d(features_d*8, 1, kernel_size=4, stride=2, padding=0),
            nn.Sigmoid(),

        )
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),

        )
    def forward(self, x):
        return self.disc(x)


###Generator###
class Generator(nn.Module):
    def __init__(self, z_dim, channels_img, features_g):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            # Input : N x z_dim x 1 x 1
            self._block(z_dim, features_g*16, 4, 1, 0), #N x f_g*16 x 4 x 4
            self._block(features_g*16, features_g*8,4, 2, 1),
            self._block(features_g*8, features_g*4, 4, 2, 1),
            self._block(features_g*4, features_g*2, 4, 2, 1),
            nn.ConvTranspose2d(
                features_g*2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            nn.Tanh(),
        )
    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    def forward(self, x):
        return self.gen(x)
def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)



device = "cuda" if torch.cuda.is_available() else "cpu"
lr = 2e-4
z_dim = 100
image_size = 64
batch_size = 128
num_epochs = 5
features_gen = 64
features_disc = 64
channels_img = 1


disc = Discriminator(channels_img, features_disc).to(device)
gen = Generator(z_dim, channels_img, features_gen).to(device)
initialize_weights(gen)
initialize_weights(disc)


transforms = transforms.Compose(
    [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.5 for _ in range(channels_img)], [0.5 for _ in range(channels_img)]
        ),
    ]
)

dataset = datasets.MNIST(root="dataset/", train=True, transform=transforms, download=True)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
opt_disc = optim.Adam(disc.parameters(), lr=lr, betas=(0.5, 0.999))
opt_gen = optim.Adam(gen.parameters(), lr=lr, betas=(0.5,0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32, z_dim, 1, 1).to(device)
writer_fake = SummaryWriter(f"logs/fake")
writer_real = SummaryWriter(f"logs/real")
step = 0

gen.train()
disc.train()

for epoch in range(num_epochs):
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(device)

        ###Train discriminator
        noise = torch.randn(batch_size, z_dim,1,1).to(device)
        fake = gen(noise)
        disc_real = disc(real).reshape(-1)
        lossD_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        lossD = (lossD_fake+lossD_real)/2
        disc.zero_grad()
        lossD.backward(retain_graph=True)
        opt_disc.step()

        ###Train Generator
        output = disc(fake).reshape(-1)
        lossG = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        lossG.backward()
        opt_gen.step()

        if batch_idx % 100 ==0:
            print(
                f"Epoch [{epoch+1}/{num_epochs}] Loss D: {lossD:.4f}, Loss G: {lossG:.4f}"
            )
            with torch.no_grad():
                fake = gen(fixed_noise)
                img_grid_fake = torchvision.utils.make_grid(fake[:32], normalize=True)
                img_grid_real = torchvision.utils.make_grid(real[:32], normalize=True)
                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )

                step+=1