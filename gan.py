import torch, torchvision
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import numpy as np

letant_dim = 96
image_size = [1, 28, 28]
batchsz = 64
lr = 0.0003
device = torch.device('cuda')

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(letant_dim, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),

            nn.Linear(1024, np.prod(image_size, dtype=np.int32)),
            nn.Sigmoid()
        )
    def forward(self, x):
        # x.shape = [batchsize, letant]
        out = self.model(x)
        # out.shape = [batchsize, 28*28]
        out = out.reshape(x.shape[0], *image_size)
        # out.shape = [batchsize, 1, 28, 28]

        return out
#return x  x.shape = [batchsize, 1, 28, 28]


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(np.prod(image_size, dtype=np.int32), 512),
            nn.GELU(),

            nn.Linear(512, 256),
            nn.GELU(),

            nn.Linear(256, 128),
            nn.GELU(),

            nn.Linear(128, 64),
            nn.GELU(),

            nn.Linear(64, 32),
            nn.GELU(),

            # nn.Linear(32, 16),
            # nn.GELU(),

            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, image):
        # image.shape = [batchsize, 1, 28, 28]
        pred = self.model(image.reshape(image.shape[0], -1))

        return pred
# return pred pred.shape = [0]


dataset = datasets.MNIST('mnist_data', train=True, transform=transforms.Compose([
    transforms.Resize(28),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.5],std=[0.5])
]),download=True)
dataloader = DataLoader(dataset, batch_size=batchsz, shuffle=True, drop_last=True)

generator = Generator().to(device)
discriminator = Discriminator().to(device)

g_optimizer = optim.Adam(generator.parameters(),lr=lr, betas=(0.4, 0.8), weight_decay=0.0001)
d_optimizer = optim.Adam(discriminator.parameters(),lr=lr, betas=(0.4, 0.8), weight_decay=0.0001)

loss_fn = nn.BCELoss().to(device)

num_epoch = 200
for epoch in range(num_epoch):
    for idx, mini_batch in enumerate(dataloader):
        true_image, _ = mini_batch
        true_image = true_image.to(device)

        x = torch.randn(batchsz, letant_dim).to(device)
        pred_image = generator(x)
        pred_rate = discriminator(pred_image)
        true_rate = discriminator(true_image)

        recons_loss = torch.abs(pred_image - true_image).mean()
        g_loss = loss_fn(pred_rate, torch.ones(batchsz, 1).to(device)) + recons_loss*0.05
        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

        real_loss = loss_fn(true_rate, torch.ones(batchsz, 1).to(device))
        fake_loss = loss_fn(pred_rate.detach(), torch.zeros(batchsz, 1).to(device))
        d_loss = (real_loss + fake_loss)
        d_optimizer.zero_grad()
        d_loss.backward()
        d_optimizer.step()

        if idx % 50 == 0:
            print('step:{} , g_loss:{} , d_loss:{} , real_loss:{} , fake_loss:{}'.format(epoch*len(dataloader)+idx, g_loss, d_loss, real_loss, fake_loss))
        if idx % 400 == 0:
            image = pred_image[:16].data
            torchvision.utils.save_image(image, 'image_{}.png'.format(epoch*len(dataloader)+idx),nrow=4)
