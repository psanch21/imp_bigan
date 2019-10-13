import torch
import torch.nn as nn

from utils.spectral_norm import SpectralNorm


class Discriminator(nn.Module):
    def __init__(self, n_channels, image_size=32):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        stride_5 = 1 if image_size == 32 else 2
        # in_dim, out_dim, kernel_size, stride, padding
        self.conv1 = SpectralNorm(nn.Conv2d(n_channels, 64, 3, stride=1, padding=(1, 1)))
        self.conv2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1, 1)))
        self.conv3 = SpectralNorm(nn.Conv2d(64, 128, 3, stride=1, padding=(1, 1)))
        self.conv4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1)))
        self.conv5 = SpectralNorm(nn.Conv2d(128, 256, 3, stride=stride_5, padding=(1, 1)))
        self.conv6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=(1, 1)))
        self.conv7 = SpectralNorm(nn.Conv2d(256, 512, 3, stride=1, padding=(1, 1)))

        self.fc = SpectralNorm(nn.Linear(4 * 4 * 512, 1))

    def reset_parameters(self):
        for i in self._modules:
            m = self._modules[i]
            if isinstance(m, nn.Conv2d):
                self.m.weight.data.normal_(0.0, std=0.02)
                m.bias.data.zero_()

    def conv_net(self, x):
        leak = 0.1
        m = nn.LeakyReLU(leak)(self.conv1(x))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))
        m = nn.LeakyReLU(leak)(self.conv7(m))
        return m

    def forward(self, x):
        y = self.conv_net(x)
        y = self.fc(y.view(y.size(0), -1))
        return y.view(-1)

    def feature_extraction(self, x):
        leak = 0.1
        m = nn.LeakyReLU(leak)(self.conv1(x))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))
        m = m.view(m.size(0), -1)
        return m


# %% Joint Discriminator (X, Z)

class DiscriminatorJoin(nn.Module):
    def __init__(self, n_channels, image_size, z_dim):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # in_dim, out_dim, kernel_size, stride, padding
        self.convx_1 = SpectralNorm(nn.Conv2d(n_channels, 64, 3, stride=1, padding=(1, 1)))
        self.convx_2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1, 1)))
        if (image_size <= 32):
            self.convx_3 = SpectralNorm(nn.Conv2d(64, 128, 4, stride=1, padding=(1, 1)))
        else:
            self.convx_3 = SpectralNorm(nn.Conv2d(64, 128, 4, stride=2, padding=(1, 1)))
        self.convx_4 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1)))
        self.convx_5 = SpectralNorm(nn.Conv2d(128, 256, 4, stride=1, padding=0))
        if image_size == 28:
            self.convx_6 = SpectralNorm(nn.Conv2d(256, 256, 3, stride=2, padding=0))
        else:
            self.convx_6 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=0))

        self.convz_5 = SpectralNorm(nn.Conv2d(z_dim, 256, 1, stride=1, padding=0))
        self.conv6 = SpectralNorm(nn.Conv2d(512, 512, 1, stride=1, padding=0))
        self.conv7 = SpectralNorm(nn.Conv2d(512, 1024, 1, stride=1, padding=0))

        self.fc = SpectralNorm(nn.Linear(1024, 1))

    def reset_parameters(self):
        for i in self._modules:
            m = self._modules[i]
            if isinstance(m, nn.Conv2d):
                self.m.weight.data.normal_(0.0, std=0.02)
                m.bias.data.zero_()

    def conv_net_x(self, x):
        leak = 0.1
        m = nn.LeakyReLU(leak)(self.convx_1(x))
        m = nn.LeakyReLU(leak)(self.convx_2(m))
        m = nn.LeakyReLU(leak)(self.convx_3(m))
        m = nn.LeakyReLU(leak)(self.convx_4(m))
        m = nn.LeakyReLU(leak)(self.convx_5(m))
        m = nn.LeakyReLU(leak)(self.convx_6(m))
        return m

    def conv_net_xz(self, x):
        leak = 0.1
        m = nn.LeakyReLU(leak)(self.conv6(x))
        m = nn.LeakyReLU(leak)(self.conv7(m))
        return m

    def forward(self, x, z):
        x_h = self.conv_net_x(x)

        z = z.view(z.size(0), -1, 1, 1)  # view(shape)
        z_h = self.convz_5(z)

        h = torch.cat((x_h, z_h), 1)
        y = self.conv_net_xz(h)
        y = self.fc(y.view(y.size(0), -1))
        return y.view(-1)

    def feature_extraction(self, x):
        leak = 0.1
        m = nn.LeakyReLU(leak)(self.conv1(x))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))
        return x.view(m.size(0), -1)


class DiscriminatorJoinDeep(nn.Module):
    def __init__(self, n_channels, image_size, z_dim):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # in_dim, out_dim, kernel_size, stride, padding
        self.convx_1 = SpectralNorm(nn.Conv2d(n_channels, 64, 3, stride=1, padding=(1, 1)))
        self.convx_2 = SpectralNorm(nn.Conv2d(64, 64, 4, stride=2, padding=(1, 1)))
        if (image_size <= 32):
            self.convx_3 = SpectralNorm(nn.Conv2d(64, 128, 4, stride=1, padding=(1, 1)))
        else:
            self.convx_3 = SpectralNorm(nn.Conv2d(64, 128, 4, stride=2, padding=(1, 1)))
        self.convx_4 = SpectralNorm(nn.Conv2d(128, 128, 3, stride=1, padding=(1, 1)))
        self.convx_5 = SpectralNorm(nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1)))
        self.convx_6 = SpectralNorm(nn.Conv2d(128, 256, 4, stride=1, padding=0))
        if image_size == 28:
            self.convx_7 = SpectralNorm(nn.Conv2d(256, 256, 3, stride=2, padding=0))
        else:
            self.convx_7 = SpectralNorm(nn.Conv2d(256, 256, 4, stride=2, padding=0))

        self.convx_8 = SpectralNorm(nn.Conv2d(256, 256, 3, stride=1, padding=1))

        self.convz_1 = SpectralNorm(nn.Conv2d(z_dim, 256, 1, stride=1, padding=0))
        self.convz_2 = SpectralNorm(nn.Conv2d(256, 256, 1, stride=1, padding=0))

        self.conv1 = SpectralNorm(nn.Conv2d(512, 512, 1, stride=1, padding=0))
        self.conv2 = SpectralNorm(nn.Conv2d(512, 512, 1, stride=1, padding=0))

        self.fc = SpectralNorm(nn.Linear(512, 1))

    def reset_parameters(self):
        for i in self._modules:
            m = self._modules[i]
            if isinstance(m, nn.Conv2d):
                self.m.weight.data.normal_(0.0, std=0.02)
                m.bias.data.zero_()

    def conv_net_x(self, x):
        leak = 0.1
        m = nn.LeakyReLU(leak)(self.convx_1(x))
        m = nn.LeakyReLU(leak)(self.convx_2(m))
        m = nn.LeakyReLU(leak)(self.convx_3(m))
        m = nn.LeakyReLU(leak)(self.convx_4(m))
        m = nn.LeakyReLU(leak)(self.convx_5(m))
        m = nn.LeakyReLU(leak)(self.convx_6(m))
        m = nn.LeakyReLU(leak)(self.convx_7(m))
        m = nn.LeakyReLU(leak)(self.convx_8(m))
        return m

    def conv_net_xz(self, x):
        leak = 0.1
        m = nn.LeakyReLU(leak)(self.conv1(x))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        return m

    def forward(self, x, z):
        x_h = self.conv_net_x(x)

        z = z.view(z.size(0), -1, 1, 1)  # view(shape)
        z_h = self.convz_1(z)
        z_h = self.convz_2(z_h)

        h = torch.cat((x_h, z_h), 1)
        y = self.conv_net_xz(h)
        y = self.fc(y.view(y.size(0), -1))
        return y.view(-1)

    def feature_extraction(self, x):
        leak = 0.1
        m = nn.LeakyReLU(leak)(self.conv1(x))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))
        m = nn.LeakyReLU(leak)(self.conv7(m))
        m = nn.LeakyReLU(leak)(self.conv8(m))
        return x.view(m.size(0), -1)
