import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim, n_channels=3, image_size=32, m=0.1, trs=True):
        super().__init__()
        # [-1,256, 4, 4]
        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0),
            nn.BatchNorm2d(512, momentum=m, track_running_stats=trs),
            nn.ReLU())
        # [-1,256, 8, 8]
        if image_size == 28:
            self.deconv_2 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 3, 2, 1),
                nn.BatchNorm2d(256, momentum=m, track_running_stats=trs),
                nn.ReLU())
        else:
            self.deconv_2 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256, momentum=m, track_running_stats=trs),
                nn.ReLU())

        self.deconv_3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128, momentum=m, track_running_stats=trs),
            nn.ReLU())

        self.deconv_4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64, momentum=m, track_running_stats=trs),
            nn.ReLU())

        # [-1,256, 16, 16]3
        if image_size == 28:
            self.deconv_5 = nn.ConvTranspose2d(64, n_channels, 3, 1, 1)
        elif image_size == 32:
            self.deconv_5 = nn.ConvTranspose2d(64, n_channels, 3, 1, 1)
        else:
            self.deconv_5 = nn.ConvTranspose2d(64, n_channels, 4, 2, 1)
        # [-1,256, 32, 32]

        self.tanh = nn.Tanh()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, std=0.02)
                m.bias.data.zero_()

    def deconv_net(self, x):
        x = x.view(x.size(0), -1, 1, 1)

        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.deconv_4(x)
        x = self.deconv_5(x)
        x = self.tanh(x)
        return x

    def forward(self, x):
        y = self.deconv_net(x)
        return y


class GeneratorDeep(nn.Module):
    def __init__(self, in_dim, n_channels=3, image_size=32, m=0.1, trs=True):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # [-1,256, 4, 4]
        self.deconv_1 = nn.Sequential(
            nn.ConvTranspose2d(in_dim, 256, 4, 1, 0),
            nn.BatchNorm2d(256, momentum=m, track_running_stats=trs),
            nn.LeakyReLU())
        self.deconv_2 = nn.Sequential(
            nn.ConvTranspose2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512, momentum=m, track_running_stats=trs),
            nn.LeakyReLU())
        # [-1,256, 8, 8]
        if image_size == 28:
            self.deconv_3 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 3, 2, 1),
                nn.BatchNorm2d(256, momentum=m, track_running_stats=trs),
                nn.LeakyReLU())
        else:
            self.deconv_3 = nn.Sequential(
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256, momentum=m, track_running_stats=trs),
                nn.LeakyReLU())

        self.deconv_4 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256, momentum=m, track_running_stats=trs),
            nn.LeakyReLU())

        self.deconv_5 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128, momentum=m, track_running_stats=trs),
            nn.LeakyReLU())

        self.deconv_6 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64, momentum=m, track_running_stats=trs),
            nn.LeakyReLU())

        self.deconv_7 = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64, momentum=m, track_running_stats=trs),
            nn.LeakyReLU())

        # [-1,256, 16, 16]3
        if image_size == 28:
            self.deconv_8 = nn.ConvTranspose2d(64, n_channels, 3, 1, 1)
        elif image_size == 32:
            self.deconv_8 = nn.ConvTranspose2d(64, n_channels, 3, 1, 1)
        else:
            self.deconv_8 = nn.ConvTranspose2d(64, n_channels, 4, 2, 1)
        # [-1,256, 32, 32]

        self.tanh = nn.Tanh()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, std=0.02)
                m.bias.data.zero_()

    def deconv_net(self, x):
        x = x.view(x.size(0), -1, 1, 1)

        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        x = self.deconv_4(x)
        x = self.deconv_5(x)
        x = self.deconv_6(x)
        x = self.deconv_7(x)
        x = self.deconv_8(x)
        x = self.tanh(x)
        return x

    def forward(self, x):
        y = self.deconv_net(x)
        return y
