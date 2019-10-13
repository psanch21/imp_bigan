import torch
import torch.nn as nn


class Inference(nn.Module):
    def __init__(self, out_size, n_channels, z_dim):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # in_dim, out_dim, kernel_size, stride, padding
        self.conv1 = nn.Conv2d(n_channels, 64, 3, stride=1, padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 64, 4, stride=2, padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 128, 3, stride=1, padding=(1, 1))
        self.conv4 = nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1))
        if out_size <= 32:
            self.conv5 = nn.Conv2d(128, 256, 3, stride=1, padding=(1, 1))
        else:
            self.conv5 = nn.Conv2d(128, 256, 3, stride=2, padding=(1, 1))
        self.conv6 = nn.Conv2d(256, 256, 4, stride=2, padding=(1, 1))
        self.conv7 = nn.Conv2d(256, 512, 3, stride=1, padding=(1, 1))
        if out_size == 28:
            self.fc = nn.Linear(3 * 3 * 512, z_dim)
        else:
            self.fc = nn.Linear(4 * 4 * 512, z_dim)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, std=0.02)
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
        return y

    def feature_extraction(self, x):
        leak = 0.1
        m = nn.LeakyReLU(leak)(self.conv1(x))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        m = nn.LeakyReLU(leak)(self.conv6(m))
        return x.view(m.size(0), -1)


class InferenceDeep(nn.Module):
    def __init__(self, out_size, n_channels, z_dim):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # in_dim, out_dim, kernel_size, stride, padding
        self.conv1 = nn.Conv2d(n_channels, 64, 3, stride=1, padding=(1, 1))
        self.conv2 = nn.Conv2d(64, 64, 3, stride=1, padding=(1, 1))
        self.conv3 = nn.Conv2d(64, 64, 4, stride=2, padding=(1, 1))
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=(1, 1))
        self.conv5 = nn.Conv2d(64, 128, 3, stride=1, padding=(1, 1))
        self.conv6 = nn.Conv2d(128, 128, 4, stride=2, padding=(1, 1))
        if out_size <= 32:
            self.conv7 = nn.Conv2d(128, 256, 3, stride=1, padding=(1, 1))
        else:
            self.conv7 = nn.Conv2d(128, 256, 3, stride=2, padding=(1, 1))
        self.conv8 = nn.Conv2d(256, 512, 4, stride=2, padding=(1, 1))
        if out_size == 28:
            self.fc = nn.Linear(3 * 3 * 512, z_dim)
        else:
            self.fc = nn.Linear(4 * 4 * 512, z_dim)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, std=0.02)
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

        m = nn.LeakyReLU(leak)(self.conv8(m))

        return m

    def forward(self, x):
        y = self.conv_net(x)
        y = self.fc(y.view(y.size(0), -1))
        return y

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


# %% Connection Netwiork
class ConnectionNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU())

        self.fc3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU())

        self.fc4 = nn.Linear(512, output_dim)

        # self.fc4.weight.data.normal_(0.0, std=1)
        # self.fc4.bias.data.normal_(0.0, std=1)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, std=0.02)
                m.bias.data.zero_()

    def forward(self, x):
        y = self.fc1(x)
        y = self.fc2(y)
        y = self.fc3(y)
        y = self.fc4(y)
        return y


class ConnectionNetGauss(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.fc1 = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU())

        self.fc2 = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU())

        self.fc3 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU())

        self.fc_mean = nn.Linear(512, output_dim)

        # Variance
        self.fc_var = nn.Sequential(
            nn.Linear(512, output_dim),
            nn.Sigmoid())

        self.tanh = nn.Tanh()
        # self.fc4.weight.data.normal_(0.0, std=1)
        # self.fc4.bias.data.normal_(0.0, std=1)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0.0, std=0.02)
                m.bias.data.zero_()

    def forward(self, x):
        y = self.fc1(x)
        y = self.fc2(y)
        y = self.fc3(y)
        mean = self.tanh(self.fc_mean(y))
        var = self.fc_var(y) + 0.001
        return mean, var
