import numpy as np


# %% Scalers
class StandardScaler():
    def __init__(self):
        self.mean = list()
        self.std = list()

    def fit(self, data):
        self.mean = list()
        self.std = list()
        for i in range(data.shape[-1]):
            self.mean.append(np.mean(data[:, :, i]))
            self.std.append(np.std(data[:, :, i]))
        return

    def fit_transform(self, data):
        self.mean = list()
        self.std = list()
        for i in range(data.shape[-1]):
            self.mean.append(np.mean(data[:, :, i]))
            self.std.append(np.std(data[:, :, i]))
            data[:, :, i] = (data[:, :, i] - self.mean[i]) / self.std[i]

        return data

    def transform(self, data):
        for i in range(data.shape[-1]):
            data[:, :, i] = (data[:, :, i] - self.mean[i]) / self.std[i]
        return data

    def inverse_transform(self, data):
        for i in range(data.shape[-1]):
            data[:, :, i] = data[:, :, i] * self.std[i] + self.mean[i]
        return data


# %% MinMax Scaler
class MinMaxScaler():
    def __init__(self, feature_range=(0, 1)):
        self.min = 0
        self.max = 255

        self.min_f, self.max_f = feature_range

    def fit(self, data):
        self.min = np.min(data)
        self.max = np.max(data)

        self.scale = (self.max_f - self.min_f) / (self.max - self.min)

        return

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

    def transform(self, data):
        x_std = (data - self.min) / (self.max - self.min)
        return x_std * (self.max_f - self.min_f) + self.min_f

    def inverse_transform(self, data):
        x_std = (data - self.min_f) / (self.max_f - self.min_f)

        return x_std * (self.max - self.min) + self.min
