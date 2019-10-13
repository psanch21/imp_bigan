import collections
import os

import numpy as np
import torch
import torch.functional as F
import torch.nn as nn
import torchvision
from torchvision import transforms

import utils.aux as lib

# %%
NUM_CLASS = 10
IMAGE_SIZE = 28
CHANNEL = 1

CLASS_CLOTHING = {0: 'T-shirt/top',
                  1: 'Trouser',
                  2: 'Pullover',
                  3: 'Dress',
                  4: 'Coat',
                  5: 'Sandal',
                  6: 'Shirt',
                  7: 'Sneaker',
                  8: 'Bag',
                  9: 'Ankle boot'}


# %%
class Classifier(nn.Module):
    def __init__(self, num_of_class):
        super(Classifier, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7 * 7 * 32, num_of_class)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out

    def get_proportions(self, x):
        index = self.get_prediction(x, hard=True)
        out = collections.Counter(index)
        out = dict(collections.OrderedDict(sorted(dict(out).items())))
        total = sum(out.values(), 0.0)
        out = {k: v / total for k, v in out.items()}

        return out

    def get_prediction(self, x_input, hard=True):
        n_imgs = x_input.shape[0]
        out_list = list()
        n_batches = int(np.ceil(n_imgs / 128))
        for i in range(n_batches):
            x = torch.tensor(x_input[i * 128:(i + 1) * 128]).float()
            x = lib.cuda(x)
            out = self.forward(x)
            out = torch.argmax(out, dim=1) if hard else F.softmax(out, dim=1)
            out_list.extend(out.data.cpu().numpy())

        return np.array(out_list)

    def extract_features(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        return out

    def get_feature_dim(self):
        return 7 * 7 * 32


class TrainClf:
    def __init__(self, gpu=-1, data_root='../Data/FMNIST', save_dir='./classifiers'):
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

        print('Device detected: {}'.format(self.device))
        self.model = Classifier(num_of_class=10).to(self.device)
        self.data_root = data_root

        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

        self.save_dir = lib.create_dir(save_dir)
        self.ckpt_file = os.path.join(self.save_dir, 'fmnist_clf.pth')

    def train(self, lr=0.005, n_epochs=10, batch_size=64):
        if os.path.exists(self.ckpt_file):
            model_dict = torch.load(self.ckpt_file, map_location=self.device) if self.device == 'cpu' else torch.load(
                self.ckpt_file)
            self.model.load_state_dict(model_dict['model'])
            print('Trained FMNIST classifier found! Good!')
            return self.model
        train_data = torchvision.datasets.FashionMNIST(self.data_root, train=True, transform=self.transform,
                                                       download=True)

        train_size = int(0.8 * len(train_data))
        test_size = len(train_data) - train_size
        train_data, valid_data = torch.utils.data.random_split(train_data, [train_size, test_size])

        train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
        valid_dataloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        accuracy_valid = 0.
        patient = 0
        for epoch in range(1, n_epochs + 1):
            for batch_id, (image, label) in enumerate(train_dataloader):
                label, image = label.to(self.device), image.to(self.device)
                output = self.model(image)
                loss = criterion(output, label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if batch_id % 1000 == 0:
                    print('Loss :{:.4f} Epoch[{}/{}]'.format(loss.item(), epoch, n_epochs))

            self.model.eval()
            with torch.no_grad():
                correct = 0
                total = 0
                for image, label in valid_dataloader:
                    image = image.to(self.device)
                    label = label.to(self.device)
                    outputs = self.model(image)
                    predicted = torch.argmax(outputs, dim=1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                print('Accuracy of the model on the validation images: {} %'.format(100 * correct / total))

                acc = correct / total
                if acc >= accuracy_valid:
                    accuracy_valid = acc
                    patient = 0
                elif patient == 2:
                    break
                else:
                    patient += 1

            self.model.train()

        torch.save({'model': self.model.state_dict()}, self.ckpt_file)
        print('Saved model to ' + self.ckpt_file)
        return self.model

    def test(self, batch_size=64):
        test_data = torchvision.datasets.FashionMNIST(self.data_root, train=False, transform=self.transform,
                                                      download=True)

        test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True)

        with torch.no_grad():
            correct = 0
            total = 0
            for label, image in test_dataloader:
                image = image.to(self.device)
                label = label.to(self.device)
                outputs = self.model(image)
                predicted = torch.argmax(outputs, dim=1)
                total += label.size(0)
                correct += (predicted == label).sum().item()
            print('Test Accuracy of the model on the test images: {} %'.format(100 * correct / total))
