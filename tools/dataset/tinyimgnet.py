import numpy as np
import torch
import torch.utils.data as Data
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm

class simple_dataset(Data.Dataset):
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __getitem__(self, index: int):
        X = Image.fromarray(self.X[index])
        if self.transform is not None:
            X = self.transform(X)
        Y = self.Y[index]
        return X, Y

    def __len__(self):
        return self.X.shape[0]


def tinyimagenet_dataset(data_root='./data'):
    data = np.load(f'{data_root}/tinyimagnet.npz')
    trainX = data['trainX']
    trainY = data['trainY']
    valX = data['valX']
    valY = data['valY']

    print(trainX.shape)
    # save memory from uint8 vs float32, do it on the fly
    # trainX = trainX.float().div_(255.)
    # valX = valX.float().div_(255.)

    mean_all = std_all = 0.0
    for img in tqdm(trainX):
        img = transforms.ToTensor()(img).unsqueeze_(0)
        mean_all += torch.mean(img, dim=(2, 3))
        std_all += torch.std(img, dim=(2, 3))

    print(f"Mean of ddpm data: {mean_all/len(trainX)}")
    print(f"Std of ddpm data: {std_all/len(trainX)}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(64, padding=8),
        transforms.ToTensor()
    ])

    trainset = simple_dataset(trainX, trainY, transform_train)
    testset = simple_dataset(valX, valY, transforms.ToTensor())
    return trainset, testset


def DDPM_dataset(data_root='./data', num_classes=100):
    crop_size, padding = 32, 4

    # data = np.load(f'{data_root}/5m.npz')
    data = np.load(f'{data_root}/cifar1005m.npz')
    trainX = data['image'] #images
    trainY = data['label'] #labels
    if num_classes == 200:
        crop_size, padding = 64, 4

    print(trainX.shape)
    # save memory from uint8 vs float32, do it on the fly
    # trainX = trainX/255.

    mean_all = std_all = 0.0
    for img in tqdm(trainX):
        img = transforms.ToTensor()(img).unsqueeze_(0)
        mean_all += torch.mean(img, dim=(2, 3))
        std_all += torch.std(img, dim=(2, 3))

    print(f"Mean of ddpm data: {mean_all/len(trainX)}")
    print(f"Std of ddpm data: {std_all/len(trainX)}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(crop_size, padding=padding),
        transforms.ToTensor()
    ])

    trainset = simple_dataset(trainX, trainY, transform_train)
    return trainset
