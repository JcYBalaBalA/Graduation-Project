import torchvision.datasets as dset
import torch.utils.data
import os
import torchvision.transforms as transforms

# 目前实现选择data_transform对图片数据进行预处理操作，传入None就代表不进行预处理
# 如果有可能希望做成由用户输入data_transform，将输入代码和dset形成字符串而后使用exec执行
# return train_loader len(train_set) test_loader len(test_set)
def getData(dset_name, batch_size, data_transform):
    dataPath = "../data"
    os.makedirs(dataPath, exist_ok=True)
    if dset_name == "CIFAR10":
        trainset = dset.CIFAR10(dataPath, train=True, download=True, transform=data_transform)
        testset = dset.CIFAR10(dataPath, train=False, download=True, transform=data_transform)
    elif dset_name == "LSUN":
        trainset = dset.LSUN(dataPath, train=True, download=True, transform=data_transform)
        testset = dset.LSUN(dataPath, train=False, download=True, transform=data_transform)
    elif dset_name == "FakeData":
        trainset = dset.FakeData(dataPath, train=True, download=True, transform=data_transform)
        testset = dset.FakeData(dataPath, train=False, download=True, transform=data_transform)
    elif dset_name == "CocoCaptions":
        trainset = dset.CocoCaptions(dataPath, train=True, download=True, transform=data_transform)
        testset = dset.CocoCaptions(dataPath, train=False, download=True, transform=data_transform)
    elif dset_name == "MNIST":
        trainset = dset.mnist(dataPath, train=True, download=True, transform=data_transform)
        testset = dset.mnist(dataPath, train=False, download=True, transform=data_transform)
    elif dset_name == "CIFAR100":
        trainset = dset.CIFAR100(dataPath, train=True, download=True, transform=data_transform)
        testset = dset.CIFAR100(dataPath, train=False, download=True, transform=data_transform)
    elif dset_name == "SVHN":
        trainset = dset.SVHN(dataPath, train=True, download=True, transform=data_transform)
        testset = dset.SVHN(dataPath, train=False, download=True, transform=data_transform)
    elif dset_name == "Flickr8k":
        trainset = dset.Flickr8k(dataPath, train=True, download=True, transform=data_transform)
        testset = dset.Flickr8k(dataPath, train=False, download=True, transform=data_transform)
    elif dset_name == "Cityscapes":
        trainset = dset.Cityscapes(dataPath, train=True, download=True, transform=data_transform)
        testset = dset.Cityscapes(dataPath, train=False, download=True, transform=data_transform)
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True),len(trainset),\
           torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True), len(testset),


def getDataPath(dataPath, batch_size, data_transform):
    dataset = dset.ImageFolder(dataPath, transform=data_transform)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True),len(dataset)

# 单通道
def channel_1_transform(img_size=64):
    data_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    return data_transform

# 三通道
def channel_3_transform(img_size=64):
    data_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    return data_transform