import copy

import numpy
import numpy as np
import urllib.request
import os
import tarfile
import pickle
import torch
from PIL import Image
from sklearn.datasets import fetch_openml
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
from torch.utils.data import DataLoader, random_split
import torch.utils.data as utils
def to_dataloader(my_x, my_y, batchsize):
    my_dataset = utils.TensorDataset(torch.from_numpy(my_x), torch.from_numpy(
        my_y))  # create your datset
    my_dataloader = utils.DataLoader(
        my_dataset, batch_size=batchsize, pin_memory=True)  # create your dataloader
    return my_dataloader

def get_adni():
    # 1. 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Resize an image
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.2817, 0.2817, 0.2817],
            std=[0.3277, 0.3277, 0.3277]
        )
    ])

    # 2. 加载数据集
    data_dir = r'D:\python_new\Robust-PU-master\Robust-PU-master\figures\archive (4)\Alzheimer_MRI_4_classes_dataset'  # 替换为你的数据集路径
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # 3. 将数据集分为健康和痴呆两类
    # 定义健康和痴呆的类别
    healthy_class = 'NonDemented'
    dementia_classes = ['MildDemented', 'ModerateDemented', 'VeryMildDemented']

    # 3. 构建二分类数据集包装器
    class BinaryDementiaDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.healthy_class = healthy_class

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            class_name = self.dataset.classes[label]
            binary_label = 0 if class_name == self.healthy_class else 1
            return image, binary_label

    # 4. 包装数据集
    dataset = BinaryDementiaDataset(dataset)

    # 更新 class_to_idx 和 classes
    dataset.class_to_idx = {healthy_class: 0, 'Dementia': 1}
    dataset.classes = [healthy_class, 'Dementia']

    # 4. 划分训练集和测试集（8:2）
    train_ratio = 0.8
    test_ratio = 0.2
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 5. 创建 DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义提取数据的函数
    def extract_data_from_loader(loader):
        x_all = []
        y_all = []
        for x, y in loader:
            x_all.append(x)
            y_all.append(y)
        # 将批次数据拼接起来
        x_all = torch.cat(x_all, dim=0)
        y_all = torch.cat(y_all, dim=0)
        # 转换为 NumPy 数组
        x_all = x_all.numpy()
        y_all = y_all.numpy()
        return x_all, y_all

    xtrain, ytrain = extract_data_from_loader(train_loader)
    xtest, ytest = extract_data_from_loader(test_loader)

    return (xtrain,ytrain),(xtest,ytest)


def get_CASIS():
    # 1. 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224,224)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        transforms.Normalize(  # 归一化
            mean=[0.2817, 0.2817, 0.2817],
            std=[0.3277, 0.3277, 0.3277]
        )
    ])

    # 2. 加载数据集
    data_dir = r'D:\python_new\statistical_modeling\data\save_new'  # 替换为你的数据集路径
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # 3. 将数据集分为健康和痴呆两类
    # 定义健康和痴呆的类别
    healthy_class = 'NonDemented'
    dementia_classes = ['MildDemented', 'ModerateDemented', 'VeryMildDemented']

    # 3. 构建二分类数据集包装器
    class BinaryDementiaDataset(Dataset):
        def __init__(self, dataset):
            self.dataset = dataset
            self.healthy_class = healthy_class

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, idx):
            image, label = self.dataset[idx]
            class_name = self.dataset.classes[label]
            binary_label = 0 if class_name == self.healthy_class else 1
            return image, binary_label

    # 4. 包装数据集
    dataset = BinaryDementiaDataset(dataset)

    # 更新 class_to_idx 和 classes
    dataset.class_to_idx = {healthy_class: 0, 'Dementia': 1}
    dataset.classes = [healthy_class, 'Dementia']

    # 4. 划分训练集和测试集（8:2）
    train_ratio = 0.8
    test_ratio = 0.2
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # 5. 创建 DataLoader
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 定义提取数据的函数
    def extract_data_from_loader(loader):
        x_all = []
        y_all = []
        for x, y in loader:
            x_all.append(x)
            y_all.append(y)
        # 将批次数据拼接起来
        x_all = torch.cat(x_all, dim=0)
        y_all = torch.cat(y_all, dim=0)
        # 转换为 NumPy 数组
        x_all = x_all.numpy()
        y_all = y_all.numpy()
        return x_all, y_all

    xtrain, ytrain = extract_data_from_loader(train_loader)
    xtest, ytest = extract_data_from_loader(test_loader)

    return (xtrain,ytrain),(xtest,ytest)



def binarize_adni_class(_trainY, _testY):
    trainY = np.ones(len(_trainY), dtype=np.int32)
    trainY[(_trainY == 1) ] = -1
    testY = np.ones(len(_testY), dtype=np.int32)
    testY[(_testY == 1)] = -1
    return trainY, testY


def get_mnist():
    mnist = fetch_openml('mnist_784', data_home=".")
    x = mnist.data
    y = mnist.target
    x=np.asarray(x)
    # reshape to (#data, #channel, width, height)

    # x = np.reshape(x, (x.shape[0], 1, 28, 28))
    x = np.reshape(x, (x.shape[0], 1, 28, 28)) / 255.
    x_tr = np.asarray(x[:60000], dtype=np.float32)
    y_tr = np.asarray(y[:60000], dtype=np.int32)
    x_te = np.asarray(x[60000:], dtype=np.float32)
    y_te = np.asarray(y[60000:], dtype=np.int32)
    return (x_tr, y_tr), (x_te, y_te)
def get_f_mnist():
    fmnist = fetch_openml('Fashion-MNIST', version=1, data_home=".")
    x = fmnist.data
    y = fmnist.target
    x=np.asarray(x)
    # reshape to (#data, #channel, width, height)

    # x = np.reshape(x, (x.shape[0], 1, 28, 28))
    x = np.reshape(x, (x.shape[0], 1, 28, 28)) / 255.
    x_tr = np.asarray(x[:60000], dtype=np.float32)
    y_tr = np.asarray(y[:60000], dtype=np.int32)
    x_te = np.asarray(x[60000:], dtype=np.float32)
    y_te = np.asarray(y[60000:], dtype=np.int32)
    return (x_tr, y_tr), (x_te, y_te)
def convert_labels(labels):
    label_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7,'I':8,"W":9,"X":10,"Y":11}
    converted_labels = [label_map[label] for label in labels]
    return numpy.asarray(converted_labels,dtype=numpy.int32)
def split_matrix(matrix):
    n = len(matrix)
    last_column = [row[-1] for row in matrix]  # 提取最后一列为列表

    # 剔除最后一列，形成 n*(n-1) 矩阵
    reduced_matrix = [row[:-1] for row in matrix]

    return  numpy.asarray(reduced_matrix,dtype=numpy.float32),numpy.asarray(last_column)
def get_vaila(batch_size=500, num_labeled=2000, positive_label_list=['A', 'F'], if_normalized=True):
    val_x_size = 1000

    train_data_frame = pd.read_csv(os.getcwd() + '/data/avila-tr.txt', sep=',', header=None)
    test_data_frame = pd.read_csv(os.getcwd() + '/data/avila-ts.txt', sep=',', header=None)

    val_p_size = num_labeled / len(train_data_frame) * val_x_size
    # change DataFrame into numpy.ndarray
    train_ndarray = train_data_frame.values
    test_ndarray = test_data_frame.values
    train_x,train_y=split_matrix(train_ndarray)
    test_x,test_y=split_matrix(test_ndarray)
    train_y=convert_labels(train_y)
    test_y=convert_labels(test_y)
    train_x = np.reshape(train_x, (train_x.shape[0], 1, 1, 10))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, 1, 10))
    return (train_x,train_y),(test_x,test_y)
def get_grid(batch_size=500, num_labeled=1000, positive_label_list=[1], test_proportion=0.4,if_normalized=False):
    val_x_size = 1000
    data_frame = pd.read_csv(os.getcwd() + '/data/stable_.csv', sep=',', header=None)
    data_ndarray = data_frame.values
    if if_normalized:
        min_max_scaler = preprocessing.MinMaxScaler()
        data_ndarray = np.hstack((min_max_scaler.fit_transform(data_ndarray[:,0:-1]),data_ndarray[:,-1].reshape(-1,1)))
    train_ndarray, test_ndarray = train_test_split(data_ndarray, test_size=test_proportion)  # random_state=0 in DAN
    train_x,train_y=split_matrix(train_ndarray)
    test_x,test_y=split_matrix(test_ndarray)#1是稳定，0是不稳定
    train_x = np.reshape(train_x, (train_x.shape[0], 1, 1, 13))
    test_x = np.reshape(test_x, (test_x.shape[0], 1, 1, 13))
    return (train_x,train_y),(test_x,test_y)
from sklearn.datasets import make_moons
def get_dm():
    X, y = make_moons(n_samples=6000, noise=0.1, shuffle=True)
    X = pd.DataFrame(X, columns=['feature1', 'feature2'])
    y = pd.Series(y)
    X=numpy.asarray(X)
    y=numpy.asarray(y)
    y = [-1 if x == 0 else x for x in y]
    return (X, y), (X, y)



def binarize_vaila_class(_trainY, _testY):
    trainY = -np.ones(len(_trainY), dtype=np.int32)
    trainY[_trainY == 0] = 1

    testY = -np.ones(len(_testY), dtype=np.int32)
    testY[_testY == 0] = 1
    return trainY, testY
def binarize_grid_or_adni_class(_trainY, _testY):
    trainY = np.ones(len(_trainY), dtype=np.int32)
    trainY[_trainY == 0] = -1
    testY = np.ones(len(_testY), dtype=np.int32)
    testY[_testY == 0] = -1
    return trainY, testY

def binarize_f_mnist_class(_trainY, _testY):
    trainY = np.ones(len(_trainY), dtype=np.int32)
    trainY[(_trainY == 1) | (_trainY == 3) | (_trainY == 5) | (
        _trainY == 7) | (_trainY == 8) | (_trainY == 9)] = -1
    testY = np.ones(len(_testY), dtype=np.int32)
    testY[(_testY == 1) | (_testY == 3) | (_testY == 5) | (
        _testY == 7) | (_testY == 8) | (_testY == 9)] = -1
    return trainY, testY


def binarize_mnist_class(_trainY, _testY):
    trainY = np.ones(len(_trainY), dtype=np.int32)
    trainY[_trainY % 2 == 0] = -1
    testY = np.ones(len(_testY), dtype=np.int32)
    testY[_testY % 2 == 0] = -1
    return trainY, testY


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict


def conv_data2image(data):
    return np.rollaxis(data.reshape((3, 32, 32)), 0, 3)


def get_cifar10(path="./cifar10"):
    if not os.path.isdir(path):
        os.mkdir(path)
    url = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    file_name = os.path.basename(url)
    full_path = os.path.join(path, file_name)
    # folder = os.path.join(path, "cifar-10-batches-py")
    folder=r'D:\python_new\nnPUSB-master (2)\nnPUSB-master\nnPUSB-master\cifar-10-python\cifar-10-batches-py'
    # if cifar-10-batches-py folder doesn't exists, download from website
    if not os.path.isdir(folder):
        print("download the dataset from {} to {}".format(url, path))
        urllib.request.urlretrieve(url, full_path)
        with tarfile.open(full_path) as f:
            f.extractall(path=path)
        urllib.request.urlcleanup()

    x_tr = np.empty((0, 32*32*3))
    y_tr = np.empty(1)
    for i in range(1, 6):
        fname = os.path.join(folder, "%s%d" % ("data_batch_", i))
        data_dict = unpickle(fname)
        if i == 1:
            x_tr = data_dict['data']
            y_tr = data_dict['labels']
        else:
            x_tr = np.vstack((x_tr, data_dict['data']))
            y_tr = np.hstack((y_tr, data_dict['labels']))

    data_dict = unpickle(os.path.join(folder, 'test_batch'))
    x_te = data_dict['data']
    y_te = np.array(data_dict['labels'])

    bm = unpickle(os.path.join(folder, 'batches.meta'))
    # label_names = bm['label_names']
    # rehape to (#data, #channel, width, height)
    x_tr = np.reshape(x_tr, (np.shape(x_tr)[0], 3, 32, 32)).astype(np.float32)
    x_te = np.reshape(x_te, (np.shape(x_te)[0], 3, 32, 32)).astype(np.float32)
    # normalize
    x_tr /= 255.
    x_te /= 255.
    return (x_tr, y_tr), (x_te, y_te)  # , label_names


def binarize_cifar10_class(_trainY, _testY):
    trainY = np.ones(len(_trainY), dtype=np.int32)
    trainY[(_trainY == 2) | (_trainY == 3) | (_trainY == 4) | (
        _trainY == 5) | (_trainY == 6) | (_trainY == 7)] = -1
    testY = np.ones(len(_testY), dtype=np.int32)
    testY[(_testY == 2) | (_testY == 3) | (_testY == 4) | (
        _testY == 5) | (_testY == 6) | (_testY == 7)] = -1
    return trainY, testY

def save_txt(data, filename):
    with open(f"{filename}.txt", "w") as file:
        for item in data:
            file.write(str(item) + "\n")



def make_dataset(dataset, n_labeled, n_unlabeled,  save_bias_dataset, with_bias, resample_model=""):
    def make_PU_dataset_from_binary_dataset(x, y, labeled=n_labeled, unlabeled=n_unlabeled, bias=True,  resamplemodel=resample_model):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        assert(len(X) == len(Y))
        perm = np.random.permutation(len(Y))
        X, Y = X[perm], Y[perm]
        # number of positive
        n_p = (Y == positive).sum()
        # number of labeled
        n_lp = labeled
        # number of negative
        n_n = (Y == negative).sum()
        # number of unlabeled
        n_u = unlabeled
        if labeled + unlabeled == len(X):
            n_up = n_p - n_lp
        elif unlabeled == len(X):
            n_up = n_p
        else:
            raise ValueError("Only support |P|+|U|=|X| or |U|=|X|.")
        prior = float(n_up) / float(n_u)
        Xlp = X[Y == positive][:n_lp]
        ########
        xlp_=Xlp
        #########
        Xup = np.concatenate((X[Y == positive][n_lp:], Xlp), axis=0)[:n_up]
        Xun = X[Y == negative]
        bias=with_bias
        if bias:
            if resample_model in ['res18','3mlp']:
                from Resample import resample
                Xlp = resample(Xlp, Xup, Xun, resamplemodel)

        X = np.asarray(np.concatenate(
            (Xlp, Xup, Xun), axis=0), dtype=np.float32)
        Y = np.asarray(np.concatenate(
            (np.ones(n_lp), -np.zeros(n_u))), dtype=np.int32)##-np.ones(n_u
        perm = np.random.permutation(len(Y))
        X, Y = X[perm], Y[perm]
        return X, Y, prior

    def make_PN_dataset_from_binary_dataset(x, y):
        labels = np.unique(y)
        positive, negative = labels[1], labels[0]
        X, Y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.int32)
        n_p = (Y == positive).sum()
        n_n = (Y == negative).sum()
        Xp = X[Y == positive][:n_p]
        Xn = X[Y == negative][:n_n]
        X = np.asarray(np.concatenate((Xp, Xn)), dtype=np.float32)
        Y = np.asarray(np.concatenate(
            (np.ones(n_p), -np.zeros(n_n))), dtype=np.int32)
        perm = np.random.permutation(len(Y))
        X, Y = X[perm], Y[perm]
        return X, Y

    (_trainX, _trainY), (_testX, _testY) = dataset
    trainX, trainY, prior = make_PU_dataset_from_binary_dataset(
        _trainX, _trainY,bias=with_bias)
    testX, testY = make_PN_dataset_from_binary_dataset(_testX, _testY)
    print("training:{}".format(trainX.shape))
    print("test:{}".format(testX.shape))
    print('先验= {:.2f}'.format(prior))

    return [trainX, trainY], [testX, testY], prior, trainX.size // len(trainX)


def load_dataset(dataset_name, n_labeled, n_unlabeled, batchsize, save_bias_dataset, with_bias=True ,resample_model=""):
    print("==================")
    print("loading data...")
    if dataset_name == "mnist":
        (trainX, trainY), (testX, testY) = get_mnist()
        trainY, testY = binarize_mnist_class(trainY, testY)
    elif dataset_name == "cifar10":
        (trainX, trainY), (testX, testY) = get_cifar10()
        trainY, testY = binarize_cifar10_class(trainY, testY)
    elif dataset_name=='fmnist':
        (trainX, trainY), (testX, testY) = get_f_mnist()
        trainY, testY = binarize_f_mnist_class(trainY, testY)
    elif dataset_name=='vaila':
        (trainX, trainY), (testX, testY) = get_vaila()
        trainY, testY = binarize_vaila_class(trainY, testY)
    elif dataset_name=='grid':
        (trainX, trainY), (testX, testY) = get_grid()
        trainY, testY = binarize_grid_or_adni_class(trainY, testY)
    elif dataset_name=='dm':
        (trainX, trainY), (testX, testY) = get_dm()##processed

    elif dataset_name=='adni':
        (trainX, trainY), (testX, testY) = get_adni()
    elif dataset_name == 'casis':
        (trainX, trainY), (testX, testY) = get_CASIS()


        # trainY, testY = binarize_grid_or_adni_class(trainY, testY)
    else:
        raise ValueError("dataset name {} is unknown.".format(dataset_name))

    XYtrain, XYtest, prior, dim = make_dataset(
        ((trainX, trainY), (testX, testY)), n_labeled, n_unlabeled, save_bias_dataset, with_bias, resample_model)


    # Divide testX, testY into validation sets and test sets
    num_test = len(XYtest[0])
    half = num_test // 2
    validX, validY = XYtest[0][:half], XYtest[1][:half]
    testX, testY = XYtest[0][half:], XYtest[1][half:]

    XYtrainLoader = to_dataloader(XYtrain[0], XYtrain[1], batchsize)
    XYtestLoader = to_dataloader(validX, validY, batchsize)
    XYvalidLoader = to_dataloader(testX, testY, batchsize)

    print("load data success!")
    print("==================")

    return XYtrainLoader, XYvalidLoader, XYtestLoader, prior, dim

def load_dataset_no_loder(dataset_name, n_labeled, n_unlabeled, batchsize, save_bias_dataset, with_bias=True ,resample_model=""):
    print("==================")
    print("loading data...")
    if dataset_name == "mnist":
        (trainX, trainY), (testX, testY) = get_mnist()
        trainY, testY = binarize_mnist_class(trainY, testY)
    elif dataset_name == "cifar10":
        (trainX, trainY), (testX, testY) = get_cifar10()
        trainY, testY = binarize_cifar10_class(trainY, testY)
    elif dataset_name=='fmnist':
        (trainX, trainY), (testX, testY) = get_f_mnist()
        trainY, testY = binarize_f_mnist_class(trainY, testY)
    elif dataset_name=='vaila':
        (trainX, trainY), (testX, testY) = get_vaila()
        trainY, testY = binarize_vaila_class(trainY, testY)
    elif dataset_name=='grid':
        (trainX, trainY), (testX, testY) = get_grid()
        trainY, testY = binarize_grid_or_adni_class(trainY, testY)
    elif dataset_name=='dm':
        (trainX, trainY), (testX, testY) = get_dm()##processed
    elif dataset_name=='adni':
        (trainX, trainY), (testX, testY) = get_adni()
    elif dataset_name=='casis':
        (trainX, trainY), (testX, testY) = get_CASIS()
        print(trainY.shape,testY.shape,'chn')


        # trainY, testY = binarize_grid_or_adni_class(trainY, testY)
    else:
        raise ValueError("dataset name {} is unknown.".format(dataset_name))

    XYtrain, XYtest, prior, dim = make_dataset(
        ((trainX, trainY), (testX, testY)), n_labeled, n_unlabeled, save_bias_dataset, with_bias, resample_model)
    return XYtrain, XYtest, prior, dim


if __name__ == '__main__':
    XYtrainLoader, XYvalidLoader, XYtestLoader, prior, dim = load_dataset(
        'casis', 1000, 5352, 64,save_bias_dataset='./Pos_data/adni', with_bias=True, resample_model='res18')



    for i,j in XYvalidLoader:
        print(j.shape)











