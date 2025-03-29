import numpy
import numpy as np
import torch
import torch.utils.data as utils
from model import ThreeLayerPerceptron, MultiLayerPerceptron, CNN
import torch.nn as nn
from torchvision import models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def zero_one_loss( h, t, is_logistic=False):
    positive = 1
    negative = 0 if is_logistic else -1
    # print(t,'t')
    # print(h,'h')

    n_p = (t == positive).sum()
    n_n = (t == negative).sum()
    size = n_p + n_n

    t_p = ((h == positive) * (t == positive)).sum()
    t_n = ((h == negative) * (t == negative)).sum()
    f_p = n_n - t_n
    f_n = n_p - t_p

    presicion = (0.0 if t_p == 0 else t_p / (t_p + f_p))
    recall = (0.0 if t_p == 0 else t_p / (t_p + f_n))
    acc = (0.0 if t_p == 0 else (t_p + t_n) / (t_p + t_n + f_n + f_p))
    return presicion, recall, 1 - (t_p + t_n) / size, acc

def error(model, DataLoader, is_logistic=False):
    presicion = []
    recall = []
    error_rate = []
    acc_rate = []
    model.eval()
    for data, target in DataLoader:
        data = data.to(device, non_blocking=True)
        t = target.detach().cpu().numpy()
        size = len(t)
        if is_logistic:
            h = np.reshape(torch.sigmoid(
                model(data)).detach().cpu().numpy(), size)
            h = np.where(h > 0.5, 1, 0).astype(np.int32)
        else:
            h = np.reshape(torch.sign(
                model(data)).detach().cpu().numpy(), size)

        result = zero_one_loss(h, t, is_logistic)
        presicion.append(result[0])
        recall.append(result[1])
        error_rate.append(result[2])
        acc_rate.append(result[3])

    return sum(presicion) / len(presicion), sum(recall) / len(recall), sum(acc_rate) / len(acc_rate)




def to_dataloader(my_x, my_y, batchsize):
    my_dataset = utils.TensorDataset(torch.from_numpy(my_x), torch.from_numpy(
        my_y))  # create your datset
    my_dataloader = utils.DataLoader(
        my_dataset, batch_size=batchsize, pin_memory=True)  # create your dataloader
    return my_dataloader

def resample(Xlp, Xup, Xun, model_name=""):
    print("======start resampling======")
    n_Xup = Xup.shape[0]
    n_Xlp = Xlp.shape[0]
    # positive for 1, negative for 0
    Yup = np.ones(Xup.shape[0])
    Yun = np.zeros(Xun.shape[0])

    Xu = np.asarray(np.concatenate((Xup, Xun), axis=0), dtype=np.float32)
    Yu_f = np.asarray(np.concatenate((Yup, Yun)), dtype=np.float32).reshape((-1, 1))
    Yu = np.asarray(np.concatenate((Yup, Yun)), dtype=np.int32)####

    perm = np.random.permutation(len(Yu))
    Xu, Yu_f, Yu = Xu[perm], Yu_f[perm], Yu[perm]
    batch_size = 128
    unlabel_dataset = to_dataloader(
        Xu, Yu_f, batchsize=batch_size)  # create positive and negative datsets in unlabeled dataset
    unlabel_val_dataset = to_dataloader(
        Xu, Yu, batchsize=batch_size)  # create dev set
    model=None
    # new code
    if model_name in ['res18']:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Change the number of output categories of the last fully connected layer to 2
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 1)
    elif model_name in ['3mlp']:
        if len(Xu.shape)==2:
            model=ThreeLayerPerceptron(Xu.shape[1])

    model=model.to(device)
    opt = torch.optim.Adam(
        model.parameters(), lr=1e-3, weight_decay=0.005)
    loss_func = torch.nn.BCEWithLogitsLoss()
    # train model with logistic regression
    for epoch in range(6):
        model.train()
        for data, target in unlabel_dataset:          # for each training step
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            opt.zero_grad()                # clear gradients for next train
            output = model(data)            # get output for every net
            loss = loss_func(output, target)  # compute loss for every net
            loss.backward()                # backpropagation, compute gradients
            opt.step()                     # apply gradients
        if (epoch+1) % 2 == 0:
            _, _, acc_rate= error(model,
                unlabel_val_dataset, is_logistic=True)
            print("Epoch:{0}, acc rate:{1:.3f}".format(epoch + 1, acc_rate))

    print("calculate p(y=+1|x)...")
    model.eval()

    print("resampling...")
    resample_from_unlable = True

    if resample_from_unlable:
        # resample from unlabeled positive data
        len_=len(Xup)
        Xup1=Xup[:int(len_/2)]
        Xup2=Xup[int(len_/2):]

        # 分批计算 `prob`
        prob_list = []
        with torch.no_grad():
            for i in range(0, len(Xup), batch_size):
                batch = torch.from_numpy(Xup[i:i + batch_size]).to(device)
                batch_prob = torch.sigmoid(model(batch)).detach().cpu().numpy()
                prob_list.append(batch_prob)

        prob = np.concatenate(prob_list, axis=0)

        prob = np.reshape(np.power(prob, 10), n_Xup)
        # normalize
        nor_prob = prob/np.sum(prob)
        # resample prob
        choose_index = np.random.choice(n_Xup, n_Xlp, replace=False, p=nor_prob)
        resample_Xlp = Xup[choose_index]
        resample_Xlp_ = Xup[choose_index]

    else:
        prob = torch.sigmoid(model(torch.from_numpy(
            Xlp).to(device))).detach().cpu().numpy()

        prob = np.reshape(np.power(prob, 10), n_Xlp)#########np.power(x, p)：计算数组 x 中每个元素的 p 次幂。
        # normalize
        nor_prob = prob/np.sum(prob)
        # resample prob
        choose_index = np.random.choice(n_Xlp, n_Xlp, replace=False, p=nor_prob)
        resample_Xlp = Xlp[choose_index]
        resample_Xlp_=numpy.unique(resample_Xlp)

    print('Actual sample number:',resample_Xlp_.shape[0],'all_pos_number:',n_Xlp)


    print("======finish resampling with positive data======")
    return resample_Xlp
# xlp=torch.rand(size=(1000,3,10,10))
# Xup=torch.rand(size=(29000,3,10,10))
# Xun=torch.rand(size=(30000,3,10,10))
# resample_Xlp=resample(xlp,Xup,Xun)
# print(resample_Xlp.shape)