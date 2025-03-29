import copy
import transformers
import helpers
import logging
from spl_utills import *
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from model import ThreeLayerPerceptron, MultiLayerPerceptron, CNN, Res18
# from dataset import load_dataset, to_dataloader
from dataset import load_dataset
from args import process_args, device
# from plot import draw_losses_test_data, draw_precision_recall
import lossFunc
import torch.nn.functional as F

moving_weights_all = None

def select_loss(loss_name):
    losses = {
        "sigmoid": lambda x: torch.sigmoid(-x)}
    return losses[loss_name]

def select_model(model_name):
    models = {"3mlp": ThreeLayerPerceptron,
              "6mlp": MultiLayerPerceptron, "cnn": CNN,'res18': Res18}
    return models[model_name]


def make_optimizer(model, stepsize):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=stepsize, weight_decay=0.005)
    return optimizer

def seed_all(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # 如果使用多 GPU
    torch.backends.cudnn.benchmark = False  # 禁用 cuDNN 的自动优化
    torch.backends.cudnn.deterministic = True  # 启用确定性行为
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # 禁用异步 CUDA 操作
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
class trainer():
    def __init__(self, models,  optimizers, optimizers_after, XYtrainLoader, XYvalidLoader, XYtestLoader, prior,args):
        self.models = models
        self.optimizers = optimizers
        self.optimizers_after=optimizers_after
        self.XYtrainLoader = XYtrainLoader
        self.XYvalidLoader = XYvalidLoader
        self.XYtestLoader = XYtestLoader
        self.prior = prior
        self.args=args

    # def train(self):
    def train(self,opt,pre_epochs):
        # setup some utilities for analyzing performance
        model=self.models
        trainLoader=self.XYtrainLoader
        positive_dataset, negative_dataset=self.separation_data(trainLoader)

        p_loader = DataLoader(positive_dataset, batch_size=self.args.batchsize, shuffle=True, drop_last=True,worker_init_fn=seed_worker)
        x_loader = DataLoader(negative_dataset, batch_size=self.args.batchsize, shuffle=True, drop_last=True,worker_init_fn=seed_worker)

        model.train()
        pre_loss=0.0
        best_val_acc=0
        best_val_model=None
        for pre_i in range(pre_epochs):
            x_iter = iter(x_loader)
            p_iter = iter(p_loader)

            running_loss = 0.0

            for batch_idx in range(self.args.val_iterations):
                try:
                    data_x, _ = next(x_iter)
                except:
                    x_iter = iter(x_loader)
                    data_x, _ = next(x_iter)

                try:
                    data_p, _ = next(p_iter)
                except:
                    p_iter = iter(p_loader)
                    data_p, _ = next(p_iter)

                if torch.cuda.is_available():
                    data_p, data_x = data_p.cuda(), data_x.cuda()

                # calculate the variational loss
                data_all = torch.cat((data_p, data_x))
                output_phi_all = model(data_all)
                log_phi_all = output_phi_all[:, 1]
                idx_p = slice(0, len(data_p))
                idx_x = slice(len(data_p), len(data_all))
                log_phi_x = log_phi_all[idx_x]
                log_phi_p = log_phi_all[idx_p]
                output_phi_x = output_phi_all[idx_x]
                var_loss = torch.logsumexp(log_phi_x, dim=0) - math.log(len(log_phi_x)) - 1 * torch.mean(log_phi_p)


                # perform Mixup and calculate the regularization
                target_x = output_phi_x[:, 1].exp()
                target_p = torch.ones(len(data_p), dtype=torch.float32)
                target_p = target_p.cuda() if torch.cuda.is_available() else target_p
                rand_perm = torch.randperm(data_p.size(0))


                data_p_perm, target_p_perm = data_p[rand_perm], target_p[rand_perm]
                m = torch.distributions.beta.Beta(self.args.mix_alpha, self.args.mix_alpha)
                lam = m.sample()
                data = lam * data_x + (1 - lam) * data_p_perm
                target = lam * target_x + (1 - lam) * target_p_perm
                if torch.cuda.is_available():
                    data = data.cuda()
                    target = target.cuda()
                out_log_phi_all = model(data)
                reg_mix_log = ((torch.log(target) - out_log_phi_all[:, 1]) ** 2).mean()

                # calculate gradients and update the network
                phi_loss = var_loss + self.args.lam * reg_mix_log
                opt.zero_grad()
                phi_loss.backward()
                opt.step()

                loss_all=phi_loss.item()
                running_loss+=loss_all
            accuracy, precision, recall, f1, roc_auc, average_precision = self.test(self.XYvalidLoader)
            print("{0}\t{1:-10}\t{2:-10}\t{3:-10}\t{4:-7}\t{5:-10}\t{6:-8}".format(
                pre_i, round(accuracy, 4), round(precision, 4), round(recall, 4), round(f1, 4),
                round(roc_auc, 4), round(average_precision, 4)))



            pre_loss+=running_loss / self.args.val_iterations
            if accuracy > best_val_acc:
                best_val_acc = accuracy
                best_val_model = copy.deepcopy(model.state_dict())

        model.load_state_dict(best_val_model)
        if not os.path.exists('VPUmodels'):
            os.makedirs('VPUmodels')
        torch.save(model.state_dict(),
                   f'VPUmodels/{self.args.dataset}__best_acc_{best_val_acc}_.pth')

        return pre_loss/pre_epochs, positive_dataset, negative_dataset

    def train_after(self, positive_dataset, unlabeled_dataset):

        model=self.models
        epochs = self.args.epochs
        batch_size = self.args.batchsize
        patience = self.args.patience####This is an early stop

        positive_data, positive_labels = positive_dataset.tensors
        unlabeled_data, unlabeled_labels = unlabeled_dataset.tensors

        # for SPL
        cl_scheduler_p = TrainingScheduler(self.args.scheduler_type_p, self.args.alpha_p, self.args.max_thresh_p, self.args.grow_steps_p,
                                           self.args.p,
                                           self.args.eta)
        cl_scheduler_n = TrainingScheduler(self.args.scheduler_type_n, self.args.alpha_n, self.args.max_thresh_n, self.args.grow_steps_n,
                                           self.args.p,
                                           self.args.eta)

        fea_all = []
        accuracy=None
        val_best_model=None
        val_best_acc=0
        val_best_index=None
        for episode in range(epochs):
            # get next lambda
            thresh_p = cl_scheduler_p.get_next_ratio()
            thresh_n = cl_scheduler_n.get_next_ratio()
            cur_data = torch.cat((positive_data, unlabeled_data), dim=0)
            cur_labels = torch.cat((positive_labels, torch.zeros_like(unlabeled_labels)), dim=0)
            cur_true_labels = torch.cat((positive_labels, unlabeled_labels), dim=0)

            perm = np.random.permutation(cur_data.size(0))
            cur_data, cur_labels, cur_true_labels = cur_data[perm], cur_labels[perm], cur_true_labels[perm]
            cur_loader = DataLoader(TensorDataset(cur_data, cur_labels, cur_true_labels), batch_size=batch_size,
                                    shuffle=True,drop_last=True, worker_init_fn=seed_worker)
            weighted_loader = self.weighted_dataloader(cur_loader, thresh_p, thresh_n)

            if self.args.vis and episode == 0:
                fea_all.append(self.get_fea(model, cur_loader, self.args))

            self.train_episode(model, weighted_loader)


            if self.args.vis:
                fea_all.append(self.get_fea(model, cur_loader, self.args))



            test_accuracy, test_precision, test_recall, test_f1, test_roc_auc, test_average_precision = self.test(self.XYtestLoader)
            val_accuracy, val_precision, val_recall, val_f1, val_roc_auc, val_average_precision = self.test(self.XYvalidLoader)

            print("{0}\t{1:-10}\t{2:-10}\t{3:-10}\t{4:-7}\t{5:-10}\t{6:-8}".format(
                episode, round(test_accuracy, 4), round(test_precision, 4), round(test_recall, 4), round(test_f1, 4),
                round(test_roc_auc, 4), round(test_average_precision, 4)),'#####test#####','val_acc=',val_accuracy)

            # Early stop
            if val_accuracy >= val_best_acc:
                val_best_acc = val_accuracy
                val_best_index = episode
                val_best_model = copy.deepcopy(model.state_dict())
            else:
                if episode - val_best_index >= patience:
                    print(f'=== Break at epoch {val_best_index + 1} ===')
                    break

        model.load_state_dict(val_best_model)
        if not os.path.exists('VPU_CL_models'):
            os.makedirs('VPU_CL_models')
        torch.save(model.state_dict(),
                   f'VPU_CL_models/{self.args.dataset}__best_acc_{val_best_acc}_.pth')

    def weighted_dataloader(self,dataloader, thresh_p, thresh_n):
        # calculate weights for all
        model=self.models
        training = model.training
        model.eval()
        data_all, labels_all, true_labels_all, weights_all, probs_all, fea_all = [], [], [], [], [], []
        global moving_weights_all
        with torch.no_grad():
            for data, labels, true_labels in dataloader:
                if torch.cuda.is_available():
                    data, labels, true_labels = data.cuda(), labels.cuda(), true_labels.cuda()

                if self.args.hardness in ['distance', 'cos']:
                    net_out, fea = model(data, return_fea=True)
                    fea_all.append(fea)
                else:
                    net_out = model(data)
                net_out=net_out[:, 1]
                data_all.append(data)
                labels_all.append(labels)
                true_labels_all.append(true_labels)

                # unlabeled data with linear weight
                probs = torch.sigmoid(net_out)
                probs_all.append(probs)

                if self.args.hardness in ['distance', 'cos']:
                    continue

                # loss for calculating unlabeled weight
                if self.args.hardness == 'logistic':
                    loss = lossFunc.logistic_loss(net_out / self.args.temper_n, -1)
                elif self.args.hardness == 'sigmoid':
                    loss = lossFunc.sigmoid_loss(net_out / self.args.temper_n, -1)
                elif self.args.hardness == 'crps':
                    loss = lossFunc.crps(net_out / self.args.temper_n, -1)
                elif self.args.hardness == 'brier':
                    loss = lossFunc.brier(net_out / self.args.temper_n, -1)
                elif self.args.hardness == 'focal':
                    loss = lossFunc.b_focal_loss(net_out / self.args.temper_n, -1 * torch.ones_like(labels),
                                                 gamma=self.args.focal_gamma, reduction='none')
                else:
                    raise ValueError(f'Invalid surrogate loss function {self.args.hardness}')
                # weights for unlabeled
                weights = calculate_spl_weights(loss.detach(), thresh_n, self.args)

                # loss for calculating labeled weight
                if self.args.hardness == 'logistic':
                    loss = lossFunc.logistic_loss(net_out / self.args.temper_p, 1)
                elif self.args.hardness == 'sigmoid':
                    loss = lossFunc.sigmoid_loss(net_out / self.args.temper_p, 1)
                elif self.args.hardness == 'crps':
                    loss = lossFunc.crps(net_out / self.args.temper_p, 1)
                elif self.args.hardness == 'brier':
                    loss = lossFunc.brier(net_out / self.args.temper_p, 1)
                elif self.args.hardness == 'focal':
                    loss = lossFunc.b_focal_loss(net_out / self.args.temper_p, torch.ones_like(labels),
                                                 gamma=self.args.focal_gamma, reduction='none')
                else:
                    raise ValueError(f'Invalid hardness function {self.args.hardness}')
                # weights for labeled
                weights[labels == 1] = calculate_spl_weights(loss[labels == 1].detach(), thresh_p, self.args)
                weights_all.append(weights)

            data_all = torch.cat(data_all, dim=0)
            labels_all = torch.cat(labels_all, dim=0)
            true_labels_all = torch.cat(true_labels_all, dim=0)

            if self.args.hardness == 'distance':
                fea_all = torch.cat(fea_all, dim=0)
                p_fea = fea_all[labels_all == 1]
                u_fea = fea_all[labels_all == -1]
                p_mean = p_fea.mean(dim=0)
                u_mean = u_fea.mean(dim=0)
                p_dis = Metrics.euclidean_distance(fea_all, p_mean)
                u_dis = Metrics.euclidean_distance(fea_all, u_mean)
                weights_all = torch.where(labels_all == 1, calculate_spl_weights(p_dis / u_dis, thresh_p, self.args),
                                          calculate_spl_weights(u_dis / p_dis, thresh_n, self.args))
            elif self.args.hardness == 'cos':
                fea_all = torch.cat(fea_all, dim=0)
                p_fea = fea_all[labels_all == 1]
                u_fea = fea_all[labels_all == -1]
                p_mean = p_fea.mean(dim=0)
                u_mean = u_fea.mean(dim=0)
                p_sim = F.cosine_similarity(fea_all, p_mean)
                u_sim = F.cosine_similarity(fea_all, u_mean)
                weights_all = torch.where(labels_all == 1, calculate_spl_weights(1. - p_sim, thresh_p, self.args),
                                          calculate_spl_weights(1 - u_sim, thresh_n, self.args))
            else:
                weights_all = torch.cat(weights_all, dim=0)
            if moving_weights_all is None:
                moving_weights_all = weights_all
            else:
                moving_weights_all = self.args.phi * moving_weights_all + (1. - self.args.phi) * weights_all
            probs_all = torch.cat(probs_all, dim=0)


        dataloader = DataLoader(TensorDataset(data_all, labels_all, true_labels_all, moving_weights_all), shuffle=True,
                                batch_size=self.args.batchsize,worker_init_fn=seed_worker)
        model.train(training)
        return dataloader



    def test(self, test_loader):

        # set the model to evaluation mode
        models=self.models
        models.eval()

        with torch.no_grad():
            for idx, (data, target) in enumerate(test_loader):
                if torch.cuda.is_available():
                    data, target = data.cuda(), target.cuda()
                log_phi = models(data)[:, 1]
                # log_phi -= log_max_phi
                if idx == 0:
                    log_phi_all = log_phi
                    target_all = target
                else:
                    log_phi_all = torch.cat((log_phi_all, log_phi))
                    target_all = torch.cat((target_all, target))
        pred_all = np.array((log_phi_all > math.log(0.5)).cpu().detach())
        log_phi_all = np.array(log_phi_all.cpu().detach())
        target_all = np.array(target_all.cpu().detach())
        all_labels, all_preds=target_all,pred_all
        accuracy = accuracy_score(all_labels, all_preds)
        # accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds,zero_division=0)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        roc_auc = roc_auc_score(all_labels, all_preds)
        average_precision = average_precision_score(all_labels, all_preds)
        test_acc = accuracy_score(target_all, pred_all)

        # test_auc = roc_auc_score(target_all, log_phi_all)
        return test_acc,precision,recall,f1,roc_auc,average_precision



    def get_fea(self, model, dataloader, args):
        model=self.models
        training = model.training
        model.eval()
        fea_all = []
        true_labels_all = []
        labels_all = []
        with torch.no_grad():
            for data, labels, true_labels in dataloader:
                if torch.cuda.is_available():
                    data, labels, true_labels = data.cuda(), labels.cuda(), true_labels.cuda()
                net_out, fea = model(data, return_fea=True)
                fea_all.append(fea.cpu().numpy())
                labels_all.append(labels.cpu().numpy())
                true_labels_all.append(true_labels.cpu().numpy())
            fea_all = np.concatenate(fea_all, axis=0)
            labels_all = np.concatenate(labels_all, axis=0)
            true_labels_all = np.concatenate(true_labels_all, axis=0)
        model.train(training)
        return fea_all, labels_all, true_labels_all

    def train_episode(self, model, dataloader):
        optimizer=self.optimizers_after

        if self.args.cos:
            scheduler = transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.args.n_warmup,
                                                                     num_training_steps=self.args.inner_epochs)
        else:
            scheduler = transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.args.n_warmup)

        # Test on training set before training
        model.eval()
        with torch.no_grad():
            meter = helpers.AverageMeter()
            for data, labels, true_labels, weights in dataloader:
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    data, labels, true_labels, weights = data.cuda(), labels.cuda(), true_labels.cuda(), weights.cuda()

                net_out = model(data)

                # loss w.r.t. pseudo labels
                if self.args.loss_second == 'bce':
                    loss = lossFunc.bce_loss(net_out[: ,1], labels, weights)
                elif self.args.loss_second == 'focal':
                    loss = lossFunc.b_focal_loss(net_out[: ,1], labels, weights, gamma=self.args.focal_gamma)
                meter.update(loss.item(), labels.size(0))
            logging.info(f'Loss before training: {meter.avg}')

        if self.args.restart:
            model.reset_para()
        model.train()
        tot_loss_meter = helpers.AverageMeter()
        tot_true_loss_meter = helpers.AverageMeter()
        tot_acc_meter = helpers.AverageMeter()
        for inner_epoch in range(self.args.inner_epochs): ##########这里的inner_epochs是20轮，也就是论文里面的E
            loss_meter = helpers.AverageMeter()
            for data, labels, true_labels, weights in dataloader:
                optimizer.zero_grad()
                if torch.cuda.is_available():
                    data, labels, true_labels, weights = data.cuda(), labels.cuda(), true_labels.cuda(), weights.cuda()

                net_out = model(data)
                # loss w.r.t. pseudo labels
                if self.args.loss_second == 'bce':
                    loss = lossFunc.bce_loss(net_out[:, 1], labels, weights)
                elif self.args.loss_second == 'focal':
                    loss = lossFunc.b_focal_loss(net_out[:, 1], labels, weights, gamma=self.args.focal_gamma)


                # loss w.r.t. true labels
                true_loss = lossFunc.bce_loss(net_out[:, 1], true_labels, weights)
                # acc w.r.t. true labels
                acc = Metrics.accuracy(net_out[:, 1], true_labels)

                tot_loss_meter.update(loss.item(), data.size(0))
                tot_acc_meter.update(acc, data.size(0))
                tot_true_loss_meter.update(true_loss.item(), data.size(0))

                loss.backward()
                optimizer.step()
                loss_meter.update(loss.item(), labels.size(0))

            scheduler.step()
            logging.debug(f'inner epoch [{inner_epoch + 1} / {self.args.inner_epochs}]  train loss: {loss_meter.avg}')


        return tot_loss_meter.avg, tot_acc_meter.avg, tot_true_loss_meter.avg

    def separation_data(self, trainLoader):
        # 初始化两个空列表来存储正类和负类的样本
        positive_samples = []
        negative_samples = []

        # 遍历trainLoader中的每个批次
        for data, labels in trainLoader:
            # 将正类样本和负类样本分别添加到对应的列表中
            positive_samples.append(data[labels == 1])
            negative_samples.append(data[labels == 0])

        # 将列表中的样本拼接成一个大的张量
        positive_samples = torch.cat(positive_samples, dim=0)
        negative_samples = torch.cat(negative_samples, dim=0)

        # 创建新的数据集和DataLoader
        positive_dataset = TensorDataset(positive_samples, torch.ones(positive_samples.size(0)))
        negative_dataset = TensorDataset(negative_samples, torch.zeros(negative_samples.size(0)))


        return positive_dataset, negative_dataset





    def run(self):
        print("Epoch\taccuracy\tprecision\trecall\tf1_score\troc_auc\tave_precision")
        # 5. 初始化 TensorBoard 的 SummaryWriter
        writer = SummaryWriter('runs/experiment_1')  # 日志将保存在 'runs/experiment_1' 目录中
        opt_phi=self.optimizers
        pre_epochs=self.args.pre_epochs
        model=self.models
        seed_all(self.args.seed)

        ###preprocessing stage，第一阶段
        if self.args.pretrained:
            prefix = "Model loaded from:"
            print(f'{prefix.rjust(100)}: {self.args.pretrained}.')
            model.load_state_dict(torch.load(self.args.pretrained))
            trainLoader = self.XYtrainLoader
            pos_data, unlabel_data = self.separation_data(trainLoader)

        else:
            loss_all,pos_data, unlabel_data =self.train(opt_phi, pre_epochs)
            print('#######The preprocessing stage has ended#######')

        #######Stage two and stage three

        seed_all(self.args.seed)
        self.train_after(pos_data, unlabel_data)

        writer.close()


def main():
    args = process_args()
    print("using:",device)

    print('所选数据集：',args.dataset)

    XYtrainLoader, XYvalidLoader, XYtestLoader, prior, dim = load_dataset(
        args.dataset, args.labeled, args.unlabeled, args.batchsize, with_bias=True, save_bias_dataset=args.save_bias_dataset, resample_model=args.model)


    # model setup
    print("input dim: {}".format(dim))
    print("prior: {}".format(prior))
    print("loss: {}".format(args.loss))
    print("batchsize: {}".format(args.batchsize))
    print("model: {}".format(args.model))
    print("labeled: {}".format(args.labeled))
    print("unlabeled: {}".format(args.unlabeled))
    print("lr: {}".format(args.stepsize))

    print("beta: {}".format(args.beta))
    print("gamma: {}".format(args.gamma))
    print("")

    loss_type = select_loss(args.loss)

    selected_model = select_model(args.model)
    if args.model in ['res18']:
        model = selected_model()
    else:
        model = selected_model(dim)
    models = copy.deepcopy(model).to(device)
    # loss_funcs = nnPUloss(prior, loss=loss_type, gamma=args.gamma, beta=args.beta)
    # trainer setup
    optimizers =  torch.optim.Adam(models.parameters(), lr=args.stepsize, betas=(0.5, 0.99))
    optimizers_after = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # training
    PUtrainer = trainer(models, optimizers, optimizers_after,
                        XYtrainLoader, XYvalidLoader, XYtestLoader, prior, args)
    PUtrainer.run()


if __name__ == '__main__':
    import os
    import sys
    os.chdir(sys.path[0])
    print("working dir: {}".format(os.getcwd()))
    main()
