import argparse
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def process_args():
    parser = argparse.ArgumentParser(
        description='non-negative / unbiased PU learning Chainer implementation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batchsize', '-b', type=int, default=30000,
                        help='Mini batch size')
    parser.add_argument('--dataset', '-d', default='mnist-6mlp', type=str, choices=['mnist', 'cifar10','fmnist','adni','msc'],
                        help='The dataset name')
    parser.add_argument('--labeled', '-l', default=500, type=int,
                        help='# of labeled data')
    parser.add_argument('--unlabeled', '-u', default=59500, type=int,
                        help='# of unlabeled data')
    parser.add_argument('--epochs', '-e', default=100, type=int,
                        help='# of epochs to learn')
    parser.add_argument('--val_iterations', '-val_iter', default=30, type=int,
                        help='# val_iterations')
    parser.add_argument('--seed', '-s', default=0, type=int,
                        help='# random number')
    parser.add_argument('--pre_epochs', '-pre_e', default=50, type=int,
                        help='# Pre-training rounds ')
    parser.add_argument('--scheduler_type_p', type=str, default='linear',
                        help='type of training scheduler for labeled (default linear)')
    parser.add_argument('--scheduler_type_n', type=str, default='linear',
                        help='type of training scheduler for labeled (default linear)')
    parser.add_argument('--alpha_p', type=float, default=0.1, help='initial threshold for labeled (default 0.1)')
    parser.add_argument('--alpha_n', type=float, default=0.11, help='initial threshold for unlabeled (default 0.1)')
    parser.add_argument('--max_thresh_p', type=float, default=1., help='maximum of threshold for labeled (default 2.0)')
    parser.add_argument('--max_thresh_n', type=float, default=1., help='maximum of threshold for unlabeled (default 2.0)')
    parser.add_argument('--grow_steps_p', type=int, default=5, help='number of step to grow to max_thresh for labeled (default 10)')
    parser.add_argument('--grow_steps_n', type=int, default=5, help='number of step to grow to max_thresh for unlabeled (default 10)')
    parser.add_argument('--temper_p', type=float, default=1.5, help='temperature to smooth logits for labeled (default: 1.0)')
    parser.add_argument('--temper_n', type=float, default=1.5, help='temperature to smooth logits for unlabeled (default: 1.0)')
    parser.add_argument('--focal_gamma', type=float, default=1.0, help='gamma for focal loss')
    parser.add_argument('--pretrained', type=str, default=r'D:\python_new\statistical_modeling\VPUmodels\adni__best_acc_0.765625_.pth',
                        help='pre-trained model path (default None)')
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--cos', action='store_true',
                        help='Use cosine lr scheduler (default False)')
    parser.add_argument('--loss_second', type=str, default='bce', choices=['bce', 'nnpu', 'upu', 'focal'])

    parser.add_argument('--n_warmup', default=0, type=int,
                        help='Number of warm-up steps (default: 0)')
    parser.add_argument('--pre_n_warmup', type=int, default=0,
                        help='number of warm-up steps in pre-training (default 0)')
    parser.add_argument('--inner_epochs', type=int, default=5,
                        help='number of epochs to run after each dataset update (default: 1)')
    parser.add_argument('--restart', action='store_true',
                        help='reset model before training in each episode (default: False)')

    parser.add_argument('--spl_type', type=str, default='welsch', help='type of soft sp-regularizer (default welsch)')
    parser.add_argument('--phi', type=float, default=0., help='momentum for weight moving average (default: 0.)')

    parser.add_argument('--hardness', type=str, default='logistic',
                        help='hardness function used to calculate weights (default: logistic)')
    parser.add_argument('--eta', type=float, default=1.1,
                        help='alpha *= eta in each step for scheduler exp (default 1.1)')
    parser.add_argument('--p', type=int, default=2, help='p for scheduler root-p (default 2)')
    # parser.add_argument('--epoch', '-e', default=10, type=int,
    #                     help='# of epochs to learn')
    parser.add_argument('--beta', '-B', default=0., type=float,
                        help='Beta parameter of nnPUSB')
    parser.add_argument('--gamma', '-G', default=1., type=float,
                        help='Gamma parameter of nnPUSB')
    parser.add_argument('--mix-alpha', type=float, default=0.3, help="parameter in Mixup")

    parser.add_argument('--lam', type=float, default=0.03, help="weight of the regularizer")

    parser.add_argument('--loss', type=str, default="sigmoid", choices=['sigmoid'],
                        help='The name of a loss function')
    parser.add_argument('--model', '-m', default='6mlp', choices=['3mlp','6mlp','cnn'],
                        help='The name of a classification model')
    parser.add_argument('--stepsize', '-s_', default=3e-5, type=float,
                        help='Stepsize of gradient method. This is deal with VPU')###1e-3
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='learning rate (default: 1e-4)')
    parser.add_argument('--wd', default=0., type=float, help='weight decay (default 0.)')
    parser.add_argument('--patience', default=5, type=int, help='patience for early stopping (default 5)')
    parser.add_argument('--save_bias_dataset', type=str, default='./Pos_data/adni',
                        help='Whether you need to save paranoid data (default None)')

    parser.add_argument('--out', '-o', default='result',
                        help='Directory to output the result')
    parser.add_argument('--preset', '-p', type=str, default='adni',
                        choices=['mnist','mnist-6mlp','cifar10','fmnist-6mlp','adni','vaila','casis','dm'],
                        help="Preset of configuration\n" +
                             "mnist: The setting of MNIST experiment in Experiment")
    args = parser.parse_args()
    if args.preset == "mnist":
        args.labeled = 100
        args.unlabeled = 60000
        args.dataset = "mnist"
        args.batchsize = 30000
        args.epoch = 100
        args.model = "3mlp"
    elif args.preset == "mnist-6mlp":
        args.labeled = 1000
        args.unlabeled = 59000
        args.dataset = "mnist"
        args.batchsize = 500
        args.epoch = 300
        args.model = "6mlp"
    elif args.preset == "fmnist-6mlp":
        args.labeled = 500
        args.unlabeled = 60000
        args.dataset = "fmnist"
        args.batchsize = 30000
        args.epoch = 200
        args.model = "6mlp"
    elif args.preset == "grid":
        args.labeled = 1000
        args.unlabeled = 6000
        args.dataset = "grid"
        args.batchsize = 200
        args.epoch = 200
        args.model = "6mlp"
    elif args.preset == "cifar10":
        args.labeled = 1000
        args.unlabeled = 49000
        args.dataset = "cifar10"
        args.batchsize = 500
        args.model = "cnn"
        args.stepsize = 3e-5
        args.epoch = 50
    elif args.preset == "vaila":
        args.labeled = 2000
        args.unlabeled = 10430
        args.dataset = "vaila"
        args.batchsize =500
        args.model = "6mlp"
        args.epoch = 200
    elif args.preset == "dm":
        args.labeled = 200
        args.unlabeled = 5800
        args.dataset = "dm"
        args.batchsize = 256
        args.model = "3mlp"
        # args.stepsize = 1e-6
        args.epoch =200
    elif args.preset == 'adni':
        args.labeled = 1000
        args.unlabeled = 4120
        args.dataset = 'adni'
        args.batchsize = 64
        args.model ="res18"
        # args.stepsize = 1e-6
        args.epoch =10
    elif args.preset == 'casis':
        args.labeled = 1000
        args.unlabeled = 5352
        args.dataset = 'casis'
        args.batchsize = 64
        args.model ="res18"
        # args.stepsize = 1e-6
        args.epoch =10


    assert (args.batchsize > 0)
    assert (args.epochs > 0)
    assert (0 < args.labeled < 30000)
    if args.dataset == "mnist":
        assert (0 < args.unlabeled <= 60000)
    elif args.dataset == "fmnist":
        assert (0 < args.unlabeled <= 60000)
    else:
        assert (0 < args.unlabeled <= 70000)
    assert (0. <= args.beta)
    assert (0. <= args.gamma <= 1.)
    return args