# Original script adapted from https://github.com/fra31/auto-attack/blob/master/autoattack/examples/eval.py 

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse
from pathlib import Path
import numpy as np

import yaml

import torch
import torchvision.datasets as datasets
import torch.utils.data as data
import torchvision.transforms as transforms

from tools.dataset.tinyimgnet import simple_dataset

import sys
sys.path.insert(0,'..')

from models.model import LipShiFT

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/lipshift/cifar10.yaml', help='path to the config yaml file')
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--norm', type=str, default='L2')
    parser.add_argument('--epsilon', type=float, default=0.1411764705882353) 
    parser.add_argument('--model', type=str, default='checkpoints_all/final/main_results/shift_vit_cifar10_499.pth')
    parser.add_argument('--n_ex', type=int, default=10000)
    parser.add_argument('--individual', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--batch_size', type=int, default=500)
    parser.add_argument('--log_path', type=str, default='./autoattack_results/cifar10.txt')
    parser.add_argument('--version', type=str, default='standard')
    parser.add_argument('--state-path', type=Path, default=None)
    parser.add_argument('--dataset', type=str, default='cifar10')  #possible values: cifar10, cifar100, tiny
    
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    model_cfg = cfg['model']
    dataset_cfg = cfg['dataset']


    # load model
    model = LipShiFT(**model_cfg, **dataset_cfg)

    # load checkpoint
    ckpt = torch.load(args.model)
    model.load_state_dict(ckpt['backbone'])
    model.cuda()
    model.eval()

    # load data
    transform_list = [transforms.ToTensor()]
    transform_chain = transforms.Compose(transform_list)

    if args.dataset == 'cifar10':
        item = datasets.CIFAR10(root=args.data_dir, train=False, transform=transform_chain, download=True)
    elif args.dataset == 'cifar100':
        item = datasets.CIFAR100(root=args.data_dir, train=False, transform=transform_chain, download=True)
    elif args.dataset == 'tiny':
        data_tiny = np.load(f'{args.data_dir}/tinyimagnet.npz')
        valX = data_tiny['valX']
        valY = data_tiny['valY']
        
        item = simple_dataset(valX, valY, transforms.ToTensor())
    else:
        print(f'Dataset not supported!')

    test_loader = data.DataLoader(item, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    # create save dir
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    
    # load attack    
    from autoattack import AutoAttack
    adversary = AutoAttack(model, norm=args.norm, eps=args.epsilon, log_path=args.log_path,
        version=args.version)
    
    l = [x for (x, y) in test_loader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in test_loader]
    y_test = torch.cat(l, 0)
    
    # example of custom version
    if args.version == 'custom':
        adversary.attacks_to_run = ['apgd-ce', 'fab']
        adversary.apgd.n_restarts = 2
        adversary.fab.n_restarts = 2
    
    # run attack and save images
    with torch.no_grad():
        if not args.individual:
            adv_complete = adversary.run_standard_evaluation(x_test[:args.n_ex], y_test[:args.n_ex],
                bs=args.batch_size, state_path=args.state_path)

            torch.save({'adv_complete': adv_complete}, '{}/{}_{}_1_{}_eps_{:.5f}.pth'.format(
                args.save_dir, 'aa', args.version, adv_complete.shape[0], args.epsilon))

        else:
            # individual version, each attack is run on all test points
            adv_complete = adversary.run_standard_evaluation_individual(x_test[:args.n_ex],
                y_test[:args.n_ex], bs=args.batch_size)
            
            torch.save(adv_complete, '{}/{}_{}_individual_1_{}_eps_{:.5f}_plus_{}_cheap_{}.pth'.format(
                args.save_dir, 'aa', args.version, args.n_ex, args.epsilon))
                