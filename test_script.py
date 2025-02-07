import argparse
import os
import time
import numpy as np

import torch
import yaml

import models
import tools

import warnings
warnings.filterwarnings("ignore")

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def get_args():
    parser = argparse.ArgumentParser(
        'Testing a Certifiably Robust LipShiFT: a Shift-based transformer architecture')

    parser.add_argument('--config',
                        type=str,
                        help='path to the config yaml file')
    # checkpoint saving
    parser.add_argument('--work_dir', default='./checkpoint/', type=str)
    parser.add_argument('--ckpt_prefix', default='', type=str)
    parser.add_argument('--max_save', default=1, type=int)
    parser.add_argument('--resume_from', default='', type=str)
    # distributed training
    parser.add_argument('--launcher',
                        default='slurm',
                        type=str,
                        help='should be either `slurm` or `pytorch`')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=1)

    return parser.parse_args()


def main():
    args = get_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    model_cfg = cfg['model']
    train_cfg = cfg['training']
    dataset_cfg = cfg['dataset']
    if 'gloro' in cfg.keys():
        gloro_cfg = cfg['gloro']
        train_fn = getattr(models, gloro_cfg['loss_type'])

    rank, local_rank, num_gpus = tools.init_DDP(args.launcher)
    print('Inited distributed training!')

    if local_rank == 0:
        os.system(f'cat {args.config}')

    print(f'Use checkpoint prefix: {args.ckpt_prefix}')


    seed = dataset_cfg['seed'] + rank
    torch.manual_seed(seed)
    seed = np.random.seed(seed)

    _, _, val_loader, _ = tools.data_loader(
        data_name=dataset_cfg['name'],
        batch_size=train_cfg['batch_size'] // num_gpus,
        num_classes=dataset_cfg['num_classes'],
        seed=seed)

    model = models.LipShiFT(**model_cfg, **dataset_cfg)
    weights = torch.load('checkpoints_all/final/dropout/lipshift_cifar10_499.pth')
    model.load_state_dict(weights['backbone'])

    model = model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)

    os.makedirs(args.work_dir, exist_ok=True)

    t = time.time()
    
    model.eval()
    model.module.set_num_lc_iter(100)  # let the power method converge
    # only need to comput the sub_lipschitz only once for validation
    sub_lipschitz = 1.0
    if gloro_cfg['eps'] != 0:
        sub_lipschitz = model.module.lipschitz().item()

    val_correct_vra = val_correct = val_total = 0.

    for inputs, targets in val_loader:
        inputs = inputs.cuda(non_blocking=True)
        targets = targets.cuda(non_blocking=True)
        with torch.no_grad():
            y, y_, _ = models.trades_loss(model,
                                            x=inputs,
                                            label=targets,
                                            eps=gloro_cfg['eps'],
                                            lc=sub_lipschitz,
                                            return_loss=False)

        val_correct += y.argmax(1).eq(targets).sum().item()
        val_correct_vra += y_.argmax(1).eq(targets).sum().item()
        val_total += targets.size(0)
    
    collect_info = [
        val_correct_vra,
        val_correct,
        val_total
    ]
    collect_info = torch.tensor(collect_info,
                                dtype=torch.float32,
                                device=inputs.device).clamp_min(1e-9)
    torch.distributed.all_reduce(collect_info)

    acc_val = 100. * collect_info[1] / collect_info[2]

    acc_vra_val = 100. * collect_info[0] / collect_info[2]

    used = time.time() - t
    t = time.time()

    string = (f'val acc{acc_val: .2f}%, '
              f'{acc_vra_val: .2f}%. '
            f'lipschitz:{sub_lipschitz: .2f}. '
            f'Time:{used / 60: .2f} mins.')

    print(string)


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    main()
