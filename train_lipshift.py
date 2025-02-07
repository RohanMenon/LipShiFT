import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import time
from tqdm import tqdm

from timm.scheduler.cosine_lr import CosineLRScheduler
import torch
import yaml

import models
import tools

import warnings
warnings.filterwarnings("ignore")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_args():
    parser = argparse.ArgumentParser(
        'Training a Certifiably Robust LipShiFT: a Shift-based transformer architecture')

    parser.add_argument('--config',
                        default='configs/cifar10_shift_vit.yaml',
                        type=str,
                        help='path to the config yaml file')
    # checkpoint saving
    parser.add_argument('--work_dir', default='./checkpoints_all/', type=str)
    parser.add_argument('--ckpt_prefix', default='', type=str)
    parser.add_argument('--max_save', default=1, type=int)
    parser.add_argument('--resume_from', default='', type=str)
    # distributed training
    parser.add_argument('--launcher',
                        default='slurm',
                        type=str,
                        help='should be either `slurm` or `pytorch`')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=1)
    parser.add_argument('--aug', type=bool, default=False)

    return parser.parse_args()


def main():
    args = get_args()

    with open(args.config, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    model_cfg = cfg['model']
    train_cfg = cfg['training']
    dataset_cfg = cfg['dataset']
    gloro_cfg = cfg['gloro']

    if args.ckpt_prefix == '':
        prefix = f"{model_cfg['m_name']}"
        args.ckpt_prefix = prefix

    if args.resume_from:
        ckpt = torch.load(args.resume_from, 'cpu')
        backbone_ckpt = ckpt['backbone']
        optimizer_ckpt = ckpt['optimizer']
        start_epoch = ckpt['start_epoch']
        current_iter = ckpt['current_iter']
        training_logs = ckpt['training_logs']
        resume = True
    else:
        start_epoch = 0
        training_logs = []
        resume = False

    rank, local_rank, num_gpus = tools.init_DDP(args.launcher)
    print('Inited distributed training!')

    if local_rank == 0:
        os.system(f'cat {args.config}')

    print(f'Use checkpoint prefix: {args.ckpt_prefix}')

    train_loader, train_sampler, test_loader, _ = tools.data_loader(
        data_name=dataset_cfg['name'],
        batch_size=train_cfg['batch_size'] // num_gpus,
        num_classes=dataset_cfg['num_classes'],
        seed=dataset_cfg.get('seed', 2024))

    aug_loader, aug_sampler, _, _ = tools.data_loader(
        data_name=('ddpm' if args.aug else dataset_cfg['name']),
        batch_size=(train_cfg['batch_size'] * train_cfg['aug_ratio']) // num_gpus,
        num_classes=dataset_cfg['num_classes'],
        seed=dataset_cfg.get('seed', 2024))
    aug_iter = iter(aug_loader)

    model = models.LipShiFT(**model_cfg, **dataset_cfg)

    if resume:
        model.load_state_dict(backbone_ckpt)
    print(model)
    print(count_parameters(model))
    model = model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model,
                                                      device_ids=[local_rank],
                                                      output_device=local_rank)
    if cfg['training']['adam']:
        optim_fn = torch.optim.AdamW
    else:
        optim_fn = torch.optim.NAdam
    optimizer = optim_fn(model.parameters(),
                    eps=1e-8, betas=(0.9, 0.999),
                    lr=train_cfg['lr'],
                    weight_decay=train_cfg['weight_decay'])
    
    scheduler = CosineLRScheduler(
            optimizer,
            t_initial=int(cfg['training']['epochs']),
            lr_min=5e-6,
            warmup_lr_init=cfg['training']['lr'],
            warmup_t=int(cfg['training']['warmup_epochs']),
            cycle_limit=1,
            t_in_epochs=True,
        )
    
    if resume:
        optimizer.load_state_dict(optimizer_ckpt)
        scheduler.current_iter = current_iter
        scheduler.base_lr = optimizer_ckpt['param_groups'][0]['initial_lr']
        lipschitz = model.module.lipschitz().item()

    # compute variable epsilon 
    def eps_fn(epoch):
        ratio = min(epoch / train_cfg['epochs'] * 1.5, 1)
        ratio = gloro_cfg['min_eps'] + (gloro_cfg['max_eps'] -
                                        gloro_cfg['min_eps']) * ratio
        return gloro_cfg['eps'] * ratio

    os.makedirs(args.work_dir, exist_ok=True)

    train_fn = getattr(models, gloro_cfg['loss_type'])

    print('Begin Training')
    for log in training_logs:
        print(log)
    t = time.time()

    for epoch in range(start_epoch, train_cfg['epochs']):
        eps = eps_fn(epoch)
        train_sampler.set_epoch(epoch)
        aug_sampler.set_epoch(epoch)
        model.module.set_num_lc_iter(model_cfg['num_lc_iter'])

        model.train()
        correct_vra = correct = total = 0.
        for idx, (inputs, targets) in tqdm(enumerate(train_loader)):
            optimizer.zero_grad()
            bs = inputs.shape[0]
            lipschitz = model.module.lipschitz()

            try:
                input2, target2 = next(aug_iter)
            except StopIteration:
                aug_sampler.set_epoch(epoch)
                aug_iter = iter(aug_loader)
                input2, target2 = next(aug_iter)

            inputs = torch.cat([inputs, input2])
            targets = torch.cat([targets, target2])
            inputs = inputs.cuda()
            targets = targets.cuda()

            y, y_, loss = train_fn(model,
                                x=inputs,
                                label=targets,
                                lc=lipschitz,
                                eps=eps,
                                return_loss=True)
            
            loss.backward()
            if train_cfg['grad_clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(),
                                               train_cfg['grad_clip_val'])

            optimizer.step()

            correct += y.argmax(1).eq(targets)[:bs].sum().item()
            correct_vra += y_.argmax(1).eq(targets)[:bs].sum().item()
            total += bs

        scheduler.step(epoch)
    
        model.eval()
        model.module.set_num_lc_iter(100)  # let the power method converge
        # only need to compute the lipschitz only once for validation
        lipschitz = 1.0
        if gloro_cfg['eps'] != 0:
            lipschitz = model.module.lipschitz()
        val_correct_vra = val_correct = val_total = 0.
        with torch.no_grad():
            for inputs, targets in test_loader:
                inputs = inputs.cuda()
                targets = targets.cuda()
                
                y, y_, val_loss = models.trades_loss(model,
                                x=inputs,
                                label=targets,
                                eps=gloro_cfg['eps'],
                                lc=lipschitz,
                                return_loss=True)

                val_correct += y.argmax(1).eq(targets).sum().item()
                val_correct_vra += y_.argmax(1).eq(targets).sum().item()
                val_total += targets.size(0)
        
        collect_info = [
            correct_vra,
            correct,
            total,
            val_correct_vra, 
            val_correct,
            val_total
        ]
        collect_info = torch.tensor(collect_info,
                                    dtype=torch.float32,
                                    device=inputs.device).clamp_min(1e-9)
        torch.distributed.all_reduce(collect_info)

        acc_train = 100. * collect_info[1] / collect_info[2]
        acc_val = 100. * collect_info[4] / collect_info[5]

        acc_vra_train = 100. * collect_info[0] / collect_info[2]
        acc_vra_val = 100. * collect_info[3] / collect_info[5]

        used = time.time() - t
        t = time.time()

        string = (f'Epoch {epoch}: '
                f'Train acc {acc_train: .2f}%, '
                f'Train vra {acc_vra_train: .2f}%, '
                f'Val acc {acc_val: .2f}%, '
                f'Val VRA {acc_vra_val: .2f}%, '
                f'lipschitz:{lipschitz: .2E}, '
                f'Time:{used / 60: .2f} mins.')

        print(string)
        training_logs.append(string)

        if rank == 0:
            state = dict(backbone=model.module.state_dict(),
                        optimizer=optimizer.state_dict(),
                        start_epoch=epoch + 1,
                        current_iter=scheduler.t_in_epochs,
                        training_logs=training_logs,
                        configs=cfg)

            try: 
                path = f'{args.work_dir}/{args.ckpt_prefix}_{dataset_cfg['name']}_{epoch}.pth'
                torch.save(state, path)
            except PermissionError:
                print('Error saving checkpoint!')
                pass
            if epoch >= args.max_save:
                path = f'{args.work_dir}/{args.ckpt_prefix}_{dataset_cfg['name']}_{epoch - args.max_save}.pth'
                os.system('rm -f ' + (path))


if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    main()
