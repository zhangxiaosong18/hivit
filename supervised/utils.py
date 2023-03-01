# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
from collections import OrderedDict

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def load_finetune(config, model, logger):
    logger.info(f"==============> Finetuning form {config.MODEL.FINETUNE}....................")
    checkpoint = torch.load(config.MODEL.FINETUNE, map_location='cpu')
    state_dict = OrderedDict()
    model_state_dict = model.state_dict()
    for k, v in checkpoint['model'].items():
        if k in ['relative_position_index']:
            continue
        elif k.endswith('patch_embed.proj.weight') and k in model_state_dict:
            S1 = v.size(-1)
            S2 = model_state_dict[k].size(-1)
            if S1 != S2:
                v = F.interpolate(
                    v,
                    scale_factor=(S2 / S1, S2 / S1),
                    mode='bicubic',
                )
        elif k.endswith('_pos_embed') and k in model_state_dict:
            S1 = int(math.sqrt(v.size(1)))
            S2 = int(math.sqrt(model_state_dict[k].size(1)))
            if S1 != S2:
                v = F.interpolate(
                    v.reshape(1, S1, S1, -1).permute(0, 3, 1, 2),
                    scale_factor=(S2 / S1, S2 / S1),
                    mode='bicubic',
                ).flatten(2).transpose(1, 2)
        elif k.endswith('.relative_position_bias_table') and k in model_state_dict:
            S1 = int(math.sqrt(v.size(0)))
            S2 = int(math.sqrt(model_state_dict[k].size(0)))
            if S1 != S2:
                v = F.interpolate(
                    v.reshape(1, S1, S1, -1).permute(0, 3, 1, 2),
                    scale_factor=(S2 / S1, S2 / S1),
                    mode='bicubic',
                ).flatten(2).transpose(1, 2)[0]
        state_dict[k] = v
    msg = model.load_state_dict(state_dict, strict=False)
    logger.info(msg)

def load_checkpoint(config, model, optimizer, lr_scheduler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, logger):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
