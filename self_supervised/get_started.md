## Install

### Requirements:
- Python3
- CUDA 11.1
- PyTorch 1.8+ with CUDA support
- timm 0.5.4
- tensorboard

### Step-by-step installation

```bash
git clone https://github.com/zhangxiaosong18/hivit.git
cd hivit/self_supervised

conda create -n hivit python=3.9 -y
conda activate hivit

pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install timm==0.5.4 tensorboard
```

## Pre-training
```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 --use_env main_pretrain.py
    --data_path <imagenet-path> --output_dir <pertraining-output-path>
    --model mae_hivit_base_dec512d6b --norm_pix_loss --mask_ratio 0.75
    --batch_size 256 --accum_iter 2 --blr 1e-4 --weight_decay 0.05 --epochs 1600 --warmup_epochs 40 
```

## Fine-tuning
```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345 --use_env main_finetune.py
    --data_path <imagenet-path> --output_dir <finetuning-output-path> 
    --finetune <pertraining-output-path>/checkpoint-1600.pth
    --model hivit_base --dist_eval
    --batch_size 64 --accum_iter 2 --blr 5e-4 --layer_decay 0.85 --weight_decay 0.05 
    --drop_path 0.2 --reprob 0.25 --mixup 0.8 --cutmix 1.0 --epochs 100 --warmup_epochs 5
```
