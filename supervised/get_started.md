## Install

### Requirements:
- Python3
- CUDA 11.1
- PyTorch 1.8+ with CUDA support
- timm 0.5.4
- apex
- opencv-python
- termcolor
- yacs

### Step-by-step installation

```bash
git clone https://github.com/zhangxiaosong18/hivit.git
cd hivit/supervised

conda create -n hivit python=3.9 -y
conda activate hivit

pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install timm==0.5.4 opencv-python termcolor yacs
sh ../install_apex.sh
```

## Training
#### Tiny model
```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/hivit_tiny_224.yaml --data-path <imagenet-path> --batch-size 128
```

#### Small model
```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/hivit_small_224.yaml --data-path <imagenet-path> --batch-size 128
```

#### Base model
```bash
python -m torch.distributed.launch --nproc_per_node 8 --master_port 12345  main.py \
--cfg configs/hivit_base_224.yaml --data-path <imagenet-path> --batch-size 128
```
