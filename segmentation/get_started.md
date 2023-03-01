## Install

### Requirements:
- Python3
- CUDA 11.1
- PyTorch 1.8+ with CUDA support
- timm 0.5.4
- mmcv-full 1.6.0
- mmsegmentation 0.26.0
- opencv-python
- scipy
- apex

### Step-by-step installation

```bash
git clone https://github.com/zhangxiaosong18/hivit.git
cd hivit/segmentation

conda create -n hivit python=3.9 -y
conda activate hivit

pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8/index.html
pip install opencv-python scipy timm==0.5.4 mmsegmentation==0.26.0
sh ../install_apex.sh
```

## Fine-tuning

```bash
chmod -R +x tools
./tools/dist_train.sh configs/upernet_hivit_base_80k_ade20k.py 8 \
    --options model.pretrained=mae_hivit_base_1600ep_ft100ep.pth \
    --work-dir <segmentation-output-path>
```
