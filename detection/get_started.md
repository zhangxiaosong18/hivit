## Install

### Requirements:
- Python3
- CUDA 11.1
- PyTorch 1.8+ with CUDA support
- timm 0.5.4
- mmcv-full 1.6.0
- opencv-python
- apex

### Step-by-step installation

```bash
git clone https://github.com/zhangxiaosong18/hivit.git
cd hivit/detection

conda create -n hivit python=3.9 -y
conda activate hivit

pip install torch==1.8.2 torchvision==0.9.2 torchaudio==0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111
pip install mmcv-full==1.6.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.8/index.html
pip install opencv-python timm==0.5.4
sh ../install_apex.sh

pip install -e .
```

## Fine-tuning
```bash
chmod -R +x tools
./tools/dist_train.sh configs/_hivit_/hivit_base_mask_rcnn_fpn_3x_coco.py 8 \
    --cfg-options \
    model.backbone.init_cfg.checkpoint=mae_hivit_base_1600ep.pth \
    data.samples_per_gpu=2 \
    --work-dir <detection-output-path>
```
