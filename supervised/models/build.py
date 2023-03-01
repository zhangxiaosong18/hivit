# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
from functools import partial
from torch import nn

from .swin_transformer import SwinTransformer
from .swin_mlp import SwinMLP
from .hivit import HiViT


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type == 'swin_mlp':
        model = SwinMLP(img_size=config.DATA.IMG_SIZE,
                        patch_size=config.MODEL.SWIN_MLP.PATCH_SIZE,
                        in_chans=config.MODEL.SWIN_MLP.IN_CHANS,
                        num_classes=config.MODEL.NUM_CLASSES,
                        embed_dim=config.MODEL.SWIN_MLP.EMBED_DIM,
                        depths=config.MODEL.SWIN_MLP.DEPTHS,
                        num_heads=config.MODEL.SWIN_MLP.NUM_HEADS,
                        window_size=config.MODEL.SWIN_MLP.WINDOW_SIZE,
                        mlp_ratio=config.MODEL.SWIN_MLP.MLP_RATIO,
                        drop_rate=config.MODEL.DROP_RATE,
                        drop_path_rate=config.MODEL.DROP_PATH_RATE,
                        ape=config.MODEL.SWIN_MLP.APE,
                        patch_norm=config.MODEL.SWIN_MLP.PATCH_NORM,
                        use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type == 'hivit':
        model = HiViT(
            img_size=config.DATA.IMG_SIZE, 
            patch_size=config.MODEL.VIT.PATCH_SIZE, 
            inner_patches=config.MODEL.VIT.INNER_PATCHES, 
            in_chans=config.MODEL.VIT.IN_CHANS, 
            num_classes=config.MODEL.NUM_CLASSES, 
            embed_dim=config.MODEL.VIT.EMBED_DIM, 
            depths=config.MODEL.VIT.DEPTHS, 
            num_heads=config.MODEL.VIT.NUM_HEADS, 
            stem_mlp_ratio=config.MODEL.VIT.STEM_RATIO, 
            mlp_ratio=config.MODEL.VIT.MLP_RATIO, 
            qkv_bias=True, qk_scale=None, 
            drop_rate=config.MODEL.DROP_RATE, 
            drop_path_rate=config.MODEL.DROP_PATH_RATE,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), 
            ape=config.MODEL.VIT.APE, rpe=config.MODEL.VIT.RPE, 
            use_checkpoint=config.TRAIN.USE_CHECKPOINT, 
            patch_norm=config.MODEL.VIT.PATCH_NORM,
            kernel_size=config.MODEL.VIT.STEM_KERNEL,
            pad_size=config.MODEL.VIT.STEM_PAD,
        )
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
