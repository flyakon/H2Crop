
import torch
import torch.nn as nn
from einops import rearrange
import timm.models.vision_transformer
from mmseg.models.builder import BACKBONES,MODELS
from timm.models.layers import to_2tuple
from typing import List, Tuple, Optional, Dict, Union
from transformers.models.videomae.modeling_videomae import get_sinusoid_encoding_table
from .utils import get_specific_module,PatchEmbed2D
from mmengine.runner import load_checkpoint

@BACKBONES.register_module()
class SVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """ Vision Transformer with support for global average pooling
    """
    def __init__(self, img_size=64,patch_size=16,in_channels=10,embed_dim=768,out_embed_dim=128,output_size=192,
                    num_heads=12,num_layers=12,mlp_ratio=4,qkv_bias=True,drop_rate=0.,
                    init_cfg=None,
                    with_cls_token=False, **kwargs):
        super(SVisionTransformer, self).__init__(embed_dim=embed_dim,num_heads=num_heads,depth=num_layers,
                                                mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,patch_size=patch_size,
                                                in_chans=in_channels,num_classes=0,
                                                drop_rate=drop_rate)

        self.patch_emd = PatchEmbed2D(img_size=img_size,patch_size=patch_size,in_chans=in_channels,embed_dim=embed_dim)
        self.patch_size = self.patch_emd.patch_size
        self.num_patches = self.patch_emd.num_patches
        self.pos_embedding = get_sinusoid_encoding_table(self.num_patches, embed_dim)
        if init_cfg is not None:
            if isinstance(init_cfg, dict):
                checkpoint_path = init_cfg['checkpoint']
            else:
                checkpoint_path = init_cfg.checkpoint

            load_checkpoint(self.blocks,checkpoint_path,)
            self.blocks.is_init = True
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.with_cls_token = with_cls_token
        self.out_embed_dim = out_embed_dim
        self.out_size = output_size
        self.patch_resol=self.patch_emd.patch_resol
        scale=self.out_size//self.patch_resol[0]*self.out_size//self.patch_resol[1]
        self.out_conv=nn.Conv2d(embed_dim,scale*out_embed_dim,3,1,1)
        self.pixel_shuffle=nn.PixelShuffle(self.out_size//self.patch_resol[0])


    def forward(self, inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor'):
        x = self.patch_emd(inputs)
        if self.with_cls_token:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        B, _, C = x.shape
        x = x + self.pos_embedding.type_as(x).to(x.device).clone().detach()

        for blk in self.blocks:
            x = blk(x)
        x=rearrange(x,'b (h w) c -> b c h w',h=self.patch_resol[0],w=self.patch_resol[1])
        x=self.out_conv(x)
        x=self.pixel_shuffle(x)
        return x

@BACKBONES.register_module()
class PVisionTransformer(timm.models.vision_transformer.VisionTransformer):
    def __init__(self, img_size=64,patch_size=16,in_channels=10,embed_dim=768,out_embed_dim=128,output_size=192,
                    num_heads=12,num_layers=12,mlp_ratio=4,qkv_bias=True,drop_rate=0.,
                    init_cfg=None,
                    with_cls_token=False,patch_mode='linear', **kwargs):
        super(PVisionTransformer, self).__init__(embed_dim=embed_dim,num_heads=num_heads,depth=num_layers,
                                                mlp_ratio=mlp_ratio,qkv_bias=qkv_bias,patch_size=patch_size,
                                                in_chans=in_channels,num_classes=0,
                                                drop_rate=drop_rate)

        img_size=to_2tuple(img_size)
        if patch_mode=='linear':
            self.patch_emd = nn.Linear(img_size[0]*img_size[1],embed_dim)
        elif patch_mode=='avg':
            assert embed_dim==(img_size[0]*img_size[1])//patch_size**2
            self.patch_emd = nn.AvgPool2d(patch_size,patch_size)
        self.num_patches = in_channels
        self.pos_embedding = get_sinusoid_encoding_table(self.num_patches, embed_dim)
        if init_cfg is not None:
            if isinstance(init_cfg, dict):
                checkpoint_path = init_cfg['checkpoint']
            else:
                checkpoint_path = init_cfg.checkpoint

            load_checkpoint(self.blocks, checkpoint_path, )
            self.blocks.is_init = True

        if with_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.with_cls_token = with_cls_token
        self.out_embed_dim = out_embed_dim
        self.out_size = output_size

        self.out_post=nn.Linear(embed_dim,img_size[0]*img_size[1])
        self.patch_resol=img_size
        scale=self.out_size//self.patch_resol[0]*self.out_size//self.patch_resol[1]
        self.out_conv=nn.Conv2d(in_channels,scale*out_embed_dim,3,1,1)
        self.pixel_shuffle=nn.PixelShuffle(self.out_size//self.patch_resol[0])
        self.pacth_mode=patch_mode


    def forward(self, inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor'):
        if self.pacth_mode=='linear':
            x = rearrange(inputs,'b c h w -> b c (h w)')
            x = self.patch_emd(x)
        elif self.pacth_mode=='avg':
            x = self.patch_emd(inputs)
            x=rearrange(x,'b c h w -> b c (h w)')
        else:
            raise ValueError('patch_mode error')
        if self.with_cls_token:
            x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        B, _, C = x.shape
        x = x + self.pos_embedding.type_as(x).to(x.device).clone().detach()

        for blk in self.blocks:
            x = blk(x)
        x=self.out_post(x)
        x=rearrange(x,'b c (h w) -> b c h w',h=self.patch_resol[0],w=self.patch_resol[1])
        x=self.out_conv(x)
        x=self.pixel_shuffle(x)
        return x


@BACKBONES.register_module()
class SPVisionTransformer(nn.Module):
    def __init__(self,s_vit_cfg,p_vit_cfg,embed_dim,out_embed_dim):
        super(SPVisionTransformer, self).__init__()
        self.s_vit=MODELS.build(s_vit_cfg)
        self.p_vit=MODELS.build(p_vit_cfg)
        self.out_embed_dim=out_embed_dim
        self.out_post=nn.Sequential(*[
            nn.Conv2d(embed_dim,out_embed_dim,3,1,1),
            nn.ReLU(),
            nn.Conv2d(out_embed_dim,out_embed_dim,3,1,1),])

    def forward(self, inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor'):
        s_ori_feature=self.s_vit(inputs)
        p_ori_feature=self.p_vit(inputs)
        features=torch.concatenate([s_ori_feature,p_ori_feature],dim=1)
        features=self.out_post(features)
        return {'ori_feature':features}