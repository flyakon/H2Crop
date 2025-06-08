

import torch
import torch.nn as nn
import os
from mmseg.models.builder import NECKS,MODELS
from mmengine.runner import load_checkpoint


@NECKS.register_module()
class HyperEmbedNeck(nn.Module):
    def __init__(self, in_channels,out_channels, in_keys='S2', auxiliary_keys=None):
        super(HyperEmbedNeck, self).__init__()
        self.in_keys = in_keys
        self.models=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
        )
        self.auiliary_keys = auxiliary_keys

    def forward(self, inputs):
        x_list=[]
        if isinstance(self.in_keys, list) or isinstance(self.in_keys, tuple):
           for key in self.in_keys:
                x=inputs[key]['ori_feature']
                if len(x.shape)==5:
                    B, D, C, H, W = x.shape
                    x = torch.reshape(x, (B, D * C, H, W))
                elif len(x.shape)==4:
                    B, C, H, W = x.shape
                else:
                    raise ValueError("Input shape is not supported")
                x_list.append(x)
           x_list=torch.cat(x_list,dim=1)
        else:
            x = inputs[self.in_keys]['ori_feature']
            if len(x.shape)==5:
                B, D, C, H, W = x.shape
                x = torch.reshape(x, (B, D * C, H, W))
            elif len(x.shape)==4:
                B, C, H, W = x.shape
            else:
                raise ValueError("Input shape is not supported")
            x_list=x
        if self.auiliary_keys is not None:
            for key in self.auiliary_keys:
                x_list = torch.cat([x_list, inputs[key]], dim=1)
        out = self.models(x_list)
        return out


@NECKS.register_module()
class MultiFusionNeck(nn.Module):
    def __init__(self,embed_dim,in_feature_key=('S2',),
                 feature_size=(16,16),out_size=(256,256),
                 in_fusion_key_list=({'S2':512,'HLS':512},
                                     {'S2':256},
                                     {'S2':128,},
                                     ),
                 hyper_embed_neck=None,embed_downsample=False
                 ):
        super(MultiFusionNeck, self).__init__()
        self.embed_dim=embed_dim
        self.fusion_list=nn.ModuleList()
        self.in_feature_key=in_feature_key
        self.feature_size=feature_size
        self.out_size=out_size
        self.embed_downsample=embed_downsample
        self.hyper_embed_neck=hyper_embed_neck
        self.in_fusion_key_list=in_fusion_key_list
        if self.hyper_embed_neck is not None:
            self.hyper_embed_neck=NECKS.build(self.hyper_embed_neck)
        if len(in_feature_key)==1:
            self.in_conv=nn.Identity()
        else:
            self.in_conv=nn.Sequential(
                nn.Conv2d(len(in_feature_key)*self.embed_dim,self.embed_dim,3,1,1),
                nn.BatchNorm2d(self.embed_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.embed_dim,self.embed_dim,3,1,1),
            )
        embed_dim=self.embed_dim
        pre_embed = embed_dim
        for fusion_keys in in_fusion_key_list:
            in_embed=sum(fusion_keys.values())
            if self.embed_downsample:
                embed_dim=embed_dim//2
            fusion=nn.Sequential(
                nn.Conv2d(in_embed+pre_embed,pre_embed,3,1,1),
                nn.BatchNorm2d(pre_embed),
                nn.ReLU(inplace=True),
                nn.Conv2d(pre_embed,embed_dim,3,1,1),
            )
            self.fusion_list.append(fusion)
            pre_embed=embed_dim


        self.out_conv=nn.Sequential(
            nn.Conv2d(pre_embed,pre_embed,3,1,1),
            nn.BatchNorm2d(pre_embed),
            nn.ReLU(inplace=True),
            nn.Conv2d(pre_embed,pre_embed,3,1,1),
        )

    def forward(self,inputs):
        in_features=[]
        for key in self.in_feature_key:
            features=inputs[key]['encoder_features']
            features=torch.nn.functional.interpolate(features,self.feature_size,mode='bilinear',align_corners=False)
            in_features.append(features)
        in_features=torch.cat(in_features,dim=1)
        in_features=self.in_conv(in_features)

        for i,fusion_keys in enumerate(self.in_fusion_key_list):
            in_features=torch.nn.functional.interpolate(in_features,scale_factor=2,mode='bilinear',align_corners=False)
            in_features_h, in_features_w=in_features.shape[-2:]
            in_features_idx=len(self.in_fusion_key_list)-i-1
            fusion_features=[]
            for key in fusion_keys:
                features=inputs[key]['features_list'][in_features_idx]
                features=torch.nn.functional.interpolate(features,(in_features_h,in_features_w),mode='bilinear',align_corners=False)
                fusion_features.append(features)
            fusion_features=torch.cat(fusion_features,dim=1)
            in_features=torch.cat([in_features,fusion_features],dim=1)
            in_features=self.fusion_list[i](in_features)
        out_features=self.out_conv(in_features)
        out_features=torch.nn.functional.interpolate(out_features,self.out_size,mode='bilinear',align_corners=False)
        if self.hyper_embed_neck is not None:
            hyper_embed_neck=self.hyper_embed_neck(inputs)
            out_features=torch.cat([out_features,hyper_embed_neck],dim=1)
        return out_features








