# -*- coding: utf-8 -*-
# @Time    : 20/10/2024 8:08 pm
# @Author  : Wenyuan Li
# @File    : utils.py
# @Description :
import  torch
from mmseg.models.builder import NECKS
import torch.nn as nn
import collections.abc

def get_specific_module(checkpoint_file,module_name,replace_name,state_key=None):
    state_dict=torch.load(checkpoint_file)
    if state_key is not None:
        state_dict=state_dict[state_key]
    state_dict_s={}
    for key,value in state_dict.items():
        if key.startswith(module_name):
            state_dict_s[key.replace(module_name,replace_name)]=value
    return state_dict_s


@NECKS.register_module()
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=256, patch_size=16, in_chans=6, embed_dim=768, num_frames=16, tubelet_size=2):
        super().__init__()
        img_size = img_size if isinstance(img_size, collections.abc.Iterable) else (img_size, img_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        self.tubelet_size = int(tubelet_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0]) * (
                    num_frames // self.tubelet_size)
        self.num_patch_per_phase=(img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_channels=in_chans, out_channels=embed_dim,
                              kernel_size=(self.tubelet_size, patch_size[0], patch_size[1]),
                              stride=(self.tubelet_size, patch_size[0], patch_size[1]))
    def forward(self, x, **kwargs):
        x=x.permute(0,2,1,3,4)
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class PatchEmbed2D(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=256, patch_size=16, in_chans=6, embed_dim=768):
        super().__init__()
        img_size = img_size if isinstance(img_size, collections.abc.Iterable) else (img_size, img_size)
        patch_size = patch_size if isinstance(patch_size, collections.abc.Iterable) else (patch_size, patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.patch_resol= (img_size[1] // patch_size[1], img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x, **kwargs):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x