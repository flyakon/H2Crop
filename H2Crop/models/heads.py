
from typing import Optional, Union, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models.builder import MODELS,LOSSES

from mmengine.model import BaseModule,BaseModel
@MODELS.register_module()
class CropFCNHead(BaseModel):
    def __init__(self,embed_dim,num_classes,loss_model):
        super().__init__()
        self.embed_dim=embed_dim
        self.num_classes=num_classes
        self.loss_model=LOSSES.build(loss_model)
        self.head=nn.Sequential(
            nn.Conv2d(self.embed_dim,self.embed_dim//2,kernel_size=3,stride=1,padding=1),
            # nn.BatchNorm2d(self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.embed_dim//2,self.num_classes,kernel_size=1,stride=1,padding=0)
        )

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        logits=self.head(inputs)
        if mode=='loss':
            loss=self.loss_model(logits,data_samples)
            return logits,loss
        else:
            return logits


@MODELS.register_module()
class CascadeFCNHead(BaseModel):
    def __init__(self,cascade_head_cfg,):
        super().__init__()
        self.cascade_head_cfg=cascade_head_cfg
        self.heads_dict=nn.ModuleDict()
        for head_name,head_cfg in cascade_head_cfg.items():
            self.heads_dict[head_name]=MODELS.build(head_cfg)

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        inputs=inputs[1]
        if mode=='loss':
            loss_dict={}
            results_dict={}
            last_feature=inputs
            for head_name,head in self.heads_dict.items():
                logit,loss=head(last_feature,data_samples[head_name],mode)
                loss_dict['%s_loss'%head_name]=loss
                results_dict[head_name]=logit
                last_feature=torch.cat([inputs,logit],dim=1)
            return results_dict,loss_dict
        else:
            results_dict={}
            last_feature=inputs
            for head_name,head in self.heads_dict.items():
                logit=head(last_feature,mode=mode)
                results_dict[head_name]=logit
                last_feature=torch.cat([inputs,logit],dim=1)
            return results_dict

@MODELS.register_module()
class CascadeFCNPostCropsHead(BaseModel):
    def __init__(self,cascade_head_cfg,with_priors=True):
        super().__init__()
        self.cascade_head_cfg=cascade_head_cfg
        self.heads_dict=nn.ModuleDict()
        for head_name,head_cfg in cascade_head_cfg.items():
            self.heads_dict[head_name]=MODELS.build(head_cfg)
        self.with_priors=with_priors

    def forward(self,
                inputs: torch.Tensor,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        inputs_features=inputs[1]
        if self.with_priors:
            ref_crop_data=inputs[0]['priors']
        if mode=='loss':
            loss_dict={}
            results_dict={}
            last_feature=inputs_features
            for head_name,head in self.heads_dict.items():
                num_classes=head.num_classes
                if self.with_priors:
                    ref_crops_level=ref_crop_data[head_name]
                    ref_crops_level=nn.functional.one_hot(ref_crops_level,num_classes=num_classes).permute(0,3,1,2).float()
                    last_feature=torch.cat([last_feature,ref_crops_level],dim=1)
                logit,loss=head(last_feature,data_samples[head_name],mode)
                loss_dict['%s_loss'%head_name]=loss
                results_dict[head_name]=logit
                last_feature=torch.cat([inputs_features,logit],dim=1)
            return results_dict,loss_dict
        else:
            results_dict={}
            last_feature=inputs_features
            for head_name,head in self.heads_dict.items():
                num_classes = head.num_classes
                if self.with_priors:
                    ref_crops_level = ref_crop_data[head_name]
                    ref_crops_level = nn.functional.one_hot(ref_crops_level, num_classes=num_classes).permute(0, 3, 1,
                                                                                                              2).float()
                    last_feature = torch.cat([last_feature, ref_crops_level], dim=1)
                logit=head(last_feature,mode=mode)
                results_dict[head_name]=logit
                last_feature=torch.cat([inputs_features,logit],dim=1)
            return results_dict
