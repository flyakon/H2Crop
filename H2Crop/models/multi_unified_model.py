
from typing import Optional, Union, Dict

import torch
import torch.nn as nn
from mmseg.models.builder import MODELS
from mmengine.model import BaseModel
from mmengine.runner import load_state_dict,load_checkpoint
from mmseg.registry.registry import TRANSFORMS
@MODELS.register_module()
class MultiUnifiedModel(BaseModel):
    def __init__(self,encoders,head,neck=None,load_from=None,batch_augmentation=None):
        super().__init__()
        self.encoders=MODELS.build(encoders)
        if neck is not None:
            self.neck=MODELS.build(neck)
        else:
            self.neck=None
        self.heads=MODELS.build(head)

        if load_from is not None:
            load_checkpoint(self,load_from,strict=False)
        self.batch_augmentation = batch_augmentation
        if batch_augmentation is not None:
            self.batch_augmentation = TRANSFORMS.build(batch_augmentation)

    def forward(self,
                inputs:dict,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        if self.training and mode=='loss' and self.batch_augmentation is not None:
            inputs,data_samples=self.batch_augmentation(inputs,data_samples)
        outputs=self.encoders(inputs,data_samples,mode)
        if self.neck is not None:
            outputs=self.neck(outputs)
        if mode=='tensor' or mode=='predict':
            outputs=[self.heads((inputs,outputs),mode=mode)]
        else:
            logits,outputs=self.heads((inputs,outputs),data_samples,mode)
            self.result_list=logits
        return outputs




