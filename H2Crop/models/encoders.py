
import torch
import torch.nn as nn
import numpy as np
from transformers.models.videomae import VideoMAEModel,VideoMAEConfig
from typing import List, Tuple, Optional, Dict, Union
from mmseg.registry import MODELS
from mmseg.registry.registry import TRANSFORMS
from mmengine.model import BaseModule,BaseModel
from torch.nn.modules.transformer import TransformerDecoderLayer,TransformerDecoder
from transformers.models.videomae.modeling_videomae import get_sinusoid_encoding_table
from ..utils import path_utils
from mmengine.registry import RUNNERS
import copy
import os
from torch.nn.parallel.distributed import DistributedDataParallel
from mmengine.model.wrappers.utils import is_model_wrapper
from mmengine.model.utils import revert_sync_batchnorm, convert_sync_batchnorm
from mmengine.device.utils import get_device
from mmengine.model.wrappers.distributed import MMDistributedDataParallel,detect_anomalous_params
from mmengine.registry import MODEL_WRAPPERS
from mmengine.optim import OptimWrapper
from mmengine.runner import Runner
import collections.abc

@MODELS.register_module()
class MultiModalEncoder(BaseModel):
    def __init__(self,encoders_cfg):
        super().__init__()
        self.encoders=nn.ModuleDict()
        for name,cfg in encoders_cfg.items():
            self.encoders[name]=MODELS.build(cfg)
    def forward(self,
                inputs:dict,
                data_samples: Optional[list] = None,
                mode: str = 'tensor') -> Union[Dict[str, torch.Tensor], list]:
        outputs={}
        for name,encoder in self.encoders.items():
            inputs_data = inputs[name]
            outputs[name]=encoder(inputs_data,data_samples,mode)
            if name in inputs.keys():
                inputs.pop(name)
        outputs.update(inputs)
        return outputs