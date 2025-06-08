# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import cv2
import mmcv
import mmengine.fileio as fileio
import numpy as np
import torch
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmengine.runner.runner import MMDistributedDataParallel
from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample
from mmseg.visualization import SegLocalVisualizer
from H2Crop.visualization.local_visualizer import (CropSegLocalVisualizer)
from H2Crop.dataset.transform import MapCompose
@HOOKS.register_module()
class CropSegVisualizationHook(Hook):
    """Segmentation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of hooks. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 vis_key,
                 img_size=(256,256),
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self._visualizer: CropSegLocalVisualizer = \
            CropSegLocalVisualizer.get_current_instance()
        self.interval = interval
        self.img_size=img_size
        self.show = show
        self.vis_key=vis_key
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for hooks will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')

    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch,
                    outputs,
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """
        if self.draw is False:
            return

        if self.every_n_inner_iters(batch_idx, self.interval):
            if mode=='val':
                input_imgs, mask_label_list = data_batch
                mask_label_list = mask_label_list.detach().cpu().numpy()
                # mask_label_list = np.argmax(mask_label_list, axis=1)
                input_imgs = input_imgs[self.vis_key].detach()

            else:
                input_imgs, mask_label_list = data_batch
                mask_label_list = mask_label_list.detach().cpu().numpy()
                # mask_label_list = np.argmax(mask_label_list, axis=1)
                input_imgs = input_imgs[self.vis_key].detach()
                try:
                    outputs = runner.model.result_list
                except:
                    outputs = runner.model.module.result_list
            for i, output in enumerate(outputs):
                output=torch.argmax(output,dim=0)
                img=input_imgs[i,0].cpu().numpy()
                img=img[[0,1,2]]
                img_min = np.percentile(img, q=2)
                img_max = np.percentile(img, q=98)
                img = (img - img_min) / (img_max - img_min + 1e-6)
                img = np.clip(img, 0, 1)
                img = img * 255
                img = img.astype(np.uint8)
                img=np.transpose(img,(1,2,0))
                img=cv2.resize(img,self.img_size)
                data_samples={'gt_sem_seg':cv2.resize(mask_label_list[i].astype(np.uint8),self.img_size,interpolation=cv2.INTER_NEAREST),
                              'pred_sem_seg':cv2.resize(output.detach().cpu().numpy().astype(np.uint8),self.img_size,interpolation=cv2.INTER_NEAREST)}

                window_name = '%s'%(mode)
                classes=list(range(0,53))
                classes=[str(x) for x in classes]
                self._visualizer.add_datasample(
                    window_name,
                    img,
                    classes,
                    data_sample=data_samples,
                    show=self.show,
                    wait_time=self.wait_time,
                    step=runner.iter)
                break



@HOOKS.register_module()
class CascadeCropSegVisualizationHook(Hook):
    """Segmentation Visualization Hook. Used to visualize validation and
    testing process prediction results.

    In the testing phase:

    1. If ``show`` is True, it means that only the prediction results are
        visualized without storing data, so ``vis_backends`` needs to
        be excluded.

    Args:
        draw (bool): whether to draw prediction results. If it is False,
            it means that no drawing will be done. Defaults to False.
        interval (int): The interval of hooks. Defaults to 50.
        show (bool): Whether to display the drawn image. Default to False.
        wait_time (float): The interval of show (s). Defaults to 0.
        backend_args (dict, Optional): Arguments to instantiate a file backend.
            See https://mmengine.readthedocs.io/en/latest/api/fileio.htm
            for details. Defaults to None.
            Notes: mmcv>=2.0.0rc4, mmengine>=0.2.0 required.
    """

    def __init__(self,
                 normalize_pipeline,
                 img_size=(128,128),
                 draw: bool = False,
                 interval: int = 50,
                 show: bool = False,
                 wait_time: float = 0.,
                 backend_args: Optional[dict] = None):
        self._visualizer: CropSegLocalVisualizer = \
            CropSegLocalVisualizer.get_current_instance()
        self.interval = interval
        self.img_size=img_size
        self.show = show
        if self.show:
            # No need to think about vis backends.
            self._visualizer._vis_backends = {}
            warnings.warn('The show is True, it means that only '
                          'the prediction results are visualized '
                          'without storing data, so vis_backends '
                          'needs to be excluded.')

        self.wait_time = wait_time
        self.backend_args = backend_args.copy() if backend_args else None
        self.draw = draw
        self.pipeline = MapCompose(normalize_pipeline)
        if not self.draw:
            warnings.warn('The draw is False, it means that the '
                          'hook for hooks will not take '
                          'effect. The results will NOT be '
                          'visualized or stored.')

    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch,
                    outputs,
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """
        if self.draw is False:
            return

        if self.every_n_inner_iters(batch_idx, self.interval):
            if mode=='val':
                input_imgs, data_samples = data_batch
                # mask_label_list = np.argmax(mask_label_list, axis=1)
                if isinstance(input_imgs, tuple) or isinstance(input_imgs, list):
                    input_imgs = input_imgs[0]
                if isinstance(data_samples, tuple) or isinstance(data_samples, list):
                    data_samples = data_samples[0]
                input_imgs = input_imgs.detach()
                input_imgs = self.pipeline.reverse(input_imgs)[0]
                outputs = outputs[0]
            else:
                input_imgs, data_samples = data_batch
                if isinstance(input_imgs, tuple) or isinstance(input_imgs, list):
                    input_imgs = input_imgs[0]
                if isinstance(data_samples, tuple) or isinstance(data_samples, list):
                    data_samples = data_samples[0]
                # mask_label_list = np.argmax(mask_label_list, axis=1)
                input_imgs = input_imgs.detach()
                input_imgs = self.pipeline.reverse(input_imgs)[0]
                try:
                    outputs=runner.model.result_dict
                except:
                    outputs=runner.model.module.result_dict
            for i in range(len(input_imgs)):

                img=input_imgs[i,0].cpu().numpy()
                img=img[[1,2,3]]
                img_min = np.percentile(img, q=2)
                img_max = np.percentile(img, q=98)
                img = (img - img_min) / (img_max - img_min + 1e-6)
                img = np.clip(img, 0, 1)
                img = img * 255
                img = img.astype(np.uint8)
                img=np.transpose(img,(1,2,0))
                img=cv2.resize(img,self.img_size)
                # output = torch.argmax(output, dim=0)
                for key in outputs.keys():
                    # outputs[key]=outputs[key].detach().cpu().numpy()
                    vis_data_samples={'gt_sem_seg':cv2.resize(data_samples[key][i].cpu().numpy().astype(np.uint8),self.img_size,interpolation=cv2.INTER_NEAREST),
                                  'pred_sem_seg':cv2.resize(outputs[key][i].detach().cpu().numpy().astype(np.uint8),self.img_size,interpolation=cv2.INTER_NEAREST)}

                    window_name = '%s_%s'%(mode,key)
                    classes=list(range(0,53))
                    classes=[str(x) for x in classes]
                    self._visualizer.add_datasample(
                        window_name,
                        img,
                        classes,
                        data_sample=vis_data_samples,
                        show=self.show,
                        wait_time=self.wait_time,
                        step=runner.iter)
                break