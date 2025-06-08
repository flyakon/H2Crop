# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
from collections import OrderedDict
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from PIL import Image
from prettytable import PrettyTable
import xlwt
from mmseg.registry import METRICS
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import copy
@METRICS.register_module()
class CropIoUMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 num_classes,
                 ignore_index: int = 255,
                 levels=('level4',),
                 print_per_class=False,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 save_metric_file=None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_classes=num_classes
        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.print_per_class = print_per_class
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only
        self.save_metric_file=save_metric_file
        self.levels=levels

    def process(self, data_batch, data_samples) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        label_map=data_batch[1]
        result_dict=data_samples[0]
        num_classes = self.num_classes
        keys=list(result_dict.keys())
        batch_size=len(result_dict[keys[0]])
        for i in range(batch_size):
            item_dict={}
            for idx,key in enumerate(self.levels):
                pred_label = result_dict[key][i].squeeze()
                pred_label = torch.argmax(pred_label, dim=0)
                label_map[key] = label_map[key].to(pred_label.device)
                if not self.format_only:
                    label = label_map[key][i].squeeze()
                    item=self.intersect_and_union(pred_label, label, num_classes[idx],
                                                  self.ignore_index)
                    item_dict[key]=item
            self.results.append(item_dict)


    def compute_metrics(self, results_dict_list: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        if self.save_metric_file is not None:
            workbook = xlwt.Workbook()

        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        result_dict={}
        for result in results_dict_list:
            for key in result.keys():
                if key not in result_dict:
                    result_dict[key]=[result[key],]
                else:
                    result_dict[key].append(result[key])
        metrics = dict()
        result_to_vis_met=[]
        for idx,level_name in enumerate(result_dict.keys()):
            results=result_dict[level_name]
            results = tuple(zip(*results))
            assert len(results) == 4

            total_area_intersect = sum(results[0])
            total_area_union = sum(results[1])
            total_area_pred_label = sum(results[2])
            total_area_label = sum(results[3])
            ret_metrics = self.total_area_to_metrics(
                total_area_intersect, total_area_union, total_area_pred_label,
                total_area_label, self.metrics, self.nan_to_num, self.beta)

            class_names = np.array(range(self.num_classes[idx]), dtype=np.int64)

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            })

            for key, val in ret_metrics_summary.items():
               if key not in metrics:
                   metrics[key]=[]
               metrics[key].append(val)

            # each class table
            if self.print_per_class or self.save_metric_file is not None:
                ret_metrics.pop('aAcc', None)
                ret_metrics_class = OrderedDict({
                    ret_metric: np.round(ret_metric_value * 100, 2)
                    for ret_metric, ret_metric_value in ret_metrics.items()
                })
                ret_metrics_class.update({'Class': class_names})
                ret_metrics_class.move_to_end('Class', last=False)
                if self.print_per_class:
                    class_table_data = PrettyTable()
                    for key, val in ret_metrics_class.items():
                        class_table_data.add_column(key, val)

                    print_log('%s per class results:'%level_name, logger)
                    print_log('\n' + class_table_data.get_string(), logger=logger)
                    result_to_vis_met.append(ret_metrics_class)
                if self.save_metric_file is not None:
                    sheet = workbook.add_sheet(level_name)
                    for i, (k, val) in enumerate(ret_metrics_class.items()):
                        sheet.write(0, i, k)
                        val=np.nan_to_num(val)
                        for j, v in enumerate(val.tolist()):
                            sheet.write(j + 1, i, v)
        class_table_data = PrettyTable()
        ret_metrics=copy.deepcopy(metrics)
        ret_metrics.update({'Levels': list(result_dict.keys())})
        ret_metrics=OrderedDict(ret_metrics)
        ret_metrics.move_to_end('Levels', last=False)
        for key, val in ret_metrics.items():
            class_table_data.add_column(key, val)

        print_log('per level results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        for key in metrics.keys():
            metrics[key]=np.mean(metrics[key])
        if self.save_metric_file is not None:
            workbook.save(self.save_metric_file)
        return metrics

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """
        h, w = pred_label.shape
        # label=torch.unsqueeze(label.float(),dim=0)
        # label = torch.unsqueeze(label, dim=0)
        # # label = torch.nn.functional.itnerpolate(label, size=(h, w), mode='nearest')
        # label=torch.squeeze(label,dim=0)
        # label = torch.squeeze(label, dim=0)
        mask = (label != ignore_index)
        pred_label = pred_label[mask]


        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1,ignore_index=None):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall+1e-8)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / (total_area_pred_label + 1e-8)
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['mFscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall


        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if ignore_index is not None:
            print(ignore_index)
            print(ret_metrics)
            for key in ret_metrics.keys():
                print(key)
                ret_metrics[key][ignore_index]=np.nan

        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })

        return ret_metrics

@METRICS.register_module()
class CropChangeIoUMetric(BaseMetric):
    """IoU evaluation metric.

    Args:
        ignore_index (int): Index that will be ignored in evaluation.
            Default: 255.
        iou_metrics (list[str] | str): Metrics to be calculated, the options
            includes 'mIoU', 'mDice' and 'mFscore'.
        nan_to_num (int, optional): If specified, NaN values will be replaced
            by the numbers defined by the user. Default: None.
        beta (int): Determines the weight of recall in the combined score.
            Default: 1.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """

    def __init__(self,
                 num_classes,
                 ignore_index: int = 255,
                 levels=('level4',),
                 print_per_class=False,
                 iou_metrics: List[str] = ['mIoU'],
                 nan_to_num: Optional[int] = None,
                 beta: int = 1,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 save_metric_file=None,
                 changed_area=True,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.num_classes=num_classes
        self.ignore_index = ignore_index
        self.metrics = iou_metrics
        self.nan_to_num = nan_to_num
        self.print_per_class = print_per_class
        self.beta = beta
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only
        self.save_metric_file=save_metric_file
        self.levels=levels
        self.changed_area=changed_area

    def process(self, data_batch, data_samples) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """

        label_map=data_batch[1]
        input_data=data_batch[0]
        prior_map=input_data['priors']
        result_dict=data_samples[0]
        num_classes = self.num_classes
        keys=list(result_dict.keys())
        batch_size=len(result_dict[keys[0]])
        for i in range(batch_size):
            item_dict={}
            for idx,key in enumerate(self.levels):
                pred_label = result_dict[key][i].squeeze()

                pred_label = torch.argmax(pred_label, dim=0)
                label_map[key] = label_map[key].to(pred_label.device)
                prior_map[key] = prior_map[key].to(pred_label.device)
                if not self.format_only:
                    label = label_map[key][i].squeeze()
                    prior=prior_map[key][i].squeeze()
                    item=self.intersect_and_union(pred_label, label, num_classes[idx],
                                                  self.ignore_index,prior,self.changed_area)
                    item_dict[key]=item
            self.results.append(item_dict)


    def compute_metrics(self, results_dict_list: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The key
                mainly includes aAcc, mIoU, mAcc, mDice, mFscore, mPrecision,
                mRecall.
        """
        if self.save_metric_file is not None:
            workbook = xlwt.Workbook()

        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()
        # convert list of tuples to tuple of lists, e.g.
        # [(A_1, B_1, C_1, D_1), ...,  (A_n, B_n, C_n, D_n)] to
        # ([A_1, ..., A_n], ..., [D_1, ..., D_n])
        result_dict={}
        for result in results_dict_list:
            for key in result.keys():
                if key not in result_dict:
                    result_dict[key]=[result[key],]
                else:
                    result_dict[key].append(result[key])
        metrics = dict()
        result_to_vis_met=[]
        for idx,level_name in enumerate(result_dict.keys()):
            results=result_dict[level_name]
            results = tuple(zip(*results))
            assert len(results) == 4

            total_area_intersect = sum(results[0])
            total_area_union = sum(results[1])
            total_area_pred_label = sum(results[2])
            total_area_label = sum(results[3])
            ret_metrics = self.total_area_to_metrics(
                total_area_intersect, total_area_union, total_area_pred_label,
                total_area_label, self.metrics, self.nan_to_num, self.beta)

            class_names = np.array(range(self.num_classes[idx]), dtype=np.int64)

            # summary table
            ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            })

            for key, val in ret_metrics_summary.items():
               if key not in metrics:
                   metrics[key]=[]
               metrics[key].append(val)

            # each class table
            if self.print_per_class or self.save_metric_file is not None:
                ret_metrics.pop('aAcc', None)
                ret_metrics_class = OrderedDict({
                    ret_metric: np.round(ret_metric_value * 100, 2)
                    for ret_metric, ret_metric_value in ret_metrics.items()
                })
                ret_metrics_class.update({'Class': class_names})
                ret_metrics_class.move_to_end('Class', last=False)
                if self.print_per_class:
                    class_table_data = PrettyTable()
                    for key, val in ret_metrics_class.items():
                        class_table_data.add_column(key, val)
                    if self.changed_area:
                        print_log('%s per class results in changed area'%level_name, logger)
                    else:
                        print_log('%s per class results in unchanged area'%level_name, logger)
                    print_log('\n' + class_table_data.get_string(), logger=logger)
                    result_to_vis_met.append(ret_metrics_class)
                if self.save_metric_file is not None:
                    sheet = workbook.add_sheet(level_name)
                    for i, (k, val) in enumerate(ret_metrics_class.items()):
                        sheet.write(0, i, k)
                        val=np.nan_to_num(val)
                        for j, v in enumerate(val.tolist()):
                            sheet.write(j + 1, i, v)
        class_table_data = PrettyTable()
        ret_metrics=copy.deepcopy(metrics)
        ret_metrics.update({'Levels': list(result_dict.keys())})
        ret_metrics=OrderedDict(ret_metrics)
        ret_metrics.move_to_end('Levels', last=False)
        for key, val in ret_metrics.items():
            class_table_data.add_column(key, val)

        print_log('per level results:', logger)
        print_log('\n' + class_table_data.get_string(), logger=logger)
        for key in metrics.keys():
            metrics[key]=np.mean(metrics[key])
        if self.save_metric_file is not None:
            workbook.save(self.save_metric_file)
        return metrics

    @staticmethod
    def intersect_and_union(pred_label: torch.tensor, label: torch.tensor,
                            num_classes: int, ignore_index: int,prior_data,changed_area):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """
        h, w = pred_label.shape
        # label=torch.unsqueeze(label.float(),dim=0)
        # label = torch.unsqueeze(label, dim=0)
        # # label = torch.nn.functional.itnerpolate(label, size=(h, w), mode='nearest')
        # label=torch.squeeze(label,dim=0)
        # label = torch.squeeze(label, dim=0)

        mask = (label != ignore_index)
        if changed_area:
            mask = mask & (prior_data !=label)
        else:
            mask = mask & (prior_data == label)
        pred_label = pred_label[mask]


        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(num_classes), min=0,
            max=num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label

    @staticmethod
    def total_area_to_metrics(total_area_intersect: np.ndarray,
                              total_area_union: np.ndarray,
                              total_area_pred_label: np.ndarray,
                              total_area_label: np.ndarray,
                              metrics: List[str] = ['mIoU'],
                              nan_to_num: Optional[int] = None,
                              beta: int = 1,ignore_index=None):
        """Calculate evaluation metrics
        Args:
            total_area_intersect (np.ndarray): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (np.ndarray): The union of prediction and ground
                truth histogram on all classes.
            total_area_pred_label (np.ndarray): The prediction histogram on
                all classes.
            total_area_label (np.ndarray): The ground truth histogram on
                all classes.
            metrics (List[str] | str): Metrics to be evaluated, 'mIoU' and
                'mDice'.
            nan_to_num (int, optional): If specified, NaN values will be
                replaced by the numbers defined by the user. Default: None.
            beta (int): Determines the weight of recall in the combined score.
                Default: 1.
        Returns:
            Dict[str, np.ndarray]: per category evaluation metrics,
                shape (num_classes, ).
        """

        def f_score(precision, recall, beta=1):
            """calculate the f-score value.

            Args:
                precision (float | torch.Tensor): The precision value.
                recall (float | torch.Tensor): The recall value.
                beta (int): Determines the weight of recall in the combined
                    score. Default: 1.

            Returns:
                [torch.tensor]: The f-score value.
            """
            score = (1 + beta**2) * (precision * recall) / (
                (beta**2 * precision) + recall+1e-8)
            return score

        if isinstance(metrics, str):
            metrics = [metrics]
        allowed_metrics = ['mIoU', 'mDice', 'mFscore']
        if not set(metrics).issubset(set(allowed_metrics)):
            raise KeyError(f'metrics {metrics} is not supported')

        all_acc = total_area_intersect.sum() / total_area_label.sum()
        ret_metrics = OrderedDict({'aAcc': all_acc})
        for metric in metrics:
            if metric == 'mIoU':
                iou = total_area_intersect / total_area_union
                acc = total_area_intersect / total_area_label
                ret_metrics['IoU'] = iou
                ret_metrics['Acc'] = acc
            elif metric == 'mDice':
                dice = 2 * total_area_intersect / (
                    total_area_pred_label + total_area_label)
                acc = total_area_intersect / total_area_label
                ret_metrics['Dice'] = dice
                ret_metrics['Acc'] = acc
            elif metric == 'mFscore':
                precision = total_area_intersect / (total_area_pred_label + 1e-8)
                recall = total_area_intersect / total_area_label
                f_value = torch.tensor([
                    f_score(x[0], x[1], beta) for x in zip(precision, recall)
                ])
                ret_metrics['mFscore'] = f_value
                ret_metrics['Precision'] = precision
                ret_metrics['Recall'] = recall


        ret_metrics = {
            metric: value.numpy()
            for metric, value in ret_metrics.items()
        }
        if ignore_index is not None:
            print(ignore_index)
            print(ret_metrics)
            for key in ret_metrics.keys():
                print(key)
                ret_metrics[key][ignore_index]=np.nan

        if nan_to_num is not None:
            ret_metrics = OrderedDict({
                metric: np.nan_to_num(metric_value, nan=nan_to_num)
                for metric, metric_value in ret_metrics.items()
            })

        return ret_metrics







