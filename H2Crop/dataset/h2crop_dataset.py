
import pickle
import torch
import os
from mmseg.registry.registry import MODELS,DATASETS
import numpy as np
import copy
import h5py
from H2Crop.dataset.transform import MapCompose


@DATASETS.register_module()
class H2CropDataset(torch.utils.data.Dataset):
    def __init__(self, data_toot_path, data_list_file,
                 data_pipelines, cascade_levels=('level4',),
                 num_classes=(150,),
                 with_priors=True,sample_step=1,with_hyper=False,
                 input_seq_len=36,input_seq_step=1,num_frames=36,with_file_path=False):

        self.data_root_path = data_toot_path
        self.data_list_file = data_list_file
        self.data_pipelines = data_pipelines
        self.cascade_level_idx = []
        assert len(cascade_levels)==len(num_classes),'cascade levels and num classes should have the same length'
        self.num_classes=num_classes
        available_levels = ['level1', 'level2', 'level3', 'level4']
        for level in cascade_levels:
            assert level in available_levels, 'level %s is not available' % level
            self.cascade_level_idx.append(available_levels.index(level))
        self.with_priors = with_priors
        self.cascade_levels = cascade_levels
        self.with_hyper=with_hyper
        self.with_file_path=with_file_path

        self.sample_step = sample_step

        self.data_pipelines = MapCompose(self.data_pipelines)

        self.crop_tile_names=np.loadtxt(self.data_list_file,dtype=str).tolist()
        if self.sample_step>1:
            self.crop_tile_names=self.crop_tile_names[::self.sample_step]
        self.data_len = len(self.crop_tile_names)
        self.input_seq_len=input_seq_len
        self.input_seq_step=input_seq_step
        self.num_frames=num_frames



    def __len__(self):
        return self.data_len

    def get_data(self, crop_tile_name):

        data_file=os.path.join(self.data_root_path,'%s.h5'%crop_tile_name)

        h5_dataset=h5py.File(data_file,'r')
        S2_data=h5_dataset['S2_data'][0:self.input_seq_len:self.input_seq_step]
        label_data=h5_dataset['label'][:]
        label_data = np.stack([label_data[i] for i in self.cascade_level_idx], axis=0)
        item_dict = {}
        assert len(S2_data)==self.num_frames
        item_dict['S2'] = torch.from_numpy(S2_data).float()
        label_data = torch.from_numpy(label_data).long()
        if self.with_priors:
            priors=h5_dataset['priors'][:]
            priors = np.stack([priors[i] for i in self.cascade_level_idx], axis=0)
            item_dict['priors'] = torch.from_numpy(priors).long()

        if self.with_hyper:
            hyper_data=h5_dataset['EnMAP_data'][:]
            item_dict['EnMAP']=torch.from_numpy(hyper_data).float()
        return item_dict,label_data


    def __getitem__(self, idx):
        crop_tile_name = self.crop_tile_names[idx]
        item_dict,label_list = self.get_data(crop_tile_name)
        item_dict,label_list = self.data_pipelines(item_dict, label_list)
        label_dict={}
        for i,level in enumerate(self.cascade_levels):
            label_dict[level]=label_list[i].long()
        if self.with_priors:
            ref_crop_data = item_dict['priors']
            item_dict['priors']={}
            for i,level in enumerate(self.cascade_levels):
                ref_crop_data_level=ref_crop_data[i].long()
                item_dict['priors'][level]=ref_crop_data_level
        if self.with_file_path:
            item_dict['file_path']=crop_tile_name
        return item_dict,label_dict








