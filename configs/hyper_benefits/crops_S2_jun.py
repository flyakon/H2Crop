
custom_imports = dict(imports=['H2Crop'], allow_failed_imports=False)

img_size=192
input_seq_len=6
input_seq_step=1
num_frames=6
with_priors=False
swin_in_embed_dim=128
embed_dim=1024
with_hyper=False
head_embed_dim=256 if with_hyper else 128
hyper_img_size=64
cascade_levels=('level1', 'level2', 'level3', 'level4')
num_classes=[6, 36, 82, 101]
pretrained_weights_path='../checkpoints/AgriFM.pth'
experiment_name='crops_S2_jun'
work_dir=r'../checkpoints/hyper_benefits/%s'%experiment_name

data_toot_path=r'../dataset/h5_data'
data_list_path=r'../dataset/data_list'

s2_mean=[4259.259273982915, 4215.908118624016, 4103.170198423672, 4499.330633142822, 5117.0459378072765,
         5324.110306431293, 5487.406745483038, 5417.512500384096, 3758.2721930309735, 3095.500449391593]
s2_std=[3566.4813736634896, 3230.3569801330054, 3133.285130121181, 3083.2344473492067, 2603.625628756515,
        2453.0269585831725, 2512.6447758166305, 2317.5420847623095, 1496.7727140001148, 1391.1260145456185]
EnMAP_mean=[309.5421, 301.3388, 318.842, 334.9932, 342.8996, 348.9319, 364.7999, 380.8993, 386.9393, 401.2269, 408.6302, 409.1955, 412.0323, 423.3943, 432.9525, 429.9166, 434.8373, 451.6289, 458.9493, 483.6162, 524.4794, 558.3706, 596.7026, 640.7237, 676.9993, 708.0853, 726.0434, 743.1663, 755.3532, 756.726, 748.9131, 730.8229, 697.7578, 695.1724, 691.0783, 690.2051, 694.713, 696.9683, 693.9366, 687.0369, 696.9279, 696.1576, 693.5495, 695.2412, 686.9135, 672.724, 667.4727, 661.8077, 660.3919, 672.5941, 712.5789, 812.3619, 1003.4266, 1241.4645, 1513.9624, 1799.9665, 2163.9276, 2502.3224, 2767.3901, 2919.408, 3077.3673, 2953.6917, 3195.6596, 3247.5064, 3273.5258, 3317.226, 3346.6789, 3405.3652, 3409.1431, 3461.9136, 3453.1753, 3501.1911, 3525.9744, 3546.8471, 3595.0703, 3595.9446, 3637.8386, 3629.5695, 3654.1613, 3680.3304, 3603.4099, 3566.0021, 3617.9968, 3400.8533, 3740.1326, 3709.5692, 3503.325, 3890.0571, 3725.8914, 3828.7373, 3564.7482, 3559.718, 3376.4558, 3514.1424, 3404.8935, 3583.8074, 3422.3933, 3621.3278, 3449.1135, 3496.4183, 3649.4688, 3721.7977, 3778.9596, 3845.9569, 3907.9693, 3950.5905, 3996.8491, 4037.7712, 4043.9568, 4049.3539, 4031.3043, 3871.7487, 3971.7644, 3748.171, 3357.5127, 3398.518, 3404.5146, 3411.1719, 3416.8007, 3433.0039, 3503.1913, 3578.394, 3629.4299, 3657.7558, 3697.7477, 3699.2401, 3664.1217, 3582.7429, 3477.1352, 1537.2348, 1545.1406, 1496.9523, 1547.0122, 1627.3118, 1720.7775, 1793.9441, 1862.6197, 1933.9718, 1994.3452, 2054.3132, 2117.2195, 2171.5675, 2235.2068, 2279.2108, 2317.5802, 2344.9017, 2373.0457, 2368.521, 2371.5362, 2342.4701, 2312.1604, 2282.1807, 2240.9143, 2201.5585, 2178.6143, 2171.5755, 2128.4321, 2209.8675, 1032.9517, 901.0521, 919.8255, 962.0414, 1017.1136, 969.2139, 910.1429, 951.6594, 1061.2858, 1113.0021, 1129.5889, 1083.9881, 1090.0604, 1118.8542, 1164.1069, 1168.2034, 1174.6866, 1199.2, 1215.3345, 1222.6234, 1242.4003, 1257.5217, 1262.8296, 1278.5319, 1286.1492, 1298.2014, 1302.0483, 1320.0263, 1314.6717, 1336.7314, 1341.0787, 1341.7823, 1323.7789, 1286.7988, 1244.6713, 1208.2193, 1165.8874, 1136.1599, 1115.6227, 1083.6456, 1050.1768, 1027.748, 997.8804, 999.716, 982.1604, 951.4429, 928.9946, 934.0334, 942.9222, 934.525, 924.3493, 915.6241, 904.5462, 890.6365, 874.7927, 851.9528, 848.8834, 842.2697, 841.0924, 820.2674]
EnMAP_std=[270.467, 272.1957, 277.3872, 283.95, 290.7022, 294.2421, 299.2277, 301.2805, 303.7201, 308.0447, 310.0696, 313.4797, 317.4772, 321.7248, 325.5506, 331.2423, 334.8835, 339.4727, 343.4966, 347.4992, 350.6359, 351.7562, 350.7576, 352.0583, 354.2478, 358.7686, 363.9742, 368.8117, 375.6676, 385.3448, 398.4585, 411.1396, 423.7397, 438.7383, 446.3431, 455.9439, 464.0364, 472.3879, 480.2796, 488.6939, 497.7553, 502.8122, 510.5936, 521.1313, 530.1831, 536.3987, 543.0293, 551.8149, 557.6057, 561.1418, 569.3059, 565.1671, 546.4452, 515.5008, 495.2985, 478.9727, 520.1152, 597.4918, 677.5248, 724.5753, 764.3551, 809.4894, 842.6053, 839.8281, 840.7008, 853.8493, 857.014, 860.9115, 889.8873, 901.2371, 892.7959, 895.1464, 899.2835, 905.7071, 914.2687, 903.978, 914.5142, 917.6921, 925.5984, 894.6436, 901.5631, 876.107, 882.6381, 822.738, 903.1283, 873.7764, 843.2229, 972.241, 858.4551, 944.2777, 818.1695, 871.4923, 811.3318, 809.5494, 816.4685, 829.6249, 816.5637, 837.6869, 818.7791, 811.5273, 860.0646, 862.1002, 874.5718, 889.9693, 904.6517, 913.7315, 923.0637, 929.8991, 927.805, 924.827, 914.6831, 868.5619, 863.8024, 801.1499, 728.0821, 753.7905, 761.8743, 767.2579, 771.5753, 774.0008, 785.8573, 798.1175, 806.2362, 807.511, 808.7266, 813.7729, 808.0963, 795.6484, 782.1625, 805.1094, 811.8097, 736.8203, 730.9914, 737.7967, 746.9563, 746.1925, 744.0245, 744.3937, 741.6858, 740.8848, 743.7291, 745.7828, 752.4778, 755.0915, 757.6327, 758.2792, 763.4531, 760.9976, 760.1705, 752.8117, 748.865, 747.6889, 743.6157, 738.8399, 739.498, 749.2507, 747.2783, 2746.9931, 926.6904, 708.7551, 692.2299, 717.0709, 747.0154, 697.231, 631.4213, 643.3273, 706.5813, 731.4724, 726.4069, 684.3617, 674.8388, 678.6149, 694.6322, 684.8869, 676.9848, 680.368, 678.088, 674.1267, 675.2698, 674.4752, 667.6647, 666.3871, 660.5991, 657.9247, 650.9512, 648.7453, 636.4705, 639.2854, 640.0761, 641.7993, 636.4189, 624.4613, 611.8678, 603.2808, 591.9094, 586.308, 582.1314, 570.4978, 561.0749, 559.2888, 548.716, 549.7297, 545.1534, 533.4571, 528.25, 541.5009, 557.4598, 560.8546, 563.8145, 565.1073, 569.8991, 570.2249, 564.2724, 552.1226, 565.8726, 570.949, 581.0677, 567.042]
mean={'S2':s2_mean,'EnMAP':EnMAP_mean}
std={'S2':s2_std,'EnMAP':EnMAP_std}

train_pipelines= [
    dict(type='MapVerticalFlip', p=0.5),
    dict(type='MapHorizontalFlip', p=0.5),
    dict(type='MapRotate',interpolation={'ref_crop_data':0}),
    dict(type='MapNormalize',mean=mean,std=std,norm_keys=['S2','EnMAP']),
    dict(type='MapResize',size={'S2':(img_size,img_size),
                                'EnMAP':(hyper_img_size,hyper_img_size),
                                'priors':(img_size,img_size),
                                'label':(img_size,img_size)},interpolation={'priors':0}),
]

valid_pipelines= [
    dict(type='MapNormalize',mean=mean,std=std,norm_keys=['S2','EnMAP']),
    dict(type='MapResize',
         size={'S2': (img_size, img_size),
                'EnMAP':(hyper_img_size,hyper_img_size),
                'priors':(img_size,img_size),
               'label':(img_size,img_size)},interpolation={'priors':0}),
]




default_scope = 'mmseg'
env_cfg = dict(
    cudnn_benchmark=True,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)
log_level = 'INFO'
tta_model = dict(type='SegTTAModel')


model_wrapper_cfg=dict(
    detect_anomalous_params=False,
    find_unused_parameters=True,
)

train_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='H2CropDataset',
        data_toot_path=data_toot_path,
        data_list_file=data_list_path+'/train.txt',
        data_pipelines=train_pipelines,
        with_priors=with_priors,
        cascade_levels=cascade_levels,
        num_classes=num_classes,
        sample_step=1,
        input_seq_step=input_seq_step,
        input_seq_len=input_seq_len,
        num_frames=num_frames,
        with_hyper=with_hyper,
    ))

val_dataloader = dict(
    batch_size=4,
    num_workers=4,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='H2CropDataset',
        data_toot_path=data_toot_path,
        data_list_file=data_list_path+'/val.txt',
        data_pipelines=valid_pipelines,
        with_priors=with_priors,
        num_classes=num_classes,
        cascade_levels=cascade_levels,
        sample_step=1,
        input_seq_step=input_seq_step,
        input_seq_len=input_seq_len,
        num_frames=num_frames,
        with_hyper=with_hyper,
    ))

test_dataloader = dict(
    batch_size=8,
    num_workers=6,
    persistent_workers=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    collate_fn=dict(type='default_collate'),
    dataset=dict(
        type='H2CropDataset',
        data_toot_path=data_toot_path,
        data_list_file=data_list_path+'/test.txt',
        data_pipelines=valid_pipelines,
        with_priors=with_priors,
        num_classes=num_classes,
        cascade_levels=cascade_levels,
        sample_step=1,
        input_seq_step=input_seq_step,
        input_seq_len=input_seq_len,
        num_frames=num_frames,
        with_hyper=with_hyper,
    ))


model=dict(
    type='MultiUnifiedModel',
    batch_augmentation=dict(type='CropCutup',cutmix_min_ratio=0.5,cutmix_max_ratio=1,p=0.5),
    encoders=dict(
        type='MultiModalEncoder',
        encoders_cfg=dict(
            S2=dict(type='PretrainingSwinTransformer3DEncoder',
                    patch_emd_cfg=dict(
                        type='SwinPatchEmbed3D',
                        patch_size=(2, 4, 4),
                        in_chans=10,
                        embed_dim=swin_in_embed_dim,
                    ),
                    backbone_cfg=dict(
                        type='SwinTransformer3D',
                        pretrained=None,
                        pretrained2d=False,
                        patch_size=(2, 4, 4),
                        embed_dim=swin_in_embed_dim,
                        depths=[2, 2, 18, 2],
                        num_heads=[4, 8, 16, 32],
                        window_size=(8, 7, 7),
                        out_indices=(0, 1, 2, 3),
                        mlp_ratio=4.,
                        qkv_bias=True,
                        qk_scale=None,
                        drop_rate=0.,
                        attn_drop_rate=0.,
                        drop_path_rate=0.2,
                        patch_norm=False,
                        frozen_stages=-1,
                        use_checkpoint=False,
                        downsample_steps=((2, 2, 2), (2, 2, 2), (1, 2, 2), (2, 2, 2)),
                        feature_fusion='cat', mean_frame_down=True
                    ),
                    init_cfg=dict(type='pretrained', checkpoint=pretrained_weights_path, )

                ),
        ),
    ),

    neck=dict(
        type='MultiFusionNeck',
        embed_dim=embed_dim,
        in_feature_key=('S2',),
        feature_size=(img_size//32,img_size//32),
        out_size=(img_size,img_size),
        embed_downsample=True,
        in_fusion_key_list=({'S2':512*1,},
                            {'S2':256*2,},
                            {'S2':128*3,},
                            ),
    ),
    head=dict(
        type='CascadeFCNPostCropsHead',
        with_priors=with_priors,
        cascade_head_cfg=dict(
            level1=dict(
                num_classes=6+1,
                type='CropFCNHead',
                embed_dim=head_embed_dim,
                loss_model=dict(type='CropBalanceloss', with_learnable_weight=False,
                                average=True,num_classes=7),
            ),
            level2=dict(
                num_classes=36 + 1,
                type='CropFCNHead',
                embed_dim=head_embed_dim + 7,
                loss_model=dict(type='CropBalanceloss', with_learnable_weight=False, average=True,num_classes=37),
            ),
            level3=dict(
                num_classes=82 + 1,
                type='CropFCNHead',
                embed_dim=head_embed_dim +37,
                loss_model=dict(type='CropBalanceloss', with_learnable_weight=False, average=True,num_classes=83),
            ),
            level4=dict(
                num_classes=101 + 1,
                type='CropFCNHead',
                embed_dim=head_embed_dim +83,
                loss_model=dict(type='CropBalanceloss', with_learnable_weight=False, average=True,num_classes=102),
            ),
        )
    )

)


resume = True
optimizer = dict(type='AdamW', lr=6e-5, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
param_scheduler = [
    dict(type='LinearLR',
         start_factor=0.01,
         by_epoch=False,
         begin=0,
         end=1000),
    dict(
        type='CosineAnnealingLR',
        eta_min=6e-6,
        by_epoch=False,
        begin=1000,
    )
]

log_processor = dict(by_epoch=True)
val_cfg = dict(type='ValLoop')
val_evaluator = dict(type='CropIoUMetric',num_classes=[7,37,83,102],
                     iou_metrics=['mIoU','mFscore'],levels=('level1','level2','level3','level4'))
test_cfg=dict(type='TestLoop')
test_evaluator = dict(type='CropIoUMetric',num_classes=[7,37,83,102],
                     iou_metrics=['mIoU','mFscore'],levels=('level1','level2','level3','level4'),
                      print_per_class=True,save_metric_file=work_dir+'/test_results.csv',)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=100, val_interval=1)


default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=True),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=True, interval=5,max_keep_ckpts=1,
                    save_best='mFscore',rule = 'greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
)