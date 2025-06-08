_base_=['../hyper_benefits/crops_S2_jun.py',]
custom_imports = dict(imports=['H2Crop'], allow_failed_imports=False)

img_size=192
with_priors=True
swin_in_embed_dim=128
embed_dim=1024
with_hyper=True
head_embed_dim=256 if with_hyper else 128
hyper_img_size=64
cascade_levels=('level4', )
num_classes=[101, ]
experiment_name='level4_crops_S2_jun'
work_dir=r'../checkpoints/independent_heads/%s'%experiment_name


train_dataloader = dict(
    dataset=dict(
        with_priors=with_priors,
        cascade_levels=cascade_levels,
        num_classes=num_classes,
        with_hyper=with_hyper,
    ))

val_dataloader = dict(
    dataset=dict(
        with_priors=with_priors,
        num_classes=num_classes,
        cascade_levels=cascade_levels,
        with_hyper=with_hyper,
    ))

test_dataloader = dict(
    dataset=dict(
        with_priors=with_priors,
        num_classes=num_classes,
        cascade_levels=cascade_levels,
        with_hyper=with_hyper,
    ))


model=dict(
    encoders=dict(
        type='MultiModalEncoder',
        encoders_cfg=dict(
            EnMAP=dict(
                type='SPVisionTransformer',
                embed_dim=128*2,
                out_embed_dim=128,
                s_vit_cfg=dict(
                    type='SVisionTransformer',
                    img_size=hyper_img_size,
                    patch_size=1,
                    in_channels=218,
                    embed_dim=768,
                    out_embed_dim=128,
                    output_size=img_size,
                    num_heads=12,
                    num_layers=6,
                    mlp_ratio=4,
                    qkv_bias=True,
                    drop_rate=0.,
                    with_cls_token=False,
                ),
                p_vit_cfg=dict(
                    type='PVisionTransformer',
                    img_size=hyper_img_size,
                    patch_size=4,
                    in_channels=218,
                    embed_dim=256,
                    patch_mode='avg',
                    out_embed_dim=128,
                    output_size=img_size,
                    num_heads=8,
                    num_layers=6,
                    mlp_ratio=4,
                    qkv_bias=True,
                    drop_rate=0.,
                    with_cls_token=False,
                ),
            ),
        ),
    ),

    neck=dict(
        hyper_embed_neck=dict(type='HyperEmbedNeck', in_channels=128,
                                   out_channels=128,
                                   in_keys=['EnMAP', ],
                                   auxiliary_keys=None),
    ),
    head=dict(
        with_priors=with_priors,
        cascade_head_cfg=dict(
            _delete_=True,
            level4=dict(
                num_classes=101+1,
                type='CropFCNHead',
                embed_dim=head_embed_dim+102,
                loss_model=dict(type='CropBalanceloss', with_learnable_weight=False,
                                average=True,num_classes=102),
            ),
        )
    )

)


test_evaluator = dict(num_classes=[102,],
                     levels=('level4',),
                      save_metric_file=work_dir+'/test_results.csv',)
