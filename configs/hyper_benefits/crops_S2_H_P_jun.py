
custom_imports = dict(imports=['H2Crop'], allow_failed_imports=False)
_base_=['crops_S2_jun.py',]
img_size=192
input_seq_len=6
num_frames=6
with_priors=True
with_hyper=True
head_embed_dim=256 if with_hyper else 128
hyper_img_size=64
experiment_name='crops_S2_H_P_jun'
work_dir=r'../checkpoints/hyper_benefits/%s'%experiment_name

train_dataloader = dict(

    dataset=dict(
        with_priors=with_priors,
        input_seq_len=input_seq_len,
        num_frames=num_frames,
        with_hyper=with_hyper,
    ))

val_dataloader = dict(
    dataset=dict(
        with_priors=with_priors,
        input_seq_len=input_seq_len,
        num_frames=num_frames,
        with_hyper=with_hyper,
    ))

test_dataloader = dict(
    dataset=dict(
        with_priors=with_priors,
        input_seq_len=input_seq_len,
        num_frames=num_frames,
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
            level1=dict(
                embed_dim=head_embed_dim+7,
            ),
            level2=dict(
                embed_dim=head_embed_dim + 7+37,

            ),
            level3=dict(
                embed_dim=head_embed_dim +37+83,

            ),
            level4=dict(
                embed_dim=head_embed_dim +83+102,
            ),
        )
    )

)

test_evaluator = dict(save_metric_file=work_dir+'/test_results.csv',)
