
custom_imports = dict(imports=['H2Crop'], allow_failed_imports=False)
_base_=['../hyper_benefits/crops_S2_jun.py',]
img_size=192
input_seq_len=8
num_frames=8
with_priors=True
with_hyper=False
head_embed_dim=256 if with_hyper else 128
experiment_name='crops_S2_P_Aug'
work_dir=r'../checkpoints/temporal_exp/%s'%experiment_name

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
    neck=dict(
        in_fusion_key_list=({'S2':512*1,},
                            {'S2':256*2,},
                            {'S2':128*4,},
                            ),
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
                embed_dim=head_embed_dim+83+102,
            ),
        )
    )

)

test_evaluator = dict(save_metric_file=work_dir+'/test_results.csv',)
