_base_ = ('../../third_party/mmyolo/configs/yolov8/'
          'yolov8_s_syncbn_fast_8xb16-500e_coco.py')
custom_imports = dict(imports=['yolo_world'],
                      allow_failed_imports=False)

# hyper-parameters
num_classes = 1203
num_training_classes = 80
max_epochs = 100  # Maximum training epochs
close_mosaic_epochs = 2
save_epoch_intervals = 2
text_channels = 512
neck_embed_channels = [128, 256, _base_.last_stage_out_channels // 2]
base_lr = 2e-3
weight_decay = 0.05 / 2
train_batch_size_per_gpu = 16 # total_batch_size=512, so we need to use accumulative_counts=4

val_batch_size_per_gpu = 32

# model settings
model = dict(
    type='MyYOLOWorldDetector',
    mm_neck=True, # apply img-text fusion or not
    num_train_classes=num_training_classes,
    num_test_classes=num_classes,
    data_preprocessor=dict(type='YOLOWDetDataPreprocessor'),

    backbone=dict(
        _delete_=True,
        type='MultiModalYOLOBackbone',
        image_model={{_base_.model.backbone}},
        text_model=dict(
            type='HuggingCLIPLanguageBackbone',
            model_name='openai/clip-vit-base-patch32',
            frozen_modules=['all']
        )
    ),

    neck=dict(
        type='YOLOWorldVLFusionNeck',
        fusion_cfg = dict(
                type='MaxSigmoidAttnFusionBLock', # TODO: change this block and experiment a lot!
                text_dimension = text_channels,
                num_heads=8, # this should be divisible by last_img_channel=512 and text_dimension=512
            ),
        in_channels=[256, 512, _base_.last_stage_out_channels],
        out_channels=[256, 512, _base_.last_stage_out_channels],
        deepen_factor=_base_.deepen_factor,
        widen_factor=_base_.widen_factor,
    ),

    bbox_head=dict(
        type='YOLOWorldHead',
        head_module=dict(
            type='YOLOWorldHeadModule',
            embed_dims=text_channels,
            num_classes=num_training_classes,
            widen_factor=_base_.widen_factor,
        )
    ),

    train_cfg=dict(assigner=dict(num_classes=num_training_classes))
)

# dataset settings
text_transform = [
    dict(type='RandomLoadText',
         num_neg_samples=(num_classes, num_classes),
         max_num_samples=num_training_classes,
         padding_to_max=True,
         padding_value=''),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                    'flip_direction', 'texts'))
]
train_pipeline = [
    *_base_.pre_transform,
    dict(type='MultiModalMosaic',
         img_scale=_base_.img_scale,
         pad_val=114.0,
         pre_transform=_base_.pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - _base_.affine_scale, 1 + _base_.affine_scale),
        max_aspect_ratio=_base_.max_aspect_ratio,
        border=(-_base_.img_scale[0] // 2, -_base_.img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *_base_.last_transform[:-1],
    *text_transform,
]
train_pipeline_stage2 = [*_base_.train_pipeline_stage2[:-1], *text_transform]


obj365v1_train_dataset = dict(
    type='MultiModalDataset',
    dataset=dict(
        type='YOLOv5Objects365V1Dataset',
        data_root='data/objects365v1/',
        ann_file='annotations/objects365_train.json',
        data_prefix=dict(img='train/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32)),
    class_text_path='data/texts/obj365v1_class_texts.json',
    pipeline=train_pipeline)


mg_train_dataset = dict(type='YOLOv5MixedGroundingDataset',
                        data_root='data/mixed_grounding/',
                        ann_file='annotations/final_mixed_train_no_coco.json',
                        data_prefix=dict(img='gqa/images/'),
                        filter_cfg=dict(filter_empty_gt=False, min_size=32),
                        pipeline=train_pipeline)


flickr_train_dataset = dict(
    type='YOLOv5MixedGroundingDataset',
    data_root='data/flickr/',
    ann_file='annotations/final_flickr_separateGT_train.json',
    data_prefix=dict(img='full_images/'),
    filter_cfg=dict(filter_empty_gt=True, min_size=32),
    pipeline=train_pipeline)


# NOTE: trained with [objects365v1 + gqa + flickr30k + coco]
train_dataloader = dict(batch_size=train_batch_size_per_gpu,
                        collate_fn=dict(type='yolow_collate'),
                        dataset=dict(_delete_=True,
                                     type='ConcatDataset',
                                     datasets=[
                                         obj365v1_train_dataset,
                                         # flickr_train_dataset,
                                         # mg_train_dataset,
                                     ],
                                     ignore_keys=['classes', 'palette']))

test_pipeline = [
    *_base_.test_pipeline[:-1],
    dict(type='LoadText'),
    dict(type='mmdet.PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                    'scale_factor', 'pad_param', 'texts'))
]
coco_val_dataset = dict(
    _delete_=True,
    type='MultiModalDataset',
    dataset=dict(type='YOLOv5LVISV1Dataset',
                 data_root='data/coco/',
                 test_mode=True,
                 ann_file='lvis/lvis_v1_minival_inserted_image_name.json',
                 data_prefix=dict(img=''),
                 batch_shapes_cfg=None),
    class_text_path='data/texts/lvis_v1_class_texts.json',
    pipeline=test_pipeline)
val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    dataset=coco_val_dataset,
)
test_dataloader = val_dataloader

val_evaluator = dict(type='mmdet.LVISMetric',
                     ann_file='data/coco/lvis/lvis_v1_minival_inserted_image_name.json',
                     metric='bbox')
test_evaluator = val_evaluator

# training settings
default_hooks = dict(
    param_scheduler=dict(max_epochs=max_epochs),
    logger=dict(type='LoggerHook', interval=1), # TODO: This is for testing
    checkpoint=dict(
        interval=save_epoch_intervals,
        rule='greater'
    ),
)

custom_hooks = [
    dict(
        type='EMAHook',
        ema_type='ExpMomentumEMA',
         momentum=0.0001,
         update_buffers=True,
         strict_load=False,
         priority=49
    ),
    dict(
        type='mmdet.PipelineSwitchHook',
        switch_epoch=max_epochs - close_mosaic_epochs,
        switch_pipeline=train_pipeline_stage2
    ),
    dict(type='CheckProcessTimePerBlock', interval=50), # to check process time per block
]

train_cfg = dict(
    max_epochs=max_epochs,
    val_interval=10,
    dynamic_intervals=[(
        (max_epochs - close_mosaic_epochs),
        _base_.val_interval_stage2),
    ])

optim_wrapper = dict(
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=base_lr,
        weight_decay=weight_decay,
        batch_size_per_gpu=train_batch_size_per_gpu
    ),
    paramwise_cfg=dict(
        bias_decay_mult=0.0,
        norm_decay_mult=0.0,
        custom_keys={'backbone.text_model': dict(lr_mult=0.01), 'logit_scale': dict(weight_decay=0.0)}
    ),
    accumulative_counts=4, # FIXME: in case of 8 GPUs --------------------------------------------------------------
    constructor='YOLOWv5OptimizerConstructor'
)

# log_config = dict(hooks = [
#     dict(type='TextLoggerHook'),
#     dict(type='MMDetWandbHook',
#          init_kwargs={
#             'project': 'YOLO-World',
#             'entity': 'yolo_world_s_dual_vlpan_l2norm_2e-3_100e_8gpus_obj365v1_train_lvis_minival',
#          },
#          interval=10,
#          log_checkpoint=True,
#          log_checkpoint_metadata=True,
#          num_eval_images=100)])

# 🐝 use wandb for tracking log
# visualizer = dict(vis_backends=[dict(type='LocalVisBackend'), dict(type='WandbVisBackend')])




