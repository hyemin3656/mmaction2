_base_ = '../../_base_/default_runtime.py'

NUM_CLASSES = 67
NUM_JOINTS = 65
CLIP_LEN = 100

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='SkeletonCNN1D',
        in_channels=2,
        num_joints=NUM_JOINTS,
        num_person=1,
        hidden_channels=(64, 128, 64),
        kernel_size=3,
        pool_kernel_size=2,
        dropout=0.1,
        data_bn=True),
    cls_head=dict(
        type='GCNHead',
        num_classes=NUM_CLASSES,
        in_channels=64,
        dropout=0.5))

dataset_type = 'PoseDataset'
ann_file = '../dataset/cropped_holistic_results_interpolated_remapped_direct/mediapipe_sign_3d_without_face_pose_score_1.pkl'

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='mediapipe-sign-3d',
            name='cnn1d_8xb16-joint-u100-50e_mediapipe-sign-keypoint-3d_without_face',
            config=dict(num_classes=NUM_CLASSES)),
        define_metric_cfg=[
            dict(name='epoch'),
            dict(name='step'),
            dict(name='lr', step_metric='epoch'),
            dict(name='loss', step_metric='epoch'),
            dict(name='loss_cls', step_metric='epoch'),
            dict(name='top1_acc', step_metric='epoch'),
            dict(name='top5_acc', step_metric='epoch'),
            dict(name='acc/*', step_metric='epoch'),
        ])
]
visualizer = dict(type='ActionVisualizer', vis_backends=vis_backends)

train_pipeline = [
    # dict(type='CenterNormalize2D'),
    # dict(type='BodyCenterNormalize2D', scale=True),
    dict(type='GenSkeFeat', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=CLIP_LEN),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

val_pipeline = [
    # dict(type='CenterNormalize2D'),
    # dict(type='BodyCenterNormalize2D', scale=True),
    dict(type='GenSkeFeat', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=CLIP_LEN, num_clips=5, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

test_pipeline = [
    # dict(type='CenterNormalize2D'),
    # dict(type='BodyCenterNormalize2D', scale=True),
    dict(type='GenSkeFeat', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=CLIP_LEN, num_clips=5, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

train_dataloader = dict(
    batch_size=16,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        type='RepeatDataset',
        times=5,
        dataset=dict(
            type=dataset_type,
            ann_file=ann_file,
            pipeline=train_pipeline,
            split='train')))
val_dataloader = dict(
    batch_size=16,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        split='val',
        test_mode=True))
test_dataloader = dict(
    batch_size=1,
    num_workers=1,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        split='test',
        test_mode=True))

val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=5, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=50,
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(
        type='AdamW', lr=0.001, weight_decay=0.01))
