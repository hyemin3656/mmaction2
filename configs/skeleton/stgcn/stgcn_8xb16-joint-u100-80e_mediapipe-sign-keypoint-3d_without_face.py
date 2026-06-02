_base_ = '../../_base_/default_runtime.py'

NUM_CLASSES = 67

POSE_OFFSET = 0
FACE_OFFSET = 23
LHAND_OFFSET = 23 #23 + 468      # 491
RHAND_OFFSET = 23 + 21 #23 + 468 + 21 # 512

def shift(edges, offset):
    return [(a + offset, b + offset) for a, b in edges]

mediapipe_sign_layout = dict(
    num_node=65,  # 533 (pose 23 + face 468 + left hand 21 + right hand 21)

    inward=[
        # -------------------------
        # Pose 0~22
        # -------------------------
        (2, 0), (7, 2),
        (5, 0), (8, 5),
        (9, 0), (10, 0),

        (11, 0), (12, 0),
        (13, 11), (14, 12),
        (15, 13), (16, 14),

        (17, 15), (19, 15), (21, 15),
        (18, 16), (20, 16), (22, 16),

        # -------------------------
        # Left hand 491~511
        # -------------------------
        (LHAND_OFFSET + 0, 15),

        (LHAND_OFFSET + 1, LHAND_OFFSET + 0),
        (LHAND_OFFSET + 2, LHAND_OFFSET + 1),
        (LHAND_OFFSET + 3, LHAND_OFFSET + 2),
        (LHAND_OFFSET + 4, LHAND_OFFSET + 3),

        (LHAND_OFFSET + 5, LHAND_OFFSET + 0),
        (LHAND_OFFSET + 6, LHAND_OFFSET + 5),
        (LHAND_OFFSET + 7, LHAND_OFFSET + 6),
        (LHAND_OFFSET + 8, LHAND_OFFSET + 7),

        (LHAND_OFFSET + 9, LHAND_OFFSET + 0),
        (LHAND_OFFSET + 10, LHAND_OFFSET + 9),
        (LHAND_OFFSET + 11, LHAND_OFFSET + 10),
        (LHAND_OFFSET + 12, LHAND_OFFSET + 11),

        (LHAND_OFFSET + 13, LHAND_OFFSET + 0),
        (LHAND_OFFSET + 14, LHAND_OFFSET + 13),
        (LHAND_OFFSET + 15, LHAND_OFFSET + 14),
        (LHAND_OFFSET + 16, LHAND_OFFSET + 15),

        (LHAND_OFFSET + 17, LHAND_OFFSET + 0),
        (LHAND_OFFSET + 18, LHAND_OFFSET + 17),
        (LHAND_OFFSET + 19, LHAND_OFFSET + 18),
        (LHAND_OFFSET + 20, LHAND_OFFSET + 19),

        (LHAND_OFFSET + 5, LHAND_OFFSET + 1),
        (LHAND_OFFSET + 9, LHAND_OFFSET + 5),
        (LHAND_OFFSET + 13, LHAND_OFFSET + 9),
        (LHAND_OFFSET + 17, LHAND_OFFSET + 13),

        # -------------------------
        # Right hand 512~532
        # -------------------------
        (RHAND_OFFSET + 0, 16),

        (RHAND_OFFSET + 1, RHAND_OFFSET + 0),
        (RHAND_OFFSET + 2, RHAND_OFFSET + 1),
        (RHAND_OFFSET + 3, RHAND_OFFSET + 2),
        (RHAND_OFFSET + 4, RHAND_OFFSET + 3),

        (RHAND_OFFSET + 5, RHAND_OFFSET + 0),
        (RHAND_OFFSET + 6, RHAND_OFFSET + 5),
        (RHAND_OFFSET + 7, RHAND_OFFSET + 6),
        (RHAND_OFFSET + 8, RHAND_OFFSET + 7),

        (RHAND_OFFSET + 9, RHAND_OFFSET + 0),
        (RHAND_OFFSET + 10, RHAND_OFFSET + 9),
        (RHAND_OFFSET + 11, RHAND_OFFSET + 10),
        (RHAND_OFFSET + 12, RHAND_OFFSET + 11),

        (RHAND_OFFSET + 13, RHAND_OFFSET + 0),
        (RHAND_OFFSET + 14, RHAND_OFFSET + 13),
        (RHAND_OFFSET + 15, RHAND_OFFSET + 14),
        (RHAND_OFFSET + 16, RHAND_OFFSET + 15),

        (RHAND_OFFSET + 17, RHAND_OFFSET + 0),
        (RHAND_OFFSET + 18, RHAND_OFFSET + 17),
        (RHAND_OFFSET + 19, RHAND_OFFSET + 18),
        (RHAND_OFFSET + 20, RHAND_OFFSET + 19),

        (RHAND_OFFSET + 5, RHAND_OFFSET + 1),
        (RHAND_OFFSET + 9, RHAND_OFFSET + 5),
        (RHAND_OFFSET + 13, RHAND_OFFSET + 9),
        (RHAND_OFFSET + 17, RHAND_OFFSET + 13),
    ],

    center=0
)

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='STGCN',
        in_channels=2, #x, y, z (depth), score
        graph_cfg=dict(
            layout=mediapipe_sign_layout,
            mode='stgcn_spatial'
        )
    ),
    cls_head=dict(
        type='GCNHead',
        num_classes=NUM_CLASSES,
        in_channels=256
    )
)

dataset_type = 'PoseDataset'
ann_file = '../dataset/cropped_holistic_results_interpolated_remapped_direct/mediapipe_sign_3d_without_face_pose_score_1.pkl'

vis_backends = [
    dict(type='LocalVisBackend'),
    dict(
        type='WandbVisBackend',
        init_kwargs=dict(
            project='mediapipe-sign-3d',
            name='stgcn_8xb16-joint-u100-80e_mediapipe-sign-keypoint-3d',
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

# j	 	관절 좌표 자체
# b		연결된 관절 간 차이 벡터
# jm	시간에 따른 joint 변화량
# bm	시간에 따른 bone 변화량

train_pipeline = [
    # 3D면 PreNormalize3D를 쓸 수 있지만,
    # MediaPipe z scale이 특수하므로 처음에는 직접 전처리하고 생략하는 것도 가능.
    # dict(type='PreNormalize3D'),
    # dict(type='CenterNormalize2D'),
    # dict(type='BodyCenterNormalize2D', scale=True),
    dict(type='GenSkeFeat', feats=['j']), #모델 입력용 skeleton feature (검출 X, formatting)
    dict(type='UniformSampleFrames', clip_len=100), #프레임 수 통일  #64
    dict(type='PoseDecode'), #UniformSampleFrames의 output index로 indexing
    dict(type='FormatGCNInput', num_person=1), 
    dict(type='PackActionInputs')
]

val_pipeline = [
    # dict(type='CenterNormalize2D'),
    # dict(type='BodyCenterNormalize2D', scale=True),
    dict(type='GenSkeFeat', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=5, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

test_pipeline = [
    # dict(type='CenterNormalize2D'),
    # dict(type='BodyCenterNormalize2D', scale=True),
    dict(type='GenSkeFeat', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=100, num_clips=5, test_mode=True),
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
    type='EpochBasedTrainLoop', max_epochs=50, val_begin=5, val_interval=1) #20
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=50, #16
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.1, momentum=0.9, weight_decay=0.001, nesterov=True))

default_hooks = dict(checkpoint=dict(interval=10), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)
