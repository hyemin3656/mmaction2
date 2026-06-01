_base_ = '../../_base_/default_runtime.py'

NUM_CLASSES = 67

POSE_OFFSET = 0
FACE_OFFSET = 23
LHAND_OFFSET = 23 + 468      # 491
RHAND_OFFSET = 23 + 468 + 21 # 512

def shift(edges, offset):
    return [(a + offset, b + offset) for a, b in edges]


# MediaPipe Face Mesh sparse contour graph
# 원래 face index 기준: 0~467
FACE_LIPS = [
    (61,146),(146,91),(91,181),(181,84),(84,17),
    (17,314),(314,405),(405,321),(321,375),(375,291),
    (61,185),(185,40),(40,39),(39,37),(37,0),
    (0,267),(267,269),(269,270),(270,409),(409,291),

    (78,95),(95,88),(88,178),(178,87),(87,14),
    (14,317),(317,402),(402,318),(318,324),(324,308),
    (78,191),(191,80),(80,81),(81,82),(82,13),
    (13,312),(312,311),(311,310),(310,415),(415,308),
]

FACE_LEFT_EYE = [
    (263,249),(249,390),(390,373),(373,374),
    (374,380),(380,381),(381,382),(382,362),
    (263,466),(466,388),(388,387),(387,386),
    (386,385),(385,384),(384,398),(398,362),
]

FACE_RIGHT_EYE = [
    (33,7),(7,163),(163,144),(144,145),
    (145,153),(153,154),(154,155),(155,133),
    (33,246),(246,161),(161,160),(160,159),
    (159,158),(158,157),(157,173),(173,133),
]

FACE_LEFT_EYEBROW = [
    (276,283),(283,282),(282,295),(295,285),
    (300,293),(293,334),(334,296),(296,336),
]

FACE_RIGHT_EYEBROW = [
    (46,53),(53,52),(52,65),(65,55),
    (70,63),(63,105),(105,66),(66,107),
]

FACE_OVAL = [
    (10,338),(338,297),(297,332),(332,284),(284,251),
    (251,389),(389,356),(356,454),(454,323),(323,361),
    (361,288),(288,397),(397,365),(365,379),(379,378),
    (378,400),(400,377),(377,152),(152,148),(148,176),
    (176,149),(149,150),(150,136),(136,172),(172,58),
    (58,132),(132,93),(93,234),(234,127),(127,162),
    (162,21),(21,54),(54,103),(103,67),(67,109),(109,10),
]

FACE_NOSE = [
    (168,6),(6,197),(197,195),(195,5),(5,4),
    (4,45),(45,220),(220,115),(115,48),
    (4,275),(275,440),(440,344),(344,278),
]

FACE_CENTER_LINKS = [
    # 얼굴 내부 feature들을 nose tip/center 쪽으로 약하게 연결
    (4, 0),      # nose tip - upper lip center 근처
    (4, 13),     # nose tip - upper lip
    (4, 14),     # nose tip - lower lip
    (4, 33),     # nose tip - right eye outer
    (4, 263),    # nose tip - left eye outer
    (4, 152),    # nose tip - chin
]

FACE_EDGES = (
    FACE_LIPS
    + FACE_LEFT_EYE
    + FACE_RIGHT_EYE
    + FACE_LEFT_EYEBROW
    + FACE_RIGHT_EYEBROW
    + FACE_OVAL
    + FACE_NOSE
    + FACE_CENTER_LINKS
)

mediapipe_sign_layout = dict(
    num_node=533,  # pose 23 + face 468 + left hand 21 + right hand 21

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
        # Face 23~490
        # -------------------------
        # face landmark는 MediaPipe 원래 index 0~467에 +23 offset
        *shift(FACE_EDGES, FACE_OFFSET),

        # pose nose와 face nose tip 연결
        # pose 0 = nose, face 4 = nose tip
        (FACE_OFFSET + 4, 0),

        # pose eyes/ears 쪽과 face eye contour 약한 연결
        (FACE_OFFSET + 33, 7),
        (FACE_OFFSET + 263, 8),

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
        in_channels=4, #x, y, z (depth), score
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
ann_file = '../dataset/cropped_holistic_results/mediapipe_sign_3d.pkl'

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

    dict(type='GenSkeFeat', feats=['j']), #모델 입력용 skeleton feature (검출 X, formatting)
    dict(type='UniformSampleFrames', clip_len=64), #프레임 수 통일 #***
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

val_pipeline = [
    dict(type='GenSkeFeat', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=64, num_clips=1, test_mode=True),
    dict(type='PoseDecode'),
    dict(type='FormatGCNInput', num_person=1),
    dict(type='PackActionInputs')
]

test_pipeline = [
    dict(type='GenSkeFeat', feats=['j']),
    dict(type='UniformSampleFrames', clip_len=64, num_clips=10, test_mode=True),
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
        split='val',
        test_mode=True))

val_evaluator = [dict(type='AccMetric')]
test_evaluator = val_evaluator

train_cfg = dict(
    type='EpochBasedTrainLoop', max_epochs=30, val_begin=5, val_interval=1) #20
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

param_scheduler = [
    dict(
        type='CosineAnnealingLR',
        eta_min=0,
        T_max=30, #16
        by_epoch=True,
        convert_to_iter_based=True)
]

optim_wrapper = dict(
    optimizer=dict(
        type='SGD', lr=0.1, momentum=0.9, weight_decay=0.0005, nesterov=True))

default_hooks = dict(checkpoint=dict(interval=10), logger=dict(interval=100))

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (16 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=128)
