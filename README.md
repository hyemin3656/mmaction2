# FAIRYTALESL - Setup Guide

## Requirements

- Python 3.7+
- CUDA 10.2+
- Recommended Environment
  - CUDA 11.8
  - cuDNN 8
  - PyTorch 2.0.1

---

# 1. Create Conda Environment

```bash
conda create -n openmmlab python=3.8 -y
conda activate openmmlab
```

---

# 2. Install PyTorch

```bash
conda install pytorch=2.0.1 torchvision=0.15.2 pytorch-cuda=11.8 \
-c pytorch -c nvidia -y
```

---

# 3. Install OpenMMLab Dependencies

```bash
pip install -U openmim

conda install "mkl<2024.1" "intel-openmp<2024.1" \
-c defaults -c conda-forge -y

mim install mmengine
mim install "mmcv==2.1.0"

# Optional
mim install mmdet
mim install mmpose
```

---

# 4. Clone Repository

```bash
git clone -b experiment/fairytalesl \
https://github.com/hyemin3656/mmaction2.git

cd mmaction2
```

---

# 5. Install MMACTION2

```bash
pip install -v -e .
```

---

# Dataset Setup

Download annotation file:

https://drive.google.com/drive/folders/1NHFPuGtnF8NU935ZJR1CU0TOaPsqznaW?usp=sharing

Directory structure:

```text
project_root/
├── mmaction2/
└── dataset/
    └── gloss_sequences_splited/
        └── mediapipe_sign_3d.pkl
```
---

# Pretrained Weights

Download pretrained weights:

https://drive.google.com/drive/folders/1h0-ojkDY26MUHfenemiv3XG6t_NkgIEZ?usp=sharing

Example:

```text
checkpoints/
└── model.pth
```

---

# Train and Test 

```bash
#sequence level stgcn-bilstm-ctcdecoder
python tools/train.py configs/skeleton/stgcn/stgcn_ctc_sign.py --seed 0
python tools/test.py configs/skeleton/stgcn/stgcn_ctc_sign.py [checkpoint path] --dump result.pkl


```

#Model Performance (validation)
st-gcn : 0.81
1d-cnn : 0.90
