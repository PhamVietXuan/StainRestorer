# StainRestorer

PVXuan: This repo is forked from [StainRestorer](https://github.com/CXH-Research/StainRestorer). I made some changes to the code: re-train the model and do something else.

[**High-Fidelity Document Stain Removal via A Large-Scale Real-World Dataset and A Memory-Augmented Transformer**](https://arxiv.org/abs/2410.22922)

<div>
<span class="author-block">
  Mingxian Li<sup> 👨‍💻‍ </sup>
</span>,
  <span class="author-block">
    Hao Sun<sup> 👨‍💻‍ </sup>
  </span>,
  <span class="author-block">
    Yingtie Lei<sup> 👨‍💻‍ </sup>
  </span>,
  <span class="author-block">
    <a href='https://zhangbaijin.github.io/'>Xiaofeng Zhang</a>
  </span>,
  <span class="author-block">
    Yihang Dong
  </span>,
  <span class="author-block">
    Yilin Zhou
  </span>,
  <span class="author-block">
    Zimeng Li
  </span>,
  <span class="author-block">
  <a href='https://cxh.netlify.app/'>Xuhang Chen</a><sup> 📮</sup>
</span>
  ( 👨‍💻‍ Equal contributions, 📮 Corresponding author)
</div>

<b>Huizhou Univeristy, University of Macau, Shanghai Jiao Tong University, SIAT CAS, Shenzhen Polytechnic University</b>

In <b>_IEEE/CVF Winter Conference on Applications of Computer Vision 2025 (WACV 2025)_</b>

# 🔮 Dataset

[Kaggle](https://www.kaggle.com/datasets/xuhangc/wacv2025-staindoc)

StainDoc is the first large-scale high-resolution dataset that includes ground truth data specifically for the task of document stain removal.

StainDoc_mark and StainDoc_seal are made with the process in [DocDiff](https://github.com/Royalvice/DocDiff).

# ⚙️ Usage
PVXuan: There is some change in the code that I made. I will update the README.md and code later.

## Install needed package 

Optional: You should create virtual environment to install the package.

for Mac/Linux
```bash
  python -m venv stainrestorer_env
  source stainrestorer_env/bin/activate

```
for Windows
```bash
  python -m venv stainrestorer_env
  stainrestorer_env\Scripts\activate
```

Install the package
```bash
  pip install -r requirements.txt
```

## Training
You may download the dataset first, and then specify TRAIN_DIR, VAL_DIR and SAVE_DIR in the section TRAINING in `config.yml`.

PVXuan note: The dataset size is too large(zip file is 49GB, and need ~ 100GB to unzip). 
the training process is quite slow(I trained 1 epoch in 1 GPU T4 in about 3 hours).

For single GPU training:
```bash
  python train.py
```
For multiple GPUs training:
```bash
  accelerate config
  accelerate launch train.py
```
If you have difficulties with the usage of `accelerate`, please refer to <a href="https://github.com/huggingface/accelerate">Accelerate</a>.

## Inference

Please first specify TRAIN_DIR, VAL_DIR and SAVE_DIR in section TESTING in `config.yml`.

```bash
  python infer.py
```

# [Citation](https://github.com/CXH-Research/StainRestorer)


