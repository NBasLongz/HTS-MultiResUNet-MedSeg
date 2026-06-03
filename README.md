# HTS-MultiResUNet for Medical Image Segmentation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-3.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository implements **HTS-MultiResUNet**, a hybrid Transformer-CNN architecture for medical image segmentation with edge-aware supervision.

## Team Members

| No. | Full Name | Student ID |
| :---: | :--- | :---: |
| 1 | Nguyễn Hữu Khánh Duy | 23520375 |
| 2 | Nguyễn Bá Long | 23520880 |
| 3 | Hồ Hoàng Quân | 23521252 |

## Overview

HTS-MultiResUNet extends MultiResUNet by combining:

- **SE-MultiRes Blocks** for channel-wise feature recalibration.
- **Transformer bottleneck** with Multi-Head Self-Attention to capture long-range context.
- **Edge-Aware Focal Tversky Loss (EA-FTL)** for class-imbalanced medical segmentation and sharper boundary learning.

## Supported Datasets

- **ISBI-2012**: Neuronal structure segmentation (Electron Microscopy, 256x256 grayscale).
- **CVC-ClinicDB**: Colonoscopy polyp segmentation (RGB, 192x256).
- **ISIC-2018**: Skin lesion segmentation (RGB, 192x256).
- **ISBI-2009**: Medical image segmentation (RGB, 256x256, small-scale).

## Data Preparation

Sắp xếp dữ liệu theo cấu trúc sau để `train.py` và `evaluate.py` hoạt động đúng:

### ISBI-2012

- `data/isbi2012/train-volume.tif`
- `data/isbi2012/train-labels.tif`

### CVC-ClinicDB

- `data/cvc_clinicdb/Original/` chứa ảnh gốc `1.png`, `2.png`, ...
- `data/cvc_clinicdb/Ground Truth/` chứa mask `1.png`, `2.png`, ...

### ISIC-2018

- `data/isic2018/ISIC2018_Task1-2_Training_Input/` chứa ảnh gốc `ISIC_0000000.jpg`, ...
- `data/isic2018/ISIC2018_Task1_Training_GroundTruth/` chứa mask `ISIC_0000000_segmentation.png`, ...

### ISBI-2009

- `data/isbi2009/images/` chứa ảnh gốc `image_0.png`, ...
- `data/isbi2009/masks/` chứa mask `mask_0.png`, ...

## Repository Structure

```text
HTS-MultiResUNet/
├── data/                       # Dataset folders
│   ├── isbi2012/               # train-volume.tif, train-labels.tif
│   ├── cvc_clinicdb/           # Original/, Ground Truth/
│   ├── isic2018/               # ISIC2018_Task1-2_Training_Input/, ISIC2018_Task1_Training_GroundTruth/
│   └── isbi2009/               # images/, masks/
├── models/                     # Model definitions
│   ├── hts_multiresunet.py     # Proposed HTS-MultiResUNet architecture
│   └── baseline_multires.py    # Baseline MultiResUNet architecture
├── utils/                      # Dataset, loss, metrics, visualization
│   ├── dataset.py
│   ├── losses.py
│   ├── metrics.py
│   └── visualization.py
├── train.py                    # Training pipeline with K-Fold CV
├── evaluate.py                 # Evaluation and visualization pipeline
├── requirements.txt            # Python dependencies
└── README.md
```

## Installation

```bash
git clone https://github.com/NBasLongz/HTS-MultiResUNet.git
cd HTS-MultiResUNet
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

## Training

Sử dụng `train.py` để huấn luyện với K-Fold cross validation.

```bash
python train.py \
  --dataset isbi2009 \
  --image_dir ./data/isbi2009/images \
  --mask_dir ./data/isbi2009/masks \
  --model_name hts_multiresunet \
  --epochs 150 \
  --batch_size 4 \
  --lr 1e-3 \
  --save_dir ./results
```

### Train trên ISBI-2012

```bash
python train.py \
  --dataset isbi2012 \
  --image_dir ./data/isbi2012/train-volume.tif \
  --mask_dir ./data/isbi2012/train-labels.tif \
  --model_name hts_multiresunet \
  --epochs 150 \
  --batch_size 4 \
  --lr 1e-3 \
  --save_dir ./results
```

### Tham số chính

- `--dataset`: `[isbi2012, cvc_clinicdb, isic2018, isbi2009]`
- `--image_dir`: Thư mục ảnh gốc hoặc file `.tif` cho ISBI-2012
- `--mask_dir`: Thư mục mask hoặc file `.tif` cho ISBI-2012
- `--model_name`: `[hts_multiresunet, baseline_multires]` (mặc định `hts_multiresunet`)
- `--epochs`: Số epoch tối đa (mặc định `150`)
- `--batch_size`: Batch size; nếu không chỉ định, script tự chọn theo dataset
- `--lr`: Learning rate ban đầu (mặc định `1e-3`)
- `--n_folds`: Số fold cho K-Fold CV (mặc định `5`)
- `--early_stopping`: Patience cho EarlyStopping (mặc định `30`)
- `--save_dir`: Thư mục lưu weights và logs

## Evaluation & Visualization

Sử dụng `evaluate.py` để đánh giá và tạo ảnh XAI / error map.

```bash
python evaluate.py \
  --dataset isbi2009 \
  --image_dir ./data/isbi2009/images \
  --mask_dir ./data/isbi2009/masks \
  --weights_path ./results/hts_multiresunet_isbi2009_fold1.weights.h5 \
  --model_name hts_multiresunet \
  --visualize True \
  --save_dir ./results/visualizations
```

### Tham số chính

- `--weights_path`: Đường dẫn tới file weights đã lưu
- `--model_name`: `[hts_multiresunet, baseline_multires]`
- `--visualize`: `True` hoặc `False`
- `--save_dir`: Thư mục lưu kết quả trực quan

> Lưu ý: `--visualize True` chỉ kích hoạt XAI cho `hts_multiresunet`; với `baseline_multires`, chỉ sinh error map.

## Ghi chú

- `train.py` và `evaluate.py` sử dụng `utils.dataset.load_dataset()` để load và chuẩn hóa ảnh/mask.
- `isbi2012` được xử lý dưới dạng ảnh đơn kênh (grayscale); các dataset khác là ảnh RGB.
- Mỗi dataset có kích thước đầu vào cố định: `isbi2012` (256x256), `cvc_clinicdb` và `isic2018` (192x256), `isbi2009` (256x256).

## Citation

```bibtex
@article{nguyen2026htsmultiresunet,
  title={HTS-MultiResUNet: A Hybrid Transformer-CNN Architecture with Edge-Aware Focal Tversky Loss for Medical Image Segmentation},
  author={Nguyễn, Bá Long and Nguyễn, Hữu Khánh Duy and Hồ, Hoàng Quân},
  journal={TBD},
  year={2026}
}
```

## Acknowledgments

- Baseline MultiResUNet được tham khảo từ công trình gốc của N. Ibtehaz và M. S. Rahman.
- Tài nguyên tính toán được hỗ trợ bởi môi trường Kaggle Cloud (NVIDIA Tesla P100/T4x2).

## License

This project is licensed under the MIT License.
