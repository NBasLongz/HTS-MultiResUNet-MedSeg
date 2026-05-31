# HTS-MultiResUNet for Medical Image Segmentation

![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-3.x-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

This repository contains the official implementation of **HTS-MultiResUNet**, a robust deep learning architecture proposed for medical image segmentation. 

##  Team Members

| No. | Full Name | Student ID | GitHub |
| :---: | :--- | :---: | :--- |
| 1 | Nguyễn Hữu Khánh Duy | 23520375 |  |
| 2 | Nguyễn Bá Long | 23520880 | [NBasLongz](https://github.com/NBasLongz) |
| 3 | Hồ Hoàng Quân | 23521252 |  |

##  Overview

Accurate segmentation of medical images (e.g., polyps, skin lesions, neuronal structures) is critical for clinical diagnosis. **HTS-MultiResUNet** enhances the standard MultiResUNet by introducing:
1. **SE-MultiRes Block:** Integrates Squeeze-and-Excitation (SE) mechanisms to recalibrate channel-wise feature responses dynamically.
2. **Transformer Bottleneck:** Employs Multi-Head Self-Attention (MHSA) to capture global dependencies and contextual information at the deepest layer.
3. **Edge-Aware Focal Tversky Loss (EA-FTL):** A novel objective function combining Focal Tversky Loss and Sobel-based boundary structural error to address severe class imbalance and morphological degradation at object boundaries.

##  Datasets
 
 The model has been rigorously evaluated on 4 diverse public medical imaging datasets:
 - **ISBI-2012:** Neuronal structure segmentation (Electron Microscopy).
 - **CVC-ClinicDB:** Colonoscopy polyp segmentation.
 - **ISIC-2018:** Skin lesion segmentation (Dermoscopy).
 - **ISBI-2009:** Medical image segmentation (Small-scale dataset containing 97 images to evaluate model robustness on limited data).

### Chuẩn bị dữ liệu (Data Preparation)

Vui lòng tổ chức dữ liệu của bạn theo cấu trúc thư mục như sau để mã nguồn hoạt động chính xác:

1. **ISBI-2012 challenge**: Bộ dữ liệu được lưu dưới dạng file TIFF 3D:
   - Ảnh gốc: `data/isbi2012/train-volume.tif`
   - Mask nhãn: `data/isbi2012/train-labels.tif`
2. **CVC-ClinicDB**: Tổ chức dưới dạng các thư mục ảnh gốc và ground-truth tương ứng:
   - Thư mục ảnh gốc: `data/cvc_clinicdb/Original/` (chứa các ảnh dạng `1.png`, `2.png`, ...)
   - Thư mục mask nhãn: `data/cvc_clinicdb/Ground Truth/` (chứa các mask tương ứng `1.png`, `2.png`, ...)
3. **ISIC-2018**: Tổ chức thư mục ảnh gốc và masks:
   - Thư mục ảnh gốc: `data/isic2018/ISIC2018_Task1-2_Training_Input/` (chứa các ảnh dạng `ISIC_0000000.jpg`, ...)
   - Thư mục mask nhãn: `data/isic2018/ISIC2018_Task1_Training_GroundTruth/` (chứa các mask tương ứng `ISIC_0000000_segmentation.png`, ...)
4. **ISBI-2009**: Tổ chức thư mục ảnh gốc và masks:
   - Thư mục ảnh gốc: `data/isbi2009/images/` (chứa các ảnh dạng `image_0.png`, ...)
   - Thư mục mask nhãn: `data/isbi2009/masks/` (chứa các mask tương ứng `mask_0.png`, ...)
 
 ##  Repository Structure
 
 ```text
 HTS-MultiResUNet/
 │
 ├── data/                       # Directory to store datasets
 │   ├── isbi2012/               # train-volume.tif, train-labels.tif
 │   ├── cvc_clinicdb/           # Original/, Ground Truth/
 │   ├── isic2018/               # ISIC2018_Task1-2_Training_Input/, ISIC2018_Task1_Training_GroundTruth/
 │   └── isbi2009/               # images/, masks/
│
├── models/                     # Model architecture definitions
│   ├── hts_multiresunet.py     # Proposed architecture
│   └── baseline_multires.py    # Baseline MultiResUNet
│
├── utils/                      # Helper functions
│   ├── losses.py               # Implementation of EA-FTL
│   ├── metrics.py              # Dice, IoU calculation
│   └── visualization.py        # Grad-CAM & Error map generation
│
├── train.py                    # Main training script
├── evaluate.py                 # Evaluation & metrics calculation
├── requirements.txt            # Python dependencies
└── README.md
```

##  Installation & Setup

1. **Clone the repository:**

```bash
git clone [https://github.com/NBasLongz/HTS-MultiResUNet.git](https://github.com/NBasLongz/HTS-MultiResUNet.git)
cd HTS-MultiResUNet

```

2. **Create a virtual environment (optional but recommended):**

```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

```

3. **Install dependencies:**

```bash
pip install -r requirements.txt

```

*Core dependencies include: `tensorflow`, `opencv-python`, `numpy`, `matplotlib`, `tqdm`, `scikit-learn`, `tifffile`.*

##  Training the Model
 
 To train the model on a specific dataset, use the `train.py` script. The script hỗ trợ quy trình 5-Fold Cross Validation thực tế trên dữ liệu thật.
 
 ```bash
 # Ví dụ: Train trên ISBI-2009 bằng dữ liệu thực tế
 python train.py --dataset isbi2009 --image_dir ./data/isbi2009/images --mask_dir ./data/isbi2009/masks --epochs 150 --batch_size 4 --lr 1e-3 --save_dir ./results
 
 # Ví dụ: Train trên ISBI-2012 (dùng file TIFF 3D)
 python train.py --dataset isbi2012 --image_dir ./data/isbi2012/train-volume.tif --mask_dir ./data/isbi2012/train-labels.tif --epochs 150 --batch_size 4 --lr 1e-3 --save_dir ./results
 ```
 
 **Key Arguments:**
 
 * `--dataset`: Chọn một trong các tập `[isbi2012, cvc_clinicdb, isic2018, isbi2009]`.
 * `--image_dir`: Đường dẫn đến thư mục chứa ảnh gốc (hoặc file `.tif` đối với ISBI-2012).
 * `--mask_dir`: Đường dẫn đến thư mục chứa mask nhãn (hoặc file `.tif` đối với ISBI-2012).
 * `--model_name`: Lựa chọn mô hình `[hts_multiresunet, baseline_multires]` (Mặc định: `hts_multiresunet`).
 * `--epochs`: Số lượng epochs tối đa (Mặc định: 150).
 * `--batch_size`: Batch size (Mặc định: 4).
 * `--lr`: Learning rate ban đầu (Mặc định: 1e-3).
 * `--save_dir`: Thư mục lưu checkpoint weights và kết quả log (Mặc định: `./results`).
 
 ##  Evaluation & Visualization
 
 Để đánh giá mô hình đã được train và tự động sinh ảnh phân tích giải thích (Grad-CAM XAI) cùng bản đồ lỗi biên biên (Edge Error Maps):
 
 ```bash
 # Đánh giá mô hình trên tập kiểm tra ISBI-2009 và sinh hình ảnh trực quan
 python evaluate.py --dataset isbi2009 --image_dir ./data/isbi2009/images --mask_dir ./data/isbi2009/masks --weights_path ./results/hts_multiresunet_isbi2009_fold1.weights.h5 --visualize True --save_dir ./results/visualizations
 ```
 
 *Note: Khi chạy evaluate với `--visualize True` trên mô hình `hts_multiresunet`, hệ thống sẽ tự động vẽ bản đồ XAI (CNN Features + Attention maps) và lưu kết quả vào thư mục được chỉ định.*

##  Citation

If you find this repository or the HTS-MultiResUNet architecture useful in your research, please consider citing our work:

```bibtex
@article{nguyen2026htsmultiresunet,
  title={HTS-MultiResUNet: A Hybrid Transformer-CNN Architecture with Edge-Aware Focal Tversky Loss for Medical Image Segmentation},
  author={Nguyễn, Bá Long and Nguyễn, Hữu Khánh Duy and Hồ, Hoàng Quân},
  journal={TBD},
  year={2026}
}

```

##  Acknowledgments

* The baseline MultiResUNet implementation is inspired by the original work of N. Ibtehaz and M. S. Rahman.
* Computational resources were supported by Kaggle Cloud Environment (NVIDIA Tesla P100/T4x2).

##  License

This project is licensed under the MIT License - see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.

