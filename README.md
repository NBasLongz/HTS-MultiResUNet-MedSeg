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

##  Repository Structure

```text
HTS-MultiResUNet/
│
├── data/                       # Directory to store datasets
│   ├── isbi2012/
│   ├── cvc_clinicdb/
│   ├── isic2018/
│   └── isbi2009/
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

*Core dependencies include: `tensorflow`, `opencv-python`, `numpy`, `matplotlib`, `tqdm`, `scikit-learn`.*

##  Training the Model

To train the model on a specific dataset, use the `train.py` script. The script supports 5-Fold Cross Validation as standard.

```bash
# Example: Train on ISBI-2009
python train.py --dataset isbi2009 --epochs 150 --batch_size 4 --lr 1e-3

```

**Key Arguments:**

* `--dataset`: Choose from `[isbi2012, cvc_clinicdb, isic2018, isbi2009]`.
* `--epochs`: Number of training epochs (Default: 150).
* `--batch_size`: Batch size (Default: 4).
* `--loss`: Objective function to use, e.g., `ea-ftl` (Default).

##  Evaluation & Visualization

To evaluate a trained model and generate Interpretability Analysis (Feature Representation) and Boundary Quality Analysis (Edge Error Maps):

```bash
# Evaluate on ISBI-2012 test set
python evaluate.py --dataset isbi2012 --weights_path /path/to/hts_fold1.weights.h5 --visualize True

```

*Note: Ensure that the weights are strictly loaded (`scale=False` in BN layers) for perfect architecture matching.*

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

