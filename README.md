This repository contains the code for DGA-Net, including core files such as model structure definitions and data preprocessing workflows. Disclaimer Due to the confidentiality agreement of the author's affiliated institution and intellectual property protection requirements, the model weight files and detailed training process cannot be made public temporarily. The author has provided relevant files including model architecture and data preprocessing as much as possible. All related files (including weights, complete training process, etc.) will be made public after the paper is officially accepted.

# 1.Dependency library versions [requirements.txt]
# Deep learning framework
torch==1.13.1+cu117
torchvision==0.14.1+cu117

# Data processing and visualization
numpy==1.23.5
pandas==1.5.3
matplotlib==3.6.2
SimpleITK==2.2.1 

# Model training assistance
tqdm==4.64.1
scikit-learn==1.2.0 

# Other dependencies
opencv-python==4.6.0.66
Pillow==9.3.0 
scipy==1.9.3
nibabel==5.1.0
scikit-image==0.19.3
python-dotenv==1.0.0
seaborn==0.12.2

# 2.Description of the runtime environment
Processor (CPU) is 12th Gen Intel Core i7-12700K with a clock speed of 3.60 GHz, 
GPU Memory is 24 GB, Graphics Card is NVIDIA GeForce RTX 3090 (GDDR6X VRAM), OS is Ubuntu 20.04, Environment Setup is PyTorch-GPU 1.13.1, 
Programming Language is Python 3.10. 

# 3.Data partition [dataset_splits]
LiTS2017:

The dataset split (7:1:2, 92/13/26 cases) is based on the official LiTS2017 training set (131 cases). We provide the split IDs ("Volume" refers to the CT scan volume, and "segmentation" refers to segmentation annotation.):

-Training set (92 cases): IDs [(volume-0.nii.gz, segmentation-0.nii.gz), (volume-1.nii.gz, segmentation-1.nii.gz), ..., (volume-092.nii.gz, segmentation-092.nii.gz)] (consecutive indexing of official training cases)

-Validation set (13 cases): IDs [(volume-093.nii.gz, segmentation-093.nii.gz), (volume-094.nii.gz, segmentation-094.nii.gz), ..., (volume-105.nii.gz, segmentation-105.nii.gz)]

-Test set (26 cases): IDs [(volume-106.nii.gz, segmentation-106.nii.gz), (volume-107.nii.gz, segmentation-107.nii.gz), ..., (volume-131.nii.gz, segmentation-131.nii.gz)]

3DIRCADb Dataset (15 test cases)：
Test Set： Patient_01~Patient_11, Patient_13~Patient_16

# 4.The function of each file.
# Core Configuration and Initialization Files
[init.py (root directory)]: Initializes the project package structure to enable cross-file import and calling of various modules.

[config.py]: Stores core configuration information such as model training parameters, data processing settings, and hyperparameters to uniformly manage experimental parameters.

# Data-Related Files
[datasets.py]: Defines data loading classes to realize the reading of the LiTS2017 and 3DIRCADb datasets and their connection with preprocessing operations.

3DIRCADb_case_splits.txt: Records the division scheme of the 3DIRCADb dataset into training set, validation set, and test set.

LiTS2017_case_splits.txt: Records the splitting ratio and sample allocation of the LiTS2017 dataset.

[preprocess.py]: Implements preprocessing operations for CT images, including intensity clipping, normalization, resampling, etc.

# Model Structure Files
[dag_net.py]: Constructs the overall network architecture of S2DA-Net, integrating the dual-branch encoder and the GMCA decoder.

[fsmf_module.py]: Implements the Fourier Spectral-learning Multi-scale Fusion (FSMF) module to extract amplitude and phase features.

[maha_module.py]: Implements the Multi-axis Aggregation Hadamard Attention (MAHA) module to fuse spatial information and reduce computational load.

[gmca_module.py]: Implements the Group Multi-Head Cross-Attention Aggregation (GMCA) module to integrate multi-branch features and capture long-term dependencies.

# Training and Loss Function Files
[train.py]: Defines the model training process, including logic for data loading, model training, validation, and saving.

[loss.py]: Implements a composite loss function combining cross-entropy loss and Dice loss for model training optimization.

# Evaluation and Metric Files
[evaluate.py]: Implements the logic for model performance evaluation, calculating experimental metrics such as DPC, DG, and VOE, and outputting the results.

[metrics.py]: Defines calculation methods for evaluation metrics including Dice score and volume overlap error (VOE), which are called by the evaluation module.

# Tool Auxiliary File
[logger.py]: Implements a log recording function to track the training process, parameter configuration, and experimental results.

# 5. Script-to-Table Mapping for DGA-Net Experiments
This document maps each main/ablation table in the manuscript to the corresponding reproduction script, ensuring full reproducibility.

## 1. Main Comparative Experiments (LiTS2017 Dataset)
| Table Number | Table Name | Corresponding Script | Key Notes |
|--------------|------------|---------------------|-----------|
| Table 1 | Liver Segmentation on LiTS2017 | `main.py`  | Evaluates DGA-Net vs. 8 SOTA models (U-Net, UNet++, etc.) for liver segmentation; outputs DPC/DG/VOE/RAVD/ASSD |
| Table 2 | Tumor Segmentation on LiTS2017 | `main.py`  | Evaluates tumor segmentation performance; includes paired t-tests with α=0.05 |


## 2. Ablation Experiments (LiTS2017 Dataset)
| Table Number | Table Name | Corresponding Script                    | Key Notes |
|--------------|------------|-----------------------------------------|---------|
| Table 3 | Ablation on Liver Segmentation | `main_ablation.py`                      | Ablates FSMF/ConvFFT/GMCA/MAHA modules, single-branch, and fusion methods |
| Table 4 | Ablation on Tumor Segmentation | `main_ablation.py`               | Same ablation settings as Table 3, focused on tumor segmentation |
| Table 5 | Influence of Tumor Size | `main_ablation.py` | Stratifies tumors by size (<2cm/2-5cm/>5cm); outputs size-specific DPC |
| Table 6 | Influence of Gaussian Noise | `main_ablation.py`                  | Adds Gaussian noise to input images; evaluates robustness of all ablation variants |

## 3. Cross-Dataset Evaluation (3DIRCADb Dataset)
| Table Number | Table Name | Corresponding Script   | Key Notes |
|--------------|------------|------------------------|-----------|
| Table 7 | Liver Segmentation on 3DIRCADb | `main.py`              |  Uses LiTS2017-trained model; directly tests on 3DIRCADb |
| Table 8 | Tumor Segmentation on 3DIRCADb | `main.py`              |  Includes paired t-tests |

## Key Reproduction Guarantees
1. All scripts use fixed random seeds (PyTorch/NumPy/Python: 42) for deterministic results.
2. Batch size, epochs, and optimizer settings are unified as per the manuscript.
3. Baseline models are retrained under the same pipeline (no pre-trained weights from external sources).


