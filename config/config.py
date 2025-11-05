import torch
import random

class Config:
    """Configuration class for model training and data processing."""

    # Data configuration
    DATASET = "LiTS2017"  # Available options: LiTS2017/3DIRCADb
    RAW_DATA_PATH = "./data/raw/"  # Path to store raw dataset files
    PROCESSED_DATA_PATH = "./data/processed/"  # Path to store preprocessed data
    TRAIN_TEST_SPLIT = [0.7, 0.1, 0.2]  # Train/validation/test split ratios
    TARGET_SPACING = (0.7676, 0.7676, 5.0)  # Target spacing for resampling (x,y,z in mm)
    HU_RANGE = (-17, 201)  # Hounsfield Unit clipping range for CT preprocessing
    MEAN = 99.04  # Global mean intensity value (from paper)
    STD = 39.36  # Global standard deviation (from paper)

    # Model architecture configuration
    IN_CHANNELS = 1  # Input channels (1 for CT grayscale images)
    NUM_CLASSES = 3  # Number of segmentation classes: background(0)/liver(1)/tumor(2)
    ENCODER_DEPTH = 5  # Number of encoder layers in the network
    FSMF_NUM_SCALES = 3  # Number of multi-scale fusion branches in FSMF module
    MAHA_NUM_GROUPS = 4  # Number of feature groups in MAHA module
    GMCA_NUM_HEADS = 4  # Number of attention heads in GMCA module

    # Training hyperparameters
    BATCH_SIZE = 4  # Number of samples per training batch
    MAX_EPOCHS = 1000  # Maximum number of training epochs
    LEARNING_RATE = 0.01  # Initial learning rate for optimizer
    WEIGHT_DECAY = 0.0001  # L2 regularization factor
    OPTIMIZER = "SGD"  # Optimization algorithm (SGD as used in paper)
    LR_SCHEDULER = "poly"  # Learning rate scheduler type
    POLY_POWER = 0.9  # Power factor for polynomial LR decay
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # Training device
    random.seed(42)
    # Evaluation configuration
    EVAL_METRICS = ["DPC", "DG", "VOE", "RAVD", "ASSD"]  # Evaluation metrics for segmentation
    CROSS_VALIDATION_FOLDS = 8  # 8-fold cross validation (as specified in paper)


config = Config()  # Create configuration instance