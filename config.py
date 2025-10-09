"""
BioBERT-based Medical Article Quality Classification
Implementasi untuk paper: NLP for quality evaluation of Wikipedia medical articles
Dengan improvisasi menggunakan Transformer (BioBERT) dan multi-input architecture
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import logging
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Config:
    """Configuration class untuk hyperparameters"""
    # Model settings
    MODEL_NAME = 'dmis-lab/biobert-v1.1'  # BioBERT pre-trained model
    MAX_LENGTH = 512
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 2e-5
    WARMUP_STEPS = 500
    WEIGHT_DECAY = 0.01
    
    # Architecture settings
    HIDDEN_DIM = 256
    DROPOUT_RATE = 0.3
    TABULAR_INPUT_DIM = 8  # Jumlah fitur tabular
    
    # Training settings
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    SEED = 42
    NUM_CLASSES = 6  # Stub, Start, C, B, GA, FA
    
    # Data augmentation
    USE_AUGMENTATION = True
    USE_CONTEXTUAL_AUG = True
    USE_BACK_TRANSLATION = False
    USE_SMOTE = False  # Use SMOTE for tabular features
    BACK_TRANS_LANGUAGES = ['de', 'fr']  # German, French
    AUG_TARGET_CLASSES = ['GA', 'FA']  # Minority classes to augment
    AUG_SAMPLES_PER_CLASS = 30  # Maximum augmented samples to generate per class (not multiplier!)
    
    SMOTE_SAMPLING = {
        'GA': 0.4,  # 40% oversampling
        'FA': 1.8   # 180% oversampling
    }
    
    # Paths
    OUTPUT_DIR = 'outputs'
    LOG_FILE = 'training_log.csv'
    MODEL_SAVE_PATH = 'best_model.pt'
    
print("All configs loaded successfully!")