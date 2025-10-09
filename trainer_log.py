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

class TrainingLogger:
    """Logger untuk menyimpan metrics training"""
    
    def __init__(self, log_file):
        self.log_file = log_file
        self.logs = []
    
    def log_epoch(self, epoch, train_loss, train_acc, val_loss, val_acc, 
                  val_f1_macro, val_roc_auc_macro, learning_rate):
        log_entry = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc,
            'val_f1_macro': val_f1_macro,
            'val_roc_auc_macro': val_roc_auc_macro,
            'learning_rate': learning_rate,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        self.logs.append(log_entry)
    
    def save_logs(self):
        df = pd.DataFrame(self.logs)
        df.to_csv(self.log_file, index=False)
        logger.info(f"Training logs saved to {self.log_file}")