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

class BioBERTMultiInputClassifier(nn.Module):
    """
    Multi-Input Architecture:
    - Input A: Text (BioBERT)
    - Input B: Tabular features (Dense layers)
    - Concatenation + Classification
    """
    
    def __init__(self, config):
        super(BioBERTMultiInputClassifier, self).__init__()
        
        # BioBERT encoder untuk text
        self.biobert = AutoModel.from_pretrained(config.MODEL_NAME)
        self.bert_hidden_size = self.biobert.config.hidden_size  # 768 untuk BERT-base
        
        # Dense layers untuk tabular features
        self.tabular_fc1 = nn.Linear(config.TABULAR_INPUT_DIM, 128)
        self.tabular_fc2 = nn.Linear(128, 64)
        self.tabular_dropout = nn.Dropout(config.DROPOUT_RATE)
        
        # Combined features processing
        combined_dim = self.bert_hidden_size + 64  # 768 + 64 = 832
        self.fc1 = nn.Linear(combined_dim, config.HIDDEN_DIM)
        self.fc2 = nn.Linear(config.HIDDEN_DIM, config.HIDDEN_DIM // 2)
        self.dropout = nn.Dropout(config.DROPOUT_RATE)
        
        # Output layer
        self.classifier = nn.Linear(config.HIDDEN_DIM // 2, config.NUM_CLASSES)
        
        # Activation functions
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
    
    def forward(self, input_ids, attention_mask, tabular_features):
        # Process text dengan BioBERT
        bert_output = self.biobert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        # Ambil [CLS] token embedding (representasi seluruh text)
        text_features = bert_output.last_hidden_state[:, 0, :]  # Shape: (batch_size, 768)
        
        # Process tabular features
        tab_features = self.relu(self.tabular_fc1(tabular_features))
        tab_features = self.tabular_dropout(tab_features)
        tab_features = self.relu(self.tabular_fc2(tab_features))  # Shape: (batch_size, 64)
        
        # Concatenate text + tabular features
        combined = torch.cat([text_features, tab_features], dim=1)  # Shape: (batch_size, 832)
        
        # Dense layers untuk classification
        x = self.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer
        logits = self.classifier(x)
        
        return logits