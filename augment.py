import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup, pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import logging
from datetime import datetime
from tqdm import tqdm
import random
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logger = logging.getLogger(__name__)

class TextAugmenter:
    """
    Text Augmentation dengan dua metode:
    1. Contextual Augmentation: menggunakan masked language model
    2. Back Translation: translate ke bahasa lain lalu kembali ke English
    """
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        # Setup Contextual Augmentation (menggunakan BioBERT untuk MLM)
        if config.USE_CONTEXTUAL_AUG:
            logger.info("Loading BioBERT for Contextual Augmentation...")
            self.mlm_model_name = 'dmis-lab/biobert-v1.1'
            self.mlm_tokenizer = AutoTokenizer.from_pretrained(self.mlm_model_name)
            self.mlm_model = AutoModel.from_pretrained(self.mlm_model_name).to(self.device)
            self.mlm_pipeline = pipeline(
                'fill-mask',
                model=self.mlm_model_name,
                tokenizer=self.mlm_tokenizer,
                device=0 if self.device.type == 'cuda' else -1
            )
            logger.info("✓ Contextual Augmentation ready")
        
        # Setup Back Translation
        if config.USE_BACK_TRANSLATION:
            logger.info("Loading Back Translation models...")
            self.back_trans_models = {}
            for lang in config.BACK_TRANS_LANGUAGES:
                try:
                    # Model untuk en -> target language
                    forward_model_name = f'Helsinki-NLP/opus-mt-en-{lang}'
                    # Model untuk target language -> en
                    backward_model_name = f'Helsinki-NLP/opus-mt-{lang}-en'
                    
                    self.back_trans_models[lang] = {
                        'forward_tokenizer': MarianTokenizer.from_pretrained(
                            forward_model_name, cache_dir=config.CACHE_DIR
                        ),
                        'forward_model': MarianMTModel.from_pretrained(
                            forward_model_name, cache_dir=config.CACHE_DIR
                        ).to(self.device),
                        'backward_tokenizer': MarianTokenizer.from_pretrained(
                            backward_model_name, cache_dir=config.CACHE_DIR
                        ),
                        'backward_model': MarianMTModel.from_pretrained(
                            backward_model_name, cache_dir=config.CACHE_DIR
                        ).to(self.device)
                    }
                    logger.info(f"✓ Back Translation ready for: en <-> {lang}")
                except Exception as e:
                    logger.warning(f"Could not load translation models for {lang}: {e}")
            
            if not self.back_trans_models:
                logger.warning("No back translation models loaded!")
                config.USE_BACK_TRANSLATION = False
    
    def contextual_augment(self, text, num_replacements=None):
        """
        Contextual Augmentation: ganti random words dengan prediksi dari MLM
        
        Args:
            text: input text
            num_replacements: jumlah kata yang akan diganti (None = auto based on probability)
        """
        if not self.config.USE_CONTEXTUAL_AUG:
            return text
        
        try:
            words = text.split()
            if len(words) < 5:  # Skip teks terlalu pendek
                return text
            
            # Tentukan jumlah kata yang akan diganti
            if num_replacements is None:
                num_replacements = max(1, int(len(words) * self.config.CONTEXTUAL_AUG_PROB))
            
            # Pilih random positions untuk diganti
            positions_to_replace = random.sample(range(len(words)), 
                                                 min(num_replacements, len(words)))
            
            augmented_words = words.copy()
            for pos in positions_to_replace:
                # Skip jika kata terlalu pendek atau stopword common
                if len(words[pos]) <= 3 or words[pos].lower() in ['the', 'and', 'or', 'is', 'are']:
                    continue
                
                # Buat masked sentence
                masked_words = words.copy()
                masked_words[pos] = self.mlm_tokenizer.mask_token
                masked_text = ' '.join(masked_words)
                
                # Prediksi kata pengganti
                try:
                    predictions = self.mlm_pipeline(masked_text, top_k=3)
                    if predictions:
                        # Pilih random dari top-3 predictions
                        new_word = random.choice(predictions)['token_str'].strip()
                        augmented_words[pos] = new_word
                except:
                    continue
            
            return ' '.join(augmented_words)
        
        except Exception as e:
            logger.warning(f"Contextual augmentation failed: {e}")
            return text
    
    def back_translate(self, text, target_lang=None):
        """
        Back Translation: en -> target_lang -> en
        
        Args:
            text: input text
            target_lang: target language (None = random choice)
        """
        if not self.config.USE_BACK_TRANSLATION or not self.back_trans_models:
            return text
        
        try:
            # Pilih bahasa target
            if target_lang is None:
                target_lang = random.choice(list(self.back_trans_models.keys()))
            
            if target_lang not in self.back_trans_models:
                return text
            
            models = self.back_trans_models[target_lang]
            
            # Truncate teks jika terlalu panjang
            max_length = 500
            if len(text.split()) > max_length:
                words = text.split()[:max_length]
                text = ' '.join(words)
            
            # Forward translation: en -> target_lang
            forward_inputs = models['forward_tokenizer'](
                text, return_tensors='pt', padding=True, truncation=True, max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                forward_output = models['forward_model'].generate(**forward_inputs)
            
            translated_text = models['forward_tokenizer'].decode(
                forward_output[0], skip_special_tokens=True
            )
            
            # Backward translation: target_lang -> en
            backward_inputs = models['backward_tokenizer'](
                translated_text, return_tensors='pt', padding=True, truncation=True, max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                backward_output = models['backward_model'].generate(**backward_inputs)
            
            back_translated_text = models['backward_tokenizer'].decode(
                backward_output[0], skip_special_tokens=True
            )
            
            return back_translated_text
        
        except Exception as e:
            logger.warning(f"Back translation failed: {e}")
            return text
    
    def augment_text(self, text, method='both'):
        """
        Augment text dengan method yang dipilih
        
        Args:
            text: input text
            method: 'contextual', 'back_trans', atau 'both'
        """
        if method == 'contextual':
            return self.contextual_augment(text)
        elif method == 'back_trans':
            return self.back_translate(text)
        elif method == 'both':
            # Random pilih salah satu atau kombinasi
            choice = random.choice(['contextual', 'back_trans', 'combined'])
            if choice == 'contextual':
                return self.contextual_augment(text)
            elif choice == 'back_trans':
                return self.back_translate(text)
            else:  # combined
                text = self.contextual_augment(text)
                text = self.back_translate(text)
                return text
        else:
            return text
        
print("All augments method loaded successfully!")