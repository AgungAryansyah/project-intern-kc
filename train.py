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

class Trainer:
    """Main trainer class"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.DEVICE
        
        # Setup tokenizer dan model
        logger.info(f"Loading BioBERT tokenizer and model: {config.MODEL_NAME}")
        self.tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
        self.model = BioBERTMultiInputClassifier(config).to(self.device)
        
        # Setup Text Augmenter
        logger.info("Initializing Text Augmenter...")
        self.text_augmenter = TextAugmenter(config)
        
        # Training components
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        
        # Logging
        self.train_logger = TrainingLogger(
            os.path.join(config.OUTPUT_DIR, config.LOG_FILE)
        )
        
        self.best_val_loss = float('inf')
        
        logger.info(f"Model initialized. Using device: {self.device}")
        logger.info(f"Total parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def augment_minority_classes(self, texts, tabular_features, labels, label_encoder):
        """
        Augment text untuk minority classes menggunakan Contextual Augmentation & Back Translation
        """
        logger.info("Starting text augmentation for minority classes...")
        
        augmented_texts = list(texts)
        augmented_tabular = list(tabular_features)
        augmented_labels = list(labels)
        
        # Decode labels untuk identifikasi class
        label_names = label_encoder.inverse_transform(labels)
        
        # Track augmentation statistics
        aug_stats = {cls: 0 for cls in self.config.AUG_TARGET_CLASSES}
        
        for i, (text, tab_feat, label, label_name) in enumerate(
            tqdm(zip(texts, tabular_features, labels, label_names), 
                 desc="Augmenting texts", total=len(texts))
        ):
            # Hanya augment jika termasuk target class
            if label_name in self.config.AUG_TARGET_CLASSES:
                multiplier = self.config.AUG_MULTIPLIER.get(label_name, 1)
                
                for _ in range(multiplier):
                    # Pilih random augmentation method
                    aug_method = random.choice(['contextual', 'back_trans', 'both'])
                    
                    # Augment text
                    aug_text = self.text_augmenter.augment_text(text, method=aug_method)
                    
                    # Add augmented sample
                    augmented_texts.append(aug_text)
                    augmented_tabular.append(tab_feat)
                    augmented_labels.append(label)
                    
                    aug_stats[label_name] += 1
        
        logger.info("Text augmentation completed!")
        logger.info("Augmentation statistics:")
        for cls, count in aug_stats.items():
            logger.info(f"  {cls}: +{count} augmented samples")
        
        return (np.array(augmented_texts), 
                np.array(augmented_tabular), 
                np.array(augmented_labels))
    
    def prepare_data(self, df, text_column, tabular_columns, label_column):
        """
        Prepare data dengan augmentation dan preprocessing
        
        Args:
            df: DataFrame dengan columns untuk text, tabular features, dan labels
            text_column: nama kolom untuk text artikel
            tabular_columns: list nama kolom untuk fitur tabular
            label_column: nama kolom untuk quality label
        """
        logger.info("Preparing data...")
        
        # Extract features
        texts = df[text_column].values
        tabular_features = df[tabular_columns].values
        labels = df[label_column].values
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        labels_encoded = self.label_encoder.fit_transform(labels)
        
        logger.info(f"Original dataset size: {len(texts)}")
        logger.info(f"Classes: {self.label_encoder.classes_}")
        
        # Scale tabular features
        self.scaler = StandardScaler()
        tabular_features_scaled = self.scaler.fit_transform(tabular_features)
        
        # Train-val split SEBELUM augmentation
        X_text_train, X_text_val, X_tab_train, X_tab_val, y_train, y_val = train_test_split(
            texts, tabular_features_scaled, labels_encoded,
            test_size=0.2, random_state=self.config.SEED, stratify=labels_encoded
        )
        
        logger.info(f"Train set size (before augmentation): {len(y_train)}")
        logger.info(f"Validation set size: {len(y_val)}")
        
        # TEXT AUGMENTATION untuk training set (minority classes)
        if self.config.USE_CONTEXTUAL_AUG or self.config.USE_BACK_TRANSLATION:
            X_text_train, X_tab_train, y_train = self.augment_minority_classes(
                X_text_train, X_tab_train, y_train, self.label_encoder
            )
            logger.info(f"Train set size (after text augmentation): {len(y_train)}")
        
        # SMOTE untuk tabular features (optional, sebagai pelengkap)
        if self.config.USE_SMOTE:
            logger.info("Applying SMOTE for tabular features balancing...")
            try:
                # Combine untuk SMOTE
                X_combined = np.column_stack([
                    np.arange(len(X_text_train)),  # Index untuk text
                    X_tab_train
                ])
                
                smote = SMOTE(random_state=self.config.SEED, k_neighbors=3)
                X_resampled, y_resampled = smote.fit_resample(X_combined, y_train)
                
                # Extract kembali
                text_indices = X_resampled[:, 0].astype(int)
                text_indices = np.clip(text_indices, 0, len(X_text_train) - 1)
                
                X_text_train = X_text_train[text_indices]
                X_tab_train = X_resampled[:, 1:]
                y_train = y_resampled
                
                logger.info(f"After SMOTE - Training samples: {len(y_train)}")
            except Exception as e:
                logger.warning(f"SMOTE failed: {e}. Continuing without SMOTE.")
        
        # Create datasets
        train_dataset = MedicalArticleDataset(
            X_text_train, X_tab_train, y_train,
            self.tokenizer, self.config.MAX_LENGTH
        )
        
        val_dataset = MedicalArticleDataset(
            X_text_val, X_tab_val, y_val,
            self.tokenizer, self.config.MAX_LENGTH
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            num_workers=2
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=False,
            num_workers=2
        )
        
        # Setup optimizer dan scheduler
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        total_steps = len(self.train_loader) * self.config.EPOCHS
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.WARMUP_STEPS,
            num_training_steps=total_steps
        )
        
        logger.info(f"Data preparation complete!")
        logger.info(f"Final training samples: {len(train_dataset)}")
        logger.info(f"Validation samples: {len(val_dataset)}")
        
        # Print class distribution
        train_labels = self.label_encoder.inverse_transform(y_train)
        val_labels = self.label_encoder.inverse_transform(y_val)
        
        logger.info("\nTraining set class distribution:")
        for cls in self.label_encoder.classes_:
            count = np.sum(train_labels == cls)
            logger.info(f"  {cls}: {count} samples ({count/len(train_labels)*100:.1f}%)")
        
        logger.info("\nValidation set class distribution:")
        for cls in self.label_encoder.classes_:
            count = np.sum(val_labels == cls)
            logger.info(f"  {cls}: {count} samples ({count/len(val_labels)*100:.1f}%)")
    
    def train_epoch(self):
        """Train satu epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            tabular_features = batch['tabular_features'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(input_ids, attention_mask, tabular_features)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate model"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                tabular_features = batch['tabular_features'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, tabular_features)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                
                # Get predictions
                probs = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        # Calculate metrics
        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100 * np.sum(np.array(all_preds) == np.array(all_labels)) / len(all_labels)
        
        # F1 Score (macro)
        f1_macro = f1_score(all_labels, all_preds, average='macro')
        
        # ROC AUC (macro, one-vs-rest)
        try:
            roc_auc_macro = roc_auc_score(
                all_labels, all_probs,
                multi_class='ovr', average='macro'
            )
        except:
            roc_auc_macro = 0.0
        
        return avg_loss, accuracy, f1_macro, roc_auc_macro, all_labels, all_preds
    
    def train(self):
        """Main training loop"""
        logger.info("=" * 80)
        logger.info("Starting training...")
        logger.info("=" * 80)
        
        for epoch in range(1, self.config.EPOCHS + 1):
            logger.info(f"\nEpoch {epoch}/{self.config.EPOCHS}")
            logger.info("-" * 80)
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_f1, val_roc_auc, val_labels, val_preds = self.validate()
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Log metrics
            logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
            logger.info(f"Val F1-Macro: {val_f1:.4f} | Val ROC-AUC: {val_roc_auc:.4f}")
            logger.info(f"Learning Rate: {current_lr:.2e}")
            
            # Save to logger
            self.train_logger.log_epoch(
                epoch, train_loss, train_acc, val_loss, val_acc,
                val_f1, val_roc_auc, current_lr
            )
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_model(epoch, val_loss, val_acc, val_f1, val_roc_auc)
                logger.info(f"✓ Best model saved! (Val Loss: {val_loss:.4f})")
            
            # Print detailed classification report setiap 5 epoch
            if epoch % 5 == 0:
                logger.info("\nDetailed Classification Report:")
                report = classification