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
import random
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
        Limited augmentation - only generate specified number of samples per class
        """
        logger.info("Starting text augmentation for minority classes...")
        
        augmented_texts = []
        augmented_tabular = []
        augmented_labels = []
        
        # Decode labels untuk identifikasi class
        label_names = label_encoder.inverse_transform(labels)
        
        # Track augmentation statistics
        aug_stats = {cls: 0 for cls in self.config.AUG_TARGET_CLASSES}
        
        # Collect minority class samples
        minority_samples = {cls: [] for cls in self.config.AUG_TARGET_CLASSES}
        
        for i, (text, tab_feat, label, label_name) in enumerate(zip(texts, tabular_features, labels, label_names)):
            if label_name in self.config.AUG_TARGET_CLASSES:
                minority_samples[label_name].append((text, tab_feat, label))
        
        # Generate limited augmented samples per class
        for cls in self.config.AUG_TARGET_CLASSES:
            samples = minority_samples[cls]
            if len(samples) == 0:
                continue
            
            max_aug = min(self.config.AUG_SAMPLES_PER_CLASS, len(samples))
            logger.info(f"Augmenting {max_aug} samples for class {cls}")
            
            # Randomly select samples to augment
            selected_indices = np.random.choice(len(samples), size=max_aug, replace=False)
            
            for idx in tqdm(selected_indices, desc=f"Augmenting {cls}"):
                text, tab_feat, label = samples[idx]
                
                # Pilih random augmentation method
                aug_method = 'contextual' if self.config.USE_CONTEXTUAL_AUG else 'back_trans'
                
                # Augment text
                aug_text = self.text_augmenter.augment_text(text, method=aug_method)
                
                # Add augmented sample
                augmented_texts.append(aug_text)
                augmented_tabular.append(tab_feat)
                augmented_labels.append(label)
                
                aug_stats[cls] += 1
        
        logger.info("Text augmentation completed!")
        logger.info("Augmentation statistics:")
        for cls, count in aug_stats.items():
            logger.info(f"  {cls}: +{count} augmented samples")
        
        # Combine original + augmented
        all_texts = np.concatenate([texts, np.array(augmented_texts)])
        all_tabular = np.vstack([tabular_features, np.array(augmented_tabular)])
        all_labels = np.concatenate([labels, np.array(augmented_labels)])
        
        return all_texts, all_tabular, all_labels
    
    def prepare_data(self, texts, tabular_features, labels):
        """
        Prepare data dengan augmentation dan preprocessing
        
        Args:
            texts: numpy array of text strings
            tabular_features: numpy array of tabular features
            labels: numpy array of class labels
        """
        logger.info("Preparing data...")
        
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
                report = classification_report(
                    val_labels, val_preds,
                    target_names=self.label_encoder.classes_,
                    zero_division=0
                )
                logger.info(f"\n{report}")
        
        logger.info("\n" + "=" * 80)
        logger.info("Training completed!")
        logger.info("=" * 80)
    
    def save_model(self, epoch, val_loss, val_acc, val_f1, val_roc_auc):
        """Save model checkpoint"""
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'val_acc': val_acc,
            'val_f1': val_f1,
            'val_roc_auc': val_roc_auc,
            'label_encoder': self.label_encoder,
            'scaler': self.scaler,
            'config': self.config
        }
        
        save_path = os.path.join(self.config.OUTPUT_DIR, self.config.MODEL_SAVE_PATH)
        torch.save(checkpoint, save_path)


if __name__ == "__main__":
    from config import Config, logger
    from data_loader import MedicalArticleDataLoader
    from architecture import BioBERTMultiInputClassifier
    from dataset import MedicalArticleDataset
    from augment import TextAugmenter
    from trainer_log import TrainingLogger
    
    # Set random seed
    torch.manual_seed(Config.SEED)
    np.random.seed(Config.SEED)
    
    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)
    
    logger.info("=" * 80)
    logger.info("BioBERT Medical Article Quality Classification")
    logger.info("=" * 80)
    
    # Load configuration
    config = Config()
    logger.info(f"Device: {config.DEVICE}")
    logger.info(f"Model: {config.MODEL_NAME}")
    logger.info(f"Batch size: {config.BATCH_SIZE}")
    logger.info(f"Epochs: {config.EPOCHS}")
    logger.info(f"Learning rate: {config.LEARNING_RATE}")
    
    # Load data
    logger.info("\n" + "=" * 80)
    logger.info("Loading data...")
    logger.info("=" * 80)
    
    loader = MedicalArticleDataLoader('final_dataset_fix_banget_real.jsonl')
    texts, tabular_features, labels = loader.preprocess()
    
    # Display dataset info
    info = loader.get_info()
    logger.info(f"Total samples: {info['total_samples']}")
    logger.info(f"Number of classes: {info['num_classes']}")
    logger.info(f"Classes: {info['classes']}")
    logger.info(f"Tabular features ({info['num_tabular_features']}): {info['tabular_features']}")
    
    # Display class distribution
    logger.info("\nClass distribution:")
    class_dist = loader.get_class_distribution()
    for cls, count in class_dist.items():
        logger.info(f"  {cls}: {count} ({count/len(texts)*100:.2f}%)")
    
    # Create trainer
    logger.info("\n" + "=" * 80)
    logger.info("Initializing trainer...")
    logger.info("=" * 80)
    
    trainer = Trainer(config)
    
    # Prepare data
    logger.info("\n" + "=" * 80)
    logger.info("Preparing data...")
    logger.info("=" * 80)
    
    trainer.prepare_data(texts, tabular_features, labels)
    
    # Train model
    trainer.train()
    
    logger.info("\n" + "=" * 80)
    logger.info("Training pipeline completed successfully!")
    logger.info(f"Best model saved to: {os.path.join(config.OUTPUT_DIR, config.MODEL_SAVE_PATH)}")
    logger.info(f"Training log saved to: {os.path.join(config.OUTPUT_DIR, config.LOG_FILE)}")
    logger.info("=" * 80)