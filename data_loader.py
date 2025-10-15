import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class MedicalArticleDataLoader:
    """Load and preprocess data from JSONL file"""
    
    def __init__(self, jsonl_path):
        self.jsonl_path = jsonl_path
        self.label_encoder = LabelEncoder()
        self.data = None
        self.tabular_columns = [
            'completeness',
            'informativeness',
            'numHeadings',
            'articleLength',
            'numReferences/articleLength',
            'infoBoxNormSize',
            'category',
            'domainInformativeness'
        ]
    
    def load_data(self):
        """Load JSONL file into DataFrame"""
        records = []
        with open(self.jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        
        self.data = pd.DataFrame(records)
        return self.data
    
    def preprocess(self):
        """Preprocess data for model input"""
        if self.data is None:
            self.load_data()
        
        # Encode categorical 'category' column
        self.data['category_encoded'] = self.label_encoder.fit_transform(self.data['category'])
        
        # Extract text
        texts = self.data['text'].values
        
        # Extract tabular features
        tabular_features = self.data[self.tabular_columns].copy()
        tabular_features['category'] = self.data['category_encoded']
        tabular_features = tabular_features.values.astype(np.float32)
        
        # Extract labels
        labels = self.data['class'].values
        
        return texts, tabular_features, labels
    
    def get_class_distribution(self):
        """Get class distribution statistics"""
        if self.data is None:
            self.load_data()
        
        return self.data['class'].value_counts().sort_index()
    
    def get_info(self):
        """Get dataset information"""
        if self.data is None:
            self.load_data()
        
        info = {
            'total_samples': len(self.data),
            'num_classes': self.data['class'].nunique(),
            'classes': sorted(self.data['class'].unique()),
            'num_tabular_features': len(self.tabular_columns),
            'tabular_features': self.tabular_columns
        }
        return info
    
if __name__ == "__main__":
    loader = MedicalArticleDataLoader('final_dataset_fix_banget_real.jsonl')
    
    # Load data
    df = loader.load_data()
    
    # Print dataset length
    print(f"Dataset length: {len(df)}")
    print()
    
    # Print first 3 records
    print("First 3 records:")
    print(df.head(3))
    print()
    
    # Alternative: Print first 3 records with all columns visible
    print("First 3 records (detailed):")
    for i in range(min(3, len(df))):
        print(f"\n=== Record {i+1} ===")
        for col in df.columns:
            if col == 'text':
                print(f"{col}: {df.iloc[i][col][:200]}...")  # Truncate long text
            else:
                print(f"{col}: {df.iloc[i][col]}")
    
    # Get dataset info
    print("\n" + "="*50)
    info = loader.get_info()
    print("Dataset Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Get class distribution
    print("\nClass Distribution:")
    print(loader.get_class_distribution())
