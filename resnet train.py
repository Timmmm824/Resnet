import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.transforms import transforms
import numpy as np
import pandas as pd
from glob import glob
import os
from PIL import Image
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
import joblib  # 用于保存模型
import warnings

warnings.filterwarnings('ignore')


class ResNetFeatureExtractor:
    def __init__(self):
        self.model = resnet18(pretrained=True)
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.eval()

        self.preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def extract(self, image_path):
        try:
            image = Image.open(image_path).convert('RGB')
            image_tensor = self.preprocess(image).unsqueeze(0)
            with torch.no_grad():
                features = self.model(image_tensor)
            return features.squeeze().numpy()
        except Exception as e:
            print(f"Error extracting features from {image_path}: {str(e)}")
            return np.zeros(512)


class AdvancedFeatureSystem:
    def __init__(self, class_config):
        self.class_config = class_config
        self.class_names = list(class_config.keys())
        self.feature_extractor = ResNetFeatureExtractor()
        self.model = None
        self.required_geo_dim = None

    def _validate_features(self, img_feature, geo_feature):
        if len(img_feature) != 512:
            img_feature = np.zeros(512)
            print("Warning: Image feature dimension incorrect, using zero vector")

        if self.required_geo_dim is None:
            self.required_geo_dim = len(geo_feature)
            print(f"Setting geometric feature dimension to: {self.required_geo_dim}")

        if len(geo_feature) < self.required_geo_dim:
            geo_feature = np.pad(geo_feature, (0, self.required_geo_dim - len(geo_feature)))
        elif len(geo_feature) > self.required_geo_dim:
            geo_feature = geo_feature[:self.required_geo_dim]

        return img_feature, geo_feature

    def _load_geo_features(self, class_name):
        geo_file = self.class_config[class_name]["geo_file"]
        try:
            if geo_file.endswith('.xlsx') or geo_file.endswith('.xls'):
                df = pd.read_excel(geo_file)
            else:
                try:
                    df = pd.read_csv(geo_file, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(geo_file, encoding='gbk')
            return np.nan_to_num(df.values.flatten())
        except Exception as e:
            print(f"Error loading geometric features from {geo_file}: {str(e)}")
            return np.zeros(self.required_geo_dim) if self.required_geo_dim else np.zeros(10)

    def prepare_training_data(self):
        X = []
        y = []

        geo_dims = []
        for class_name in self.class_names:
            try:
                geo_file = self.class_config[class_name]["geo_file"]
                if geo_file.endswith('.xlsx') or geo_file.endswith('.xls'):
                    df = pd.read_excel(geo_file)
                else:
                    try:
                        df = pd.read_csv(geo_file, encoding='utf-8')
                    except UnicodeDecodeError:
                        df = pd.read_csv(geo_file, encoding='gbk')
                geo_dims.append(len(df.values.flatten()))
            except:
                geo_dims.append(0)

        self.required_geo_dim = max(geo_dims) if geo_dims else 10
        print(f"统一几何特征维度为: {self.required_geo_dim}")

        for class_idx, class_name in enumerate(self.class_names):
            print(f"\nProcessing class: {class_name}")

            image_dir = self.class_config[class_name]["image_dir"]
            if not os.path.exists(image_dir):
                raise FileNotFoundError(f"Image directory not found: {image_dir}")

            image_paths = []
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff']:
                image_paths.extend(glob(os.path.join(image_dir, ext)))

            if not image_paths:
                raise ValueError(f"No images found in {image_dir}")

            img_features = []
            for img_path in image_paths:
                features = self.feature_extractor.extract(img_path)
                if features is not None:
                    img_features.append(features)

            if not img_features:
                raise ValueError(f"No valid features extracted for {class_name}")

            fused_img_feature = np.mean(img_features, axis=0)
            geo_feature = self._load_geo_features(class_name)
            img_feature, geo_feature = self._validate_features(fused_img_feature, geo_feature)
            combined_feature = np.concatenate([img_feature, geo_feature])
            X.append(combined_feature)
            y.append(class_idx)

        self.X_train = np.vstack(X)
        self.y_train = np.array(y)

        if np.isnan(self.X_train).any():
            print("Warning: NaN values detected in training data, replacing with 0")
            self.X_train = np.nan_to_num(self.X_train)

        return self.X_train, self.y_train

    def train_model(self):
        if not hasattr(self, 'X_train'):
            raise RuntimeError("Must call prepare_training_data() first")

        self.model = make_pipeline(
            SimpleImputer(strategy='constant', fill_value=0),
            StandardScaler(),
            RandomForestClassifier(n_estimators=100, random_state=42)
        )
        self.model.fit(self.X_train, self.y_train)
        return self.model

    def save_model(self, save_path):
        """保存模型到指定路径"""
        if not hasattr(self, 'model'):
            raise RuntimeError("Model not trained yet")
        # 确保保存目录存在
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        joblib.dump(self.model, save_path)
        print(f"Model saved to {save_path}")


if __name__ == "__main__":
    # 配置参数
    CLASS_CONFIG = {
        "class1": {
            "image_dir": r'C:\Users\Administrator\Desktop\data\class1\images',
            "geo_file": r'C:\Users\Administrator\Desktop\data\class1\geo_features.xlsx'
        },
        "class2": {
            "image_dir": r'C:\Users\Administrator\Desktop\data\class2\images',
            "geo_file": r'C:\Users\Administrator\Desktop\data\class2\geo_features.xlsx'
        },
        "class3": {
            "image_dir": r'C:\Users\Administrator\Desktop\data\class3\images',
            "geo_file": r'C:\Users\Administrator\Desktop\data\class3\geo_features.xlsx'
        },
        "class4": {
            "image_dir": r'C:\Users\Administrator\Desktop\data\class4\images',
            "geo_file": r'C:\Users\Administrator\Desktop\data\class4\geo_features.xlsx'
        },
        "class5": {
            "image_dir": r'C:\Users\Administrator\Desktop\data\class5\images',
            "geo_file": r'C:\Users\Administrator\Desktop\data\class5\geo_features.xlsx'
        }
    }

    # 初始化系统
    print("Initializing feature system...")
    feature_system = AdvancedFeatureSystem(CLASS_CONFIG)

    # 准备训练数据
    print("\nPreparing training data...")
    try:
        X_train, y_train = feature_system.prepare_training_data()
        print(f"\nTraining data prepared successfully!")
        print(f"Training data shape: {X_train.shape}")
        print(f"NaN values in X_train: {np.isnan(X_train).sum()}")
    except Exception as e:
        print(f"\nFailed to prepare training data: {str(e)}")
        exit()

    # 训练模型
    print("\nTraining model...")
    try:
        feature_system.train_model()
        print("Model trained successfully!")

        # 保存模型到桌面work文件夹
        save_path = r'C:\Users\Administrator\Desktop\work\trained_model.joblib'
        feature_system.save_model(save_path)

    except Exception as e:
        print(f"Failed to train model: {str(e)}")
        exit()

    print("\nTraining process completed successfully!")