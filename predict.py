import torch
import torch.nn as nn
from torchvision.models import resnet18
from torchvision.transforms import transforms
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import warnings
import os
from sklearn.metrics.pairwise import cosine_similarity

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
            return np.zeros(512)  # ResNet18默认输出512维


class ImageClassifier:
    def __init__(self, model_path):
        """加载训练好的模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        self.model = joblib.load(model_path)
        self.feature_extractor = ResNetFeatureExtractor()
        self.expected_feature_dim = 449072  # 根据训练时确定的维度

        # 这里应该加载你的v1和v2数据集特征
        # 示例: self.v1_features = [加载v1数据集的特征]
        # self.v2_features = [加载v2数据集的特征]
        # 实际使用时需要替换为你的数据集加载代码
        self.v1_features = None
        self.v2_features = None

    def _pad_features(self, features):
        """填充或截断特征到预期维度"""
        if len(features) < self.expected_feature_dim:
            return np.pad(features, (0, self.expected_feature_dim - len(features)))
        elif len(features) > self.expected_feature_dim:
            return features[:self.expected_feature_dim]
        return features

    def _calculate_similarity(self, feature, dataset_features):
        """计算当前特征与数据集特征的相似度"""
        if dataset_features is None or len(dataset_features) == 0:
            return 1.0  # 如果没有数据集，返回默认相似度

        # 使用余弦相似度
        similarities = cosine_similarity([feature], dataset_features)
        return np.mean(similarities)

    def predict_image(self, image_path, geo_feature_path):
        """
        预测单张图像的类别概率
        :return: 字典格式的类别概率 {class_name: probability} 和最终分类结果
        """
        # 1. 提取图像特征 (512维)
        img_feature = self.feature_extractor.extract(image_path)

        # 2. 加载几何特征
        try:
            if geo_feature_path.endswith('.xlsx'):
                geo_df = pd.read_excel(geo_feature_path)
            else:
                geo_df = pd.read_csv(geo_feature_path)
            geo_feature = np.nan_to_num(geo_df.values.flatten())
        except Exception as e:
            print(f"Error loading geometric features: {str(e)}")
            geo_feature = np.zeros(self.expected_feature_dim - 512)  # 补齐剩余维度

        # 3. 合并并调整特征维度
        combined = np.concatenate([img_feature, geo_feature])
        combined = self._pad_features(combined).reshape(1, -1)

        # 4. 获取原始概率
        probas = self.model.predict_proba(combined)[0]
        class_probs = {f"class{i + 1}": p for i, p in enumerate(probas)}

        # 5. 找出v1和v2 (最大和次大概率对应的类别)
        sorted_classes = sorted(class_probs.items(), key=lambda x: x[1], reverse=True)
        v1_class, v1_prob = sorted_classes[0]
        v2_class, v2_prob = sorted_classes[1] if len(sorted_classes) > 1 else (None, 0)

        # 6. 计算相似度 (这里需要你有v1和v2的数据集特征)
        # 示例: 假设我们只使用图像特征部分进行比较
        img_feature_reshaped = img_feature.reshape(1, -1)

        # 计算P1和P2 (相似度加权)
        # 注意: 这里需要你实现自己的相似度计算逻辑
        P1 = self._calculate_similarity(img_feature, self.v1_features) if self.v1_features is not None else 1.0
        P2 = self._calculate_similarity(img_feature, self.v2_features) if self.v2_features is not None else 1.0

        # 7. 计算最终分类结果
        final_result = f"{v1_class}×{P1:.2f} + {v2_class}×{P2:.2f}" if v2_class else v1_class

        return {
            "class_probabilities": {k: f"{v:.2%}" for k, v in class_probs.items()},
            "top_classes": {
                "v1": v1_class,
                "v1_prob": f"{v1_prob:.2%}",
                "v2": v2_class,
                "v2_prob": f"{v2_prob:.2%}" if v2_class else "0%"
            },
            "similarity_factors": {
                "P1": f"{P1:.2f}",
                "P2": f"{P2:.2f}" if v2_class else "0"
            },
            "final_result": final_result
        }


def main():
    # 配置路径（修改为您的实际路径）
    MODEL_PATH = r'C:\Users\Administrator\Desktop\work\trained_model.joblib'
    IMAGE_PATH = r'C:\Users\Administrator\Desktop\test\1.jpg'
    GEO_FEATURE_PATH = r'C:\Users\Administrator\Desktop\test\1.xlsx'

    # 初始化分类器
    print("Loading classifier...")
    try:
        classifier = ImageClassifier(MODEL_PATH)
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Failed to load model: {str(e)}")
        return

    # 执行预测
    print("\nStarting prediction...")
    print(f"Image: {IMAGE_PATH}")
    print(f"Geometric features: {GEO_FEATURE_PATH}")

    try:
        results = classifier.predict_image(IMAGE_PATH, GEO_FEATURE_PATH)

        print("\nPrediction Results:")
        print("Class Probabilities:")
        for class_name, prob in results["class_probabilities"].items():
            print(f"{class_name}: {prob}")

        print("\nTop Classes:")
        print(f"v1 (highest probability): {results['top_classes']['v1']} ({results['top_classes']['v1_prob']})")
        print(f"v2 (second highest): {results['top_classes']['v2']} ({results['top_classes']['v2_prob']})")

        print("\nSimilarity Factors:")
        print(f"P1 (similarity to v1 dataset): {results['similarity_factors']['P1']}")
        print(f"P2 (similarity to v2 dataset): {results['similarity_factors']['P2']}")

        print(f"\nFinal Classification Result: {results['final_result']}")

    except Exception as e:
        print(f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    main()