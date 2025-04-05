import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional

# ==================== 数据预处理模块 ====================
class CIFAR10Processor:
    """ CIFAR-10 数据处理器"""
    def __init__(self, data_dir: str, val_size: int = 5000):
        self.data_dir = data_dir
        self.val_size = val_size
        self.mean = None
        self.std = None

    def _load_batch(self, filename: str) -> Tuple[np.ndarray, np.ndarray]:
        """加载并规范化单个批次数据"""
        with open(filename, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
        images = batch['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return images.reshape(-1, 3072), np.array(batch['labels'])

    def _zscore_normalize(self, data: np.ndarray) -> np.ndarray:
        """Z-Score 归一化（提升数值稳定性）"""
        if self.mean is None or self.std is None:
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0) + 1e-8  # 防止除零
        return (data - self.mean) / self.std

    def load_split_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """加载并分割数据集"""
        # 加载训练数据
        X_train, y_train = [], []
        for i in range(1, 6):
            X, y = self._load_batch(os.path.join(self.data_dir, f'data_batch_{i}'))
            X_train.append(X)
            y_train.append(y)
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)

        # 加载测试数据
        X_test, y_test = self._load_batch(os.path.join(self.data_dir, 'test_batch'))

        # 归一化处理
        X_train = self._zscore_normalize(X_train)
        X_test = self._zscore_normalize(X_test)

        # 划分验证集
        indices = np.random.permutation(X_train.shape[0])
        X_train, y_train = X_train[indices], y_train[indices]
        X_val, y_val = X_train[:self.val_size], y_train[:self.val_size]
        X_train, y_train = X_train[self.val_size:], y_train[self.val_size:]

        # One-Hot 编码
        y_train = np.eye(10)[y_train]
        y_val = np.eye(10)[y_val]
        y_test = np.eye(10)[y_test]

        return X_train, y_train, X_val, y_val, X_test, y_test
