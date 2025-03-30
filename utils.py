import numpy as np
import pickle
import os
import matplotlib.pyplot as plt

class CIFAR10Processor:
    """CIFAR-10 数据处理器"""
    def __init__(self, data_dir, val_size=5000):
        self.data_dir = data_dir
        self.val_size = val_size
        self.mean = None
        self.std = None

    def _load_batch(self, filename):
        with open(filename, 'rb') as f:
            batch = pickle.load(f, encoding='latin1')
        images = batch['data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        return images.reshape(-1, 3072), np.array(batch['labels'])

    def _zscore_normalize(self, data):
        if self.mean is None or self.std is None:
            self.mean = data.mean(axis=0)
            self.std = data.std(axis=0) + 1e-8
        return (data - self.mean) / self.std

    def load_split_data(self):
        X_train, y_train = [], []
        for i in range(1, 6):
            X, y = self._load_batch(os.path.join(self.data_dir, f'data_batch_{i}'))
            X_train.append(X)
            y_train.append(y)
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)

        X_test, y_test = self._load_batch(os.path.join(self.data_dir, 'test_batch'))

        X_train = self._zscore_normalize(X_train)
        X_test = self._zscore_normalize(X_test)

        indices = np.random.permutation(X_train.shape[0])
        X_val, y_val = X_train[:self.val_size], y_train[:self.val_size]
        X_train, y_train = X_train[self.val_size:], y_train[self.val_size:]

        y_train = np.eye(10)[y_train]
        y_val = np.eye(10)[y_val]
        y_test = np.eye(10)[y_test]

        return X_train, y_train, X_val, y_val, X_test, y_test

class AdvancedVisualizer:
    @staticmethod
    def plot_metrics(history, save_path='metrics.png'):
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train')
        plt.plot(history['val_loss'], label='Validation')
        plt.title('Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['val_acc'])
        plt.title('Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.savefig(save_path)
        plt.close()
