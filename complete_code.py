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


# ==================== 神经网络核心模块 ====================
class NeuralNetwork:
    """三层神经网络"""
    def __init__(self,
                 input_dim: int,
                 hidden_dim: int,
                 output_dim: int,
                 activation: str = 'relu',
                 reg_lambda: float = 0.0,
                 init_scale: float = 0.01):
        # 初始化参数
        self.params = {
            'W1': np.random.randn(input_dim, hidden_dim) * self._init_scale(activation, input_dim),
            'b1': np.zeros(hidden_dim),
            'W2': np.random.randn(hidden_dim, output_dim) * init_scale,
            'b2': np.zeros(output_dim)
        }
        self.activation = activation
        self.reg_lambda = reg_lambda
        self.cache = {}

    def _init_scale(self, activation: str, fan_in: int) -> float:
        """自适应初始化策略"""
        if activation == 'relu':
            return np.sqrt(2.0 / fan_in)  # He 初始化
        elif activation == 'sigmoid':
            return np.sqrt(1.0 / fan_in)  # Xavier 初始化
        else:
            return 0.01

    def _activation(self, Z: np.ndarray) -> np.ndarray:
        """带数值裁剪的激活函数"""
        if self.activation == 'relu':
            return np.clip(Z, 0, None)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(Z, -50, 50)))  # 防止数值溢出
        raise ValueError(f"不支持的激活函数: {self.activation}")

    def _activation_grad(self, Z: np.ndarray) -> np.ndarray:
        """激活函数梯度计算"""
        if self.activation == 'relu':
            return (Z > 0).astype(float)
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-np.clip(Z, -50, 50)))
            return sig * (1 - sig)
        raise ValueError(f"不支持的激活函数: {self.activation}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        """带梯度裁剪的前向传播"""
        # 第一层
        self.cache['Z1'] = X @ self.params['W1'] + self.params['b1']
        self.cache['A1'] = self._activation(self.cache['Z1'])

        # 第二层
        self.cache['Z2'] = self.cache['A1'] @ self.params['W2'] + self.params['b2']
        
        # 数值稳定的Softmax
        max_Z = np.max(self.cache['Z2'], axis=1, keepdims=True)
        exp_Z = np.exp(self.cache['Z2'] - max_Z)
        self.cache['A2'] = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        
        return self.cache['A2']

    def backward(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """带梯度裁剪的反向传播"""
        m = X.shape[0]
        
        # 输出层梯度
        dZ2 = self.cache['A2'] - y
        dW2 = (self.cache['A1'].T @ dZ2) / m + self.reg_lambda * self.params['W2']
        db2 = np.sum(dZ2, axis=0) / m

        # 隐藏层梯度
        dZ1 = (dZ2 @ self.params['W2'].T) * self._activation_grad(self.cache['Z1'])
        dW1 = (X.T @ dZ1) / m + self.reg_lambda * self.params['W1']
        db1 = np.sum(dZ1, axis=0) / m

        return {'dW1': np.clip(dW1, -5, 5),  # 梯度裁剪防止爆炸
                'db1': np.clip(db1, -5, 5),
                'dW2': np.clip(dW2, -5, 5),
                'db2': np.clip(db2, -5, 5)}

    def compute_loss(self, y: np.ndarray) -> float:
        """带L2正则化的交叉熵损失"""
        m = y.shape[0]
        log_probs = -np.log(np.clip(self.cache['A2'][range(m), np.argmax(y, axis=1)], 1e-8, None))
        data_loss = np.sum(log_probs) / m
        reg_loss = 0.5 * self.reg_lambda * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
        return data_loss + reg_loss


# ==================== 训练模块 ====================
class SmartTrainer:
    """智能训练器（支持早停、学习率调度等）"""
    def __init__(self,
                 model: NeuralNetwork,
                 X_val: np.ndarray,
                 y_val: np.ndarray,
                 patience: int = 5,
                 save_path: str = 'best_model.npy'):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val
        self.patience = patience
        self.save_path = save_path
        self.best_acc = 0.0
        self.wait = 0
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }

    def _validate(self) -> Tuple[float, float]:
        """验证集评估"""
        probs = self.model.forward(self.X_val)
        preds = np.argmax(probs, axis=1)
        return self.model.compute_loss(self.y_val), np.mean(preds == np.argmax(self.y_val, axis=1))

    def train(self,
              X_train: np.ndarray,
              y_train: np.ndarray,
              lr: float = 0.01,
              epochs: int = 100,
              batch_size: int = 128,
              lr_decay: float = 0.95) -> Dict[str, List[float]]:
        
        m = X_train.shape[0]
        best_params = None
        
        for epoch in range(epochs):
            # 学习率调度
            current_lr = lr * (lr_decay ** epoch)
            
            # 分批次训练
            indices = np.random.permutation(m)
            for i in range(0, m, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
                
                # 前向传播
                self.model.forward(X_batch)
                
                # 反向传播
                grads = self.model.backward(X_batch, y_batch)
                
                # 参数更新
                for param in self.model.params:
                    self.model.params[param] -= current_lr * grads[f'd{param}']
            
            # 训练集损失
            self.model.forward(X_train)
            train_loss = self.model.compute_loss(y_train)
            
            # 验证集评估
            val_loss, val_acc = self._validate()
            
            # 早停机制
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                self.wait = 0
                best_params = {k: v.copy() for k, v in self.model.params.items()}
                np.save(self.save_path, best_params)
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_acc:.4f} | "
                  f"LR: {current_lr:.5f}")
        
        # 恢复最佳参数
        if best_params is not None:
            self.model.params = best_params
        
        return self.history


# ==================== 超参数模块 ====================
class HyperOptimizer:
    """并行化超参数优化器"""
    def __init__(self, input_dim: int, output_dim: int):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def random_search(self,
                      X_train: np.ndarray,
                      y_train: np.ndarray,
                      X_val: np.ndarray,
                      y_val: np.ndarray,
                      n_trials: int = 20,
                      epochs: int = 30) -> Tuple[Dict, float]:
        
        best_acc = 0.0
        best_params = {}
        
        for _ in range(n_trials):
            # 随机采样超参数
            hp = {
                'hidden_dim': np.random.choice([128, 256, 512, 1024]),
                'lr': 10**np.random.uniform(-4, -2),
                'reg': 10**np.random.uniform(-5, -1),
                'batch_size': np.random.choice([64, 128, 256])
            }
            
            print(f"\nTrial {_+1}/{n_trials}: {hp}")
            
            # 初始化模型
            model = NeuralNetwork(
                input_dim=self.input_dim,
                hidden_dim=hp['hidden_dim'],
                output_dim=self.output_dim,
                reg_lambda=hp['reg']
            )
            
            # 训练模型
            trainer = SmartTrainer(model, X_val, y_val, patience=3)
            history = trainer.train(
                X_train, y_train,
                lr=hp['lr'],
                epochs=epochs,
                batch_size=hp['batch_size']
            )
            
            # 记录最佳结果
            current_acc = max(history['val_acc'])
            if current_acc > best_acc:
                best_acc = current_acc
                best_params = hp
                print(f"New best accuracy: {best_acc:.4f}")
        
        print("\nHyperparameter search completed!")
        print(f"Best accuracy: {best_acc:.4f}")
        print("Best parameters:", best_params)
        return best_params, best_acc


# ==================== 可视化模块 ====================
class AdvancedVisualizer:
    """可视化工具"""
    @staticmethod
    def plot_weights(weights: np.ndarray, 
                    save_path: str = 'weights.png',
                    n_cols: int = 16) -> None:
        """权重可视化"""
        n_neurons = weights.shape[1]
        n_rows = int(np.ceil(n_neurons / n_cols))
        
        plt.figure(figsize=(n_cols, n_rows))
        for i in range(n_neurons):
            plt.subplot(n_rows, n_cols, i+1)
            plt.imshow((weights[:, i].reshape(32, 32, 3) * 255).astype('uint8'))
            plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()

    @staticmethod
    def plot_metrics(history: Dict[str, List[float]],
                    save_path: str = 'metrics.png') -> None:
        """训练指标可视化"""
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
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()


# ==================== 主程序 ====================
def main():
    # 初始化配置
    DATA_PATH = '/kaggle/input/cifar-10/cifar-10-batches-py/'
    SEED = 42
    np.random.seed(SEED)
    
    # 数据准备
    processor = CIFAR10Processor(DATA_PATH)
    X_train, y_train, X_val, y_val, X_test, y_test = processor.load_split_data()
    
    # 超参数优化
    optimizer = HyperOptimizer(input_dim=3072, output_dim=10)
    best_hp, best_acc = optimizer.random_search(X_train, y_train, X_val, y_val)
    
    # 最终训练
    final_model = NeuralNetwork(
        input_dim=3072,
        hidden_dim=best_hp['hidden_dim'],
        output_dim=10,
        reg_lambda=best_hp['reg']
    )
    
    trainer = SmartTrainer(final_model, X_val, y_val, patience=5)
    history = trainer.train(
        X_train, y_train,
        lr=best_hp['lr'],
        epochs=200,
        batch_size=best_hp['batch_size']
    )
    
    # 测试评估
    probs = final_model.forward(X_test)
    test_acc = np.mean(np.argmax(probs, axis=1) == np.argmax(y_test, axis=1))
    print(f"\nFinal Test Accuracy: {test_acc:.4f}")
    
    # 可视化结果
    AdvancedVisualizer.plot_metrics(history)
    AdvancedVisualizer.plot_weights(final_model.params['W1'])


if __name__ == "__main__":
    main()
