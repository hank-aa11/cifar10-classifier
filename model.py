import numpy as np
from typing import Dict

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
        if activation == 'relu':
            return np.sqrt(2.0 / fan_in)
        elif activation == 'sigmoid':
            return np.sqrt(1.0 / fan_in)
        else:
            return 0.01

    def _activation(self, Z: np.ndarray) -> np.ndarray:
        if self.activation == 'relu':
            return np.clip(Z, 0, None)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(Z, -50, 50)))
        raise ValueError(f"Unsupported activation: {self.activation}")

    def _activation_grad(self, Z: np.ndarray) -> np.ndarray:
        if self.activation == 'relu':
            return (Z > 0).astype(float)
        elif self.activation == 'sigmoid':
            sig = 1 / (1 + np.exp(-np.clip(Z, -50, 50)))
            return sig * (1 - sig)
        raise ValueError(f"Unsupported activation: {self.activation}")

    def forward(self, X: np.ndarray) -> np.ndarray:
        self.cache['Z1'] = X @ self.params['W1'] + self.params['b1']
        self.cache['A1'] = self._activation(self.cache['Z1'])
        
        self.cache['Z2'] = self.cache['A1'] @ self.params['W2'] + self.params['b2']
        max_Z = np.max(self.cache['Z2'], axis=1, keepdims=True)
        exp_Z = np.exp(self.cache['Z2'] - max_Z)
        self.cache['A2'] = exp_Z / np.sum(exp_Z, axis=1, keepdims=True)
        return self.cache['A2']

    def backward(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        m = X.shape[0]
        dZ2 = self.cache['A2'] - y
        dW2 = (self.cache['A1'].T @ dZ2) / m + self.reg_lambda * self.params['W2']
        db2 = np.sum(dZ2, axis=0) / m

        dZ1 = (dZ2 @ self.params['W2'].T) * self._activation_grad(self.cache['Z1'])
        dW1 = (X.T @ dZ1) / m + self.reg_lambda * self.params['W1']
        db1 = np.sum(dZ1, axis=0) / m

        return {'dW1': np.clip(dW1, -5, 5),
                'db1': np.clip(db1, -5, 5),
                'dW2': np.clip(dW2, -5, 5),
                'db2': np.clip(db2, -5, 5)}

    def compute_loss(self, y: np.ndarray) -> float:
        m = y.shape[0]
        log_probs = -np.log(np.clip(self.cache['A2'][range(m), np.argmax(y, axis=1)], 1e-8, None))
        data_loss = np.sum(log_probs) / m
        reg_loss = 0.5 * self.reg_lambda * (np.sum(self.params['W1']**2) + np.sum(self.params['W2']**2))
        return data_loss + reg_loss
