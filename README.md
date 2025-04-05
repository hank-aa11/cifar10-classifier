# CIFAR-10 Image Classification with Neural Network

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)

A three-layer neural network implementation for CIFAR-10 classification, featuring:

- 🚀 Training with SGD + Momentum
- 🛠 Hyperparameter search
- 📉 Learning rate scheduling
- 🛑 Early stopping
- 📊 Training visualization

## Project Structure
- data_processor.py
- model.py 
- train.py
- hyper_optimizer.py
- visualizer.py
- complete_code.py 
- README.md

## Installation

1. Clone repository:
```bash
!git clone https://github.com/hank-aa11/cifar10-classifier.git
```
2. Install dependencies:
```bash
pip install numpy matplotlib
```

## Dataset Preparation
1. Download CIFAR-10 dataset from official website

2. Extract files to maintain this structure:
- data_batch_1
- data_batch_2
- data_batch_3
- data_batch_4
- data_batch_5
- test_batch

## Usage
You can run the entire code with complete_code.py, or you can run it step by step using other modularized codes as follows.
```python
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional

from cifar10_classifier.data_processor import CIFAR10Processor
from cifar10_classifier.model import NeuralNetwork
from cifar10_classifier.trainer import SmartTrainer
from cifar10_classifier.hyper_optimizer import HyperOptimizer
from cifar10_classifier.visualizer import AdvancedVisualizer

def main():
    # 初始化配置
    DATA_PATH = 'path/to/cifar-10-batches-py/'#cifar-10-batches-py在您电脑上的位置
    SEED = 42 #任意正整数
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
```
