# CIFAR-10 Image Classification with Neural Network

[![Python 3.6+](https://img.shields.io/badge/python-3.6+-blue.svg)](https://www.python.org/downloads/)

A three-layer neural network implementation for CIFAR-10 classification, featuring:

- 🚀 Training with SGD + Momentum
- 🛠 Hyperparameter search
- 📉 Learning rate scheduling
- 🛑 Early stopping
- 📊 Training visualization

## Project Structure
- model.py 
- train.py
- test.py 
- hyper_search.py 
- utils.py 
- complete_code.py 
- README.md

## Installation

1. Clone repository:
```bash
！git clone https://github.com/hank-aa11/cifar10-classifier.git
cd cifar10-classifier
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

### 1. Training the Model
```python
from utils import CIFAR10Processor
from model import NeuralNetwork
from train import SmartTrainer

processor = CIFAR10Processor('path/to/cifar-10-batches-py')
X_train, y_train, X_val, y_val, X_test, y_test = processor.load_split_data()

model = NeuralNetwork(3072, 512, 10)
trainer = SmartTrainer(model, X_val, y_val)
history = trainer.train(X_train, y_train, lr=0.001, epochs=100)
```
Basic training (with default parameters):
```bash
python train.py \
    --data_dir /path/to/cifar-10-batches-py \
    --save_model best_model.npy
```
Custom training:
```bash
python train.py \
    --data_dir /path/to/cifar-10-batches-py \
    --hidden_dim 1024 \
    --learning_rate 0.001 \
    --batch_size 256 \
    --reg_lambda 0.0001 \
    --patience 7 \
    --save_model my_model.npy \
    --epochs 30
```

### 2. Hyperparameter Search
```python
from hyper_search import HyperOptimizer
optimizer = HyperOptimizer(3072, 10)
best_params, best_acc = optimizer.random_search(X_train, y_train, X_val, y_val)
```
```bash
python hyper_search.py \
    --data_dir /path/to/cifar-10-batches-py \
    --n_trials 50 \
    --max_epochs 30 \
    --output_log hparam_results.log
```

### 3. Evaluating on Test Set
```bash
python test.py \
    --model_path best_model.npy \
    --data_dir /path/to/cifar-10-batches-py \
    --hidden_dim 512
```

### 4. Visualization
Training Curves
```python
import numpy as np
from utils import AdvancedVisualizer

history = np.load('training_history.npy', allow_pickle=True).item()
AdvancedVisualizer.plot_metrics(history)
```
Weight Visualization
```python
from model import NeuralNetwork
from utils import AdvancedVisualizer

model = NeuralNetwork(3072, 512, 10)
model.params = np.load('best_model.npy', allow_pickle=True).item()
AdvancedVisualizer.plot_weights(model.params['W1'])
```

### 5.Reproducibility
```python
import numpy as np
np.random.seed(42)  # Before any other imports
```



# CIFAR-10 Classifier Notebook Guide (vivid)

```python
# █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
# █        INSTALLATION          █
# █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█

!git clone https://github.com/hank-aa11/cifar10-classifier.git
!cd cifar10-classifier && pip install -r requirements.txt
```

```python
# █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
# █       DATA PREP CELL        █
# █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█

from utils import CIFAR10Processor

processor = CIFAR10Processor("/path/to/cifar-10-batches-py")
X_train, y_train, X_val, y_val, X_test, y_test = processor.load_split_data()

print(f"Training set: {X_train.shape[0]} samples")
print(f"Validation set: {X_val.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
```

```python
# █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
# █      TRAINING CELL         █
# █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█

from model import NeuralNetwork
from train import SmartTrainer

model = NeuralNetwork(
    input_dim=3072,
    hidden_dim=512,
    output_dim=10,
    reg_lambda=0.0001
)

trainer = SmartTrainer(model, X_val, y_val)
history = trainer.train(
    X_train, y_train,
    lr=0.001,
    epochs=30,
    batch_size=256
)
```


```python
# █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
# █    VISUALIZATION CELL      █
# █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█

from utils import AdvancedVisualizer

AdvancedVisualizer.plot_metrics(history)  # Saves to metrics.png
AdvancedVisualizer.plot_weights(model.params['W1'])  # Saves to weights.png
```

```python
# █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
# █      EVALUATION CELL       █
# █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█

from test import evaluate_test_set

test_acc = evaluate_test_set(
    model_path="best_model.npy",
    X_test=X_test,
    y_test=y_test,
    hidden_dim=512
)

print(f"\n⭐ Final Test Accuracy: {test_acc*100:.2f}%")
```
