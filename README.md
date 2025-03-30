# CIFAR-10 Classifier 

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

**Output:**
```
Training set: 45000 samples
Validation set: 5000 samples
Test set: 10000 samples
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
    epochs=100,
    batch_size=256
)
```

**Training Progress:**
```
Epoch   1/30 | Train Loss: 2.1528 | Val Loss: 2.1670 | Val Acc: 0.3222 | LR: 0.00295
Epoch   2/30 | Train Loss: 2.0489 | Val Loss: 2.0673 | Val Acc: 0.3498 | LR: 0.00280
...
Early stopping at epoch 23
Model saved to best_model.npy
```

```python
# █▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀▀█
# █    VISUALIZATION CELL      █
# █▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄▄█

from utils import AdvancedVisualizer

AdvancedVisualizer.plot_metrics(history)  # Saves to metrics.png
AdvancedVisualizer.plot_weights(model.params['W1'])  # Saves to weights.png
```

[metrics](https://github.com/user-attachments/assets/9fd58c2a-258e-4922-b432-7dac50bc8491)

[weights_vis](https://github.com/user-attachments/assets/d8d68129-886a-4f5f-ba11-45fa3dadc2ec)


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

**Output:**
```
Loaded model from best_model.npy
⭐ Final Test Accuracy: 52.15%
```
