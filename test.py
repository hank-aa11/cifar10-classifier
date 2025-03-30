import numpy as np
from model import NeuralNetwork

def evaluate_test_set(model_path, X_test, y_test, input_dim=3072, hidden_dim=512, output_dim=10):
    model = NeuralNetwork(input_dim, hidden_dim, output_dim)
    model.params = np.load(model_path, allow_pickle=True).item()
    probs = model.forward(X_test)
    test_acc = np.mean(np.argmax(probs, axis=1) == np.argmax(y_test, axis=1))
    print(f"Test Accuracy: {test_acc:.4f}")
    return test_acc

# 示例用法
if __name__ == "__main__":
    from utils import CIFAR10Processor
    processor = CIFAR10Processor('/path/to/cifar-10-batches-py')
    *_, X_test, y_test = processor.load_split_data()
    evaluate_test_set('best_model.npy', X_test, y_test)
