import numpy as np
from model import NeuralNetwork
from train import SmartTrainer

class HyperOptimizer:
    """超参数优化器"""
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim

    def random_search(self, X_train, y_train, X_val, y_val, n_trials=20, epochs=30):
        best_acc = 0.0
        best_params = {}
        
        for _ in range(n_trials):
            hp = {
                'hidden_dim': np.random.choice([128, 256, 512, 1024]),
                'lr': 10**np.random.uniform(-4, -2),
                'reg': 10**np.random.uniform(-5, -1),
                'batch_size': np.random.choice([64, 128, 256])
            }
            
            print(f"\nTrial {_+1}/{n_trials}: {hp}")
            model = NeuralNetwork(self.input_dim, hp['hidden_dim'], self.output_dim, reg_lambda=hp['reg'])
            trainer = SmartTrainer(model, X_val, y_val, patience=3)
            history = trainer.train(X_train, y_train, lr=hp['lr'], epochs=epochs, batch_size=hp['batch_size'])
            
            current_acc = max(history['val_acc'])
            if current_acc > best_acc:
                best_acc = current_acc
                best_params = hp
                print(f"New best accuracy: {best_acc:.4f}")
        
        print("\nHyperparameter search completed!")
        print(f"Best accuracy: {best_acc:.4f}")
        print("Best parameters:", best_params)
        return best_params, best_acc
