import matplotlib.pyplot as plt
from model import NeuralNetwork
from train import SmartTrainer
from hyper_optimizer import HyperOptimizer

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
