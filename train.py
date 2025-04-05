# ==================== 训练优化模块 ====================
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
