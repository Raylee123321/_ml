import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

from model import TelemetryAnalyzer

class TelemetryDataset(Dataset):
    def __init__(self, data_dir):
        self.X = torch.from_numpy(np.load(os.path.join(data_dir, 'X.npy')))
        self.Y_cls = torch.from_numpy(np.load(os.path.join(data_dir, 'Y_cls.npy')))
        self.Y_reg = torch.from_numpy(np.load(os.path.join(data_dir, 'Y_reg.npy')))
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.Y_cls[idx], self.Y_reg[idx]

def train_model(data_dir, model_save_path, epochs=80, batch_size=16, lr=0.001):
    # 確保資料存在
    if not os.path.exists(os.path.join(data_dir, 'X.npy')):
        raise FileNotFoundError(f"找不到遙測數據，請先運行 data_prep.py")
        
    dataset = TelemetryDataset(data_dir)
    n_samples = len(dataset)
    val_size = int(n_samples * 0.2)
    train_size = n_samples - val_size
    
    # 固定隨機種子以確保結果可重現
    torch.manual_seed(42)
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"正在使用裝置: {device} 進行訓練")
    
    model = TelemetryAnalyzer().to(device)
    
    # 損失函數
    criterion_cls = nn.CrossEntropyLoss()
    criterion_reg = nn.MSELoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 訓練記錄
    history = {
        'train_loss': [], 'val_loss': [],
        'train_cls_loss': [], 'val_cls_loss': [],
        'train_reg_loss': [], 'val_reg_loss': [],
        'val_cls_acc': []
    }
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_cls_loss = 0.0
        epoch_reg_loss = 0.0
        
        for x_batch, y_cls, y_reg in train_loader:
            x_batch = x_batch.to(device)
            y_cls = y_cls.to(device)
            y_reg = y_reg.to(device)
            
            optimizer.zero_grad()
            
            logits, reg_preds, _ = model(x_batch)
            
            loss_cls = criterion_cls(logits, y_cls)
            loss_reg = criterion_reg(reg_preds, y_reg)
            
            # 多任務權重平衡：將 regression 權重放大以平衡尺度
            loss = loss_cls + 3.0 * loss_reg
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * x_batch.size(0)
            epoch_cls_loss += loss_cls.item() * x_batch.size(0)
            epoch_reg_loss += loss_reg.item() * x_batch.size(0)
            
        epoch_loss /= len(train_dataset)
        epoch_cls_loss /= len(train_dataset)
        epoch_reg_loss /= len(train_dataset)
        
        # 驗證
        model.eval()
        val_loss = 0.0
        val_cls_loss = 0.0
        val_reg_loss = 0.0
        correct = 0
        
        with torch.no_grad():
            for x_batch, y_cls, y_reg in val_loader:
                x_batch = x_batch.to(device)
                y_cls = y_cls.to(device)
                y_reg = y_reg.to(device)
                
                logits, reg_preds, _ = model(x_batch)
                
                loss_cls = criterion_cls(logits, y_cls)
                loss_reg = criterion_reg(reg_preds, y_reg)
                loss = loss_cls + 3.0 * loss_reg
                
                val_loss += loss.item() * x_batch.size(0)
                val_cls_loss += loss_cls.item() * x_batch.size(0)
                val_reg_loss += loss_reg.item() * x_batch.size(0)
                
                # 計算準確度
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_cls).sum().item()
                
        val_loss /= len(val_dataset)
        val_cls_loss /= len(val_dataset)
        val_reg_loss /= len(val_dataset)
        val_acc = correct / len(val_dataset)
        
        scheduler.step(val_loss)
        
        # 儲存歷史
        history['train_loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        history['train_cls_loss'].append(epoch_cls_loss)
        history['val_cls_loss'].append(val_cls_loss)
        history['train_reg_loss'].append(epoch_reg_loss)
        history['val_reg_loss'].append(val_reg_loss)
        history['val_cls_acc'].append(val_acc)
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:02d}/{epochs:02d} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Cls Acc: {val_acc*100.1:.1f}%")
            
        # 儲存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            
    print(f"訓練完成！最佳驗證 Loss: {best_val_loss:.4f}。模型已保存至 {model_save_path}")
    
    # 繪製訓練損失曲線
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Total Loss')
    plt.plot(history['val_loss'], label='Val Total Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Total Loss Curve')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['val_cls_acc'], label='Val Cls Accuracy', color='green')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    chart_path = os.path.join(os.path.dirname(model_save_path), 'training_progress.png')
    plt.savefig(chart_path)
    plt.close()
    print(f"訓練進度圖已儲存至 {chart_path}")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, 'data')
    model_save_path = os.path.join(current_dir, 'telemetry_model.pth')
    
    train_model(data_dir, model_save_path)
