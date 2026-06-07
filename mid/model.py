import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttentionPooling(nn.Module):
    """
    自注意力池化層 (Self-Attention Pooling)
    用以對序列的時間/空間步進行加權求和，並提取注意力權重以實現可解釋性 (Explainable AI)
    """
    def __init__(self, hidden_dim):
        super(SelfAttentionPooling, self).__init__()
        self.attn_linear = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x 維度: (batch_size, seq_len, hidden_dim)
        attn_weights = self.attn_linear(x) # (batch_size, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1) # 在 seq_len 維度做 Softmax
        
        # 加權求和: (batch_size, hidden_dim)
        pooled = torch.sum(x * attn_weights, dim=1)
        return pooled, attn_weights

class TelemetryAnalyzer(nn.Module):
    """
    基於 BiLSTM 與 Self-Attention 的多任務 F1 遙測分析模型
    輸入 A 與 B 對齊後的遙測數據序列，同時預測瓶頸類別與具體行為數值
    """
    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2):
        super(TelemetryAnalyzer, self).__init__()
        
        # 雙向 LSTM 骨幹網路
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.2 if num_layers > 1 else 0.0
        )
        
        # 雙向 LSTM 的輸出維度是 hidden_dim * 2
        self.feature_dim = hidden_dim * 2
        
        # 自注意力池化層
        self.attention_pooling = SelfAttentionPooling(self.feature_dim)
        
        # 分類分支 (Classification Head) - 預測行為瓶頸類別
        # 0: Normal, 1: Late Braking, 2: Early Braking, 3: Slow Throttle
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 4)
        )
        
        # 迴歸分支 (Regression Head) - 預測行為指標
        # 輸出: [brake_delta (s), exit_speed_delta (km/h), time_loss (s)]
        self.regressor = nn.Sequential(
            nn.Linear(self.feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)
        )
        
    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        lstm_out, _ = self.lstm(x) # lstm_out: (batch_size, seq_len, feature_dim)
        
        # 使用自注意力機制聚合序列
        pooled_feat, attn_weights = self.attention_pooling(lstm_out) # (batch_size, feature_dim)
        
        # 分類與迴歸預測
        logits = self.classifier(pooled_feat) # (batch_size, 4)
        reg_preds = self.regressor(pooled_feat) # (batch_size, 3)
        
        return logits, reg_preds, attn_weights

if __name__ == '__main__':
    # 測試模型前向傳播
    model = TelemetryAnalyzer()
    test_input = torch.randn(8, 100, 6) # batch_size=8, seq_len=100, features=6
    logits, reg, attn = model(test_input)
    print("模型前向傳播測試成功：")
    print(f"輸入維度: {test_input.shape}")
    print(f"分類輸出維度 (logits): {logits.shape}")
    print(f"迴歸輸出維度 (reg): {reg.shape}")
    print(f"注意力權重維度 (attn): {attn.shape}")
