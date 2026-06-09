# F1 賽車遙測行為預測與瓶頸分析 (AI Analysis)

本專案實作了一個基於深度學習的 F1 遙測數據分析系統。不同於一般的單純圖表繪製，本系統透過訓練一個多任務神經網路（BiLSTM + Self-Attention），將兩位車手的遙測數據輸入模型，讓 AI 自動識別駕駛行為的瓶頸，並生成精準的自然語言分析報告。

例如：
> **「在這個連續彎道中，A 車手因為晚煞車了 0.05 秒，導致後續出彎速度比 B 車手慢了 3.0 km/h，總共損失 0.20 秒。」**

---

## 系統架構與設計原理

### 1. 遙測數據對齊與缺陷注入 (`data_prep.py`)
* **資料獲取**：使用 `FastF1` 套件，下載真實 F1 賽事（例如 2023 年比利時 SPA 站排位賽）中頂尖車手（如 Max Verstappen）的最快圈遙測數據。*（若網路受限，系統會自動切換至高品質物理模擬引擎作為備用方案）*。
* **距離軸對齊 (Distance-based Alignment)**：將車手的遙測數據以「距離」為基準進行重採樣（每 2 公尺一個採樣點），使兩位車手的數據在空間上完全對齊。
* **缺陷注入 (Data Augmentation)**：為了解決真實遙測缺乏標註瓶頸標籤的問題，我們以基準車手為底，隨機在各個彎道注入三種駕駛缺陷：
  1. **晚煞車 (Late Braking)**：延遲煞車點，導致入彎過快錯過 Apex，推遲出彎給油並降低出彎速度與損失時間。
  2. **早煞車 (Early Braking)**：過早煞車，導致入彎速度偏低而損失時間。
  3. **給油猶豫 (Slow Throttle)**：在 Apex 後踩油門的斜率偏低，導致出彎加速慢。

### 2. 深度學習模型架構 (`model.py`)
模型採用 **多任務學習 (Multi-task Learning)** 的雙向 LSTM 結構：
* **輸入層**：對齊後的遙測序列網格，特徵包含 `[Speed_A, Throttle_A, Brake_A, Speed_B, Throttle_B, Brake_B]`，輸入維度為 `(batch_size, seq_len=100, features=6)`。
* **BiLSTM 骨幹**：提取雙向時空序列特徵。
* **自注意力池化 (Self-Attention Pooling)**：對序列的所有步進行加權求和，除了提取全局特徵，還能**輸出注意力權重 (Attention Weights)**，自動定位最關鍵的駕駛缺陷發生位置（可解釋性 AI）。
* **分類頭 (Classification Head)**：預測缺陷種類（正常/晚煞車/早煞車/油門遲疑）。
* **迴歸頭 (Regression Head)**：預測三個關鍵行為指標：`[煞車時間差 (s), 出彎速度差 (km/h), 該彎道損失時間 (s)]`。

### 3. 模型訓練 (`train.py`)
* 使用 Adam 最佳化器，聯合 CrossEntropy 損失與 MSE 損失進行訓練。
* 保存最佳權重至 `telemetry_model.pth`，並自動繪製 `training_progress.png` 記錄 Loss 與準確率的收斂。

### 4. 滾動掃描與診斷報告 (`analyze.py`)
* 使用滑動視窗對 A 車手的整圈遙測數據進行滾動掃描（Sliding Window）。
* 使用非極大值抑制（NMS）將高度重疊的檢測區域進行合併。
* 當時間損失大於閾值（如 0.05 秒）時，觸發 AI 分析，並套用自然語言模板生成中文報告。
* 輸出精緻的 `bottleneck_analysis_#.png` 圖表，視覺化兩車手遙測波形對比與 AI 注意力權重分佈。

---

## 專案檔案結構

* [data_prep.py](file:///e:/co/_ml/mid/data_prep.py)：獲取遙測、插值對齊、注入缺陷並生成 `.npy` 資料集。
* [model.py](file:///e:/co/_ml/mid/model.py)：定義基於 PyTorch 的 `TelemetryAnalyzer` 多任務網路。
* [train.py](file:///e:/co/_ml/mid/train.py)：數據載入與多任務聯合訓練。
* [analyze.py](file:///e:/co/_ml/mid/analyze.py)：滾動掃描遙測並生成 AI 行為分析報告與視覺化對比圖。
* [README.md](file:///e:/co/_ml/mid/README.md)：本說明文件。


