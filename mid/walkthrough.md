# 專案實作完成報告 (Walkthrough)

我們已順利完成了 **F1 賽車遙測行為預測與瓶頸分析 (AI Analysis)** 專案。
此專案結合了真實的 **FastF1** 遙測數據、空間插值對齊、多任務序列神經網路與自注意力（Self-Attention）機制，成功實現了自動尋找駕駛瓶頸並生成自然語言診斷報告的功能。

---

## 實作成果展示

### 1. 模型訓練進度 (`training_progress.png`)
模型在包含 96 個有標籤行為序列的資料集上訓練了 80 個 Epoch，準確率順利收斂至 **89.6%**，多任務聯合損失也平穩下降。

![模型訓練 Loss 與分類準確率](C:\Users\rayma\.gemini\antigravity-ide\brain\f97790f2-e5cf-4d2d-b9ca-8215c7f00ad6\training_progress.png)

---

### 2. AI 遙測診斷報告與對比圖

我們對測試車手 A 的單圈遙測進行了滾動掃描，模型成功偵測出多個行為瓶頸，並產出對照圖表。

#### 瓶頸 1: 晚煞車 (Late Braking)
* **檢測區域**：120m - 318m (關鍵點: 292m)
* **AI 診斷**：晚煞車 (Late Braking)
* **分析結果**：
  > 「在這個連續彎道中，A 車手因為晚煞車了 0.08 秒，導致後續出彎速度比 B 車手慢了 6.3 km/h，總共損失 0.28 秒。」

![晚煞車瓶頸與 Attention 權重對照圖](C:\Users\rayma\.gemini\antigravity-ide\brain\f97790f2-e5cf-4d2d-b9ca-8215c7f00ad6\bottleneck_analysis_1.png)
*(圖中底部的紫色區塊為 AI Attention Weight，藍色虛線為神經網路精準定位出的關鍵失誤點！)*

---

#### 瓶頸 2: 出彎給油猶豫 (Slow Throttle)
* **檢測區域**：400m - 598m (關鍵點: 552m)
* **AI 診斷**：出彎給油猶豫 (Slow Throttle)
* **分析結果**：
  > 「在這個出彎階段，A 車手加油門顯得遲疑，出彎速度比 B 車手慢了 5.6 km/h，總共損失 0.28 秒。」

![油門遲疑瓶頸與對照圖](C:\Users\rayma\.gemini\antigravity-ide\brain\f97790f2-e5cf-4d2d-b9ca-8215c7f00ad6\bottleneck_analysis_2.png)

---

## 2026 摩納哥排位賽真實遙測對比 (ANT vs VER)

我們使用新增的 [compare_monaco.py](file:///e:/co/_ml/mid/compare_monaco.py) 載入了 2026 年摩納哥大獎賽排位賽中，新秀 Kimi Antonelli (`ANT`) 與 Max Verstappen (`VER`) 的最快單圈遙測進行雙向對比：

### 1. ANT 相對 VER 表現落後的瓶頸區段 (以區段 #1 為例)
* **區域**: 120m - 318m (關鍵失誤點: 170m)
* **AI 診斷**: ANT 相比於 VER 早煞車 (Early Braking)
* **AI 分析**: 
  > 「在此彎道中，ANT 因為早煞車了 0.26 秒，入彎速度滑落較早，導致此區段損失 0.145 秒。」

![ANT 落後 VER 早煞車對照圖](C:\Users\rayma\.gemini\antigravity-ide\brain\f97790f2-e5cf-4d2d-b9ca-8215c7f00ad6\monaco_bottleneck_a_1.png)

---

### 2. ANT 相對 VER 表現領先的優勢區段 (以區段 #1 為例)
* **區域**: 90m - 288m (關鍵優勢點: 120m)
* **AI 診斷**: VER 相比於 ANT 早煞車 (Early Braking)
* **AI 分析**:
  > 「在此彎道中，VER 因為早煞車了 0.23 秒，入彎速度滑落較早，導致此區段損失 0.177 秒。」

![ANT 領先 VER 優勢對照圖](C:\Users\rayma\.gemini\antigravity-ide\brain\f97790f2-e5cf-4d2d-b9ca-8215c7f00ad6\monaco_advantage_a_1.png)

---

## 程式碼檔案結構與位置

我們已在 [e:\co\_ml\mid](file:///e:/co/_ml/mid) 目錄下建立了以下檔案：

1. **[data_prep.py](file:///e:/co/_ml/mid/data_prep.py)**：負責連接 FastF1 API 下載真實遙測、利用距離插值對齊、自動偵測彎道並注入駕駛缺陷來生成訓練集。
2. **[model.py](file:///e:/co/_ml/mid/model.py)**：定義 `TelemetryAnalyzer` 神經網路。整合 BiLSTM 與 Self-Attention 池化，搭配 Classification & Regression 雙解碼頭。
3. **[train.py](file:///e:/co/_ml/mid/train.py)**：執行 Dataset 建立與多任務權重聯合訓練，並繪製 `training_progress.png`。
4. **[analyze.py](file:///e:/co/_ml/mid/analyze.py)**：載入模型，進行滑動視窗滾動掃描，使用非極大值抑制合併區域，輸出 AI 行為診斷報告，並繪製對照圖表。
5. **[compare_monaco.py](file:///e:/co/_ml/mid/compare_monaco.py)**：新增的針對性分析腳本，加載 2026 摩納哥排位賽進行車手雙向對比。
6. **[README.md](file:///e:/co/_ml/mid/README.md)**：專案使用指南與技術規格說明文件。
