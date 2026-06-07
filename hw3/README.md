# Tiny Autograd & Optimization Framework (`nn0.py`)

本專案是一個用純 Python 打造的輕量級自動微分與神經網路訓練框架。無須依賴 PyTorch 或 TensorFlow，透過簡單的數學原理與運算子過載，完整重現現代深度學習框架的核心基石。

---

## 🚀 核心特色

### 1. 自動微分引擎：`Value` 類別
自動微分是整個框架的靈魂，利用 **運算子過載 (Operator Overloading)** 來動態記錄數學運算的軌跡。
* **動態運算圖建構**：每個 `Value` 節點除了儲存數值 (`data`) 外，還會記錄它的來源子節點 (`_children`) 以及局部導數 (`_local_grads`)。
* **拓撲排序 (Topological Sort)**：在呼叫 `backward()` 時，演算法會先對所有關聯節點進行拓撲排序，確保計算梯度時，父節點的梯度必定在子節點之前計算完成。
* **鏈式法則 (Chain Rule) 實作**：反向傳播時，利用 $child.grad += local\_grad \times v.grad$ 沿著計算圖將全域梯度向後傳遞。

### 2. 優化演算法：`Adam` 優化器
`Adam` (Adaptive Moment Estimation) 是現代深度學習最主流的優化器。本框架完整實作了其動態調整學習率的機制：
* **一階與二階動量**：`m` (Momentum) 記錄梯度的指數移動平均，加速越過局部最小值；`v` (Velocity) 記錄梯度平方的指數移動平均，用於縮放每個參數的更新幅度。
* **偏差修正 (Bias Correction)**：計算 $\hat{m}$ 與 $\hat{v}$，修正訓練初期動量偏向 $0$ 的誤差。
* **學習率衰減**：支持學習率線性衰減，公式為 $lr_t = lr \times (1 - \frac{step}{num\_steps})$，使模型在訓練後期更穩定地收斂。

### 3. Transformer 關鍵組件與數值穩定算子
* **RMSNorm (Root Mean Square Layer Normalization)**：現代大型語言模型（如 Llama）偏好的歸一化機制，只縮放而不進行平移（Offset），計算效率更高。
* **數值穩定 Softmax**：計算前先減去 `max_val` 再進行 `exp()`，防止指數運算導致數值溢位 (`Floating point overflow`)。
* **融合 Cross Entropy**：結合 Log-Softmax 與 NLLLoss，利用 Log-Sum-Exp 技巧展開公式，避開了機率極低時取對數發生的數值崩潰：
  $$-\log\left(\frac{e^{x_t}}{\sum e^{x_i}}\right) = \log\left(\sum e^{x_i}\right) - x_t$$

---

## 📂 程式碼架構

* **`Value`**： autograd 節點，支援加、減、乘、除、負數、次方、`log()`、`exp()` 及 `relu()`。
* **`Adam`**： 優化器，負責參數更新與梯度清零。
* **`linear(x, w)`**： 實作 $W \cdot x$ 的矩陣乘法。
* **`softmax(logits)`**： 計算數值穩定的類別機率。
* **`rmsnorm(x)`**： 實作 RMS 歸一化。
* **`cross_entropy(logits, target_idx)`**： 數值穩定的交叉熵損失函數。
* **`gd(model, optimizer, tokens, step, num_steps)`**： 適用於 Token-based 的模型一步梯度下降訓練迴圈。

---

## ⚡ 快速開始與範例

[nn0.py](file:///e:/newhwforCJ/nn0.py) 內部已集成 self-contained 的示範程式。你可以直接在終端機執行：

```bash
python nn0.py
```

### 1. 驗證自動微分
以下程式碼展示了如何建立計算圖並求導：
```python
from nn0 import Value

a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
d = a * b + c
d.backward()

print(f"d 的數值: {d.data}") # 輸出: 4.0
print(f"a 的梯度: {a.grad}") # 輸出: -3.0 (即 b 的值)
print(f"b 的梯度: {b.grad}") # 輸出: 2.0 (即 a 的值)
```

### 2. 訓練 XOR 神經網路
[nn0.py](file:///e:/newhwforCJ/nn0.py) 包含了一個簡單的雙層 MLP 用於解決非線性的 XOR 分類問題，程式執行後會輸出如下的訓練收斂過程：

```text
=== 2. 使用 Adam 訓練 MLP 解決 XOR 分類問題 ===
開始訓練 XOR 神經網路...
Epoch   1/300 | Loss: 0.867302
Epoch  50/300 | Loss: 0.015569
Epoch 100/300 | Loss: 0.004452
Epoch 150/300 | Loss: 0.002247
Epoch 200/300 | Loss: 0.001388
Epoch 250/300 | Loss: 0.000948
Epoch 300/300 | Loss: 0.000690

訓練完成！XOR 預測結果測試：
輸入: [0.0, 0.0] | 預測機率 (0, 1): (0.9987, 0.0013) | 預測類別: 0 | 真實類別: 0
輸入: [0.0, 1.0] | 預測機率 (0, 1): (0.0002, 0.9998) | 預測類別: 1 | 真實類別: 1
...
```
