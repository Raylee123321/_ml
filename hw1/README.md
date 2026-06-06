# TSP 2-Opt 本地搜尋求解器 (TSP Solver)

本專案實作了旅行推銷員問題 (Traveling Salesperson Problem, TSP) 的本地搜尋求解器，使用 2-opt 鄰居變換、高度評估函數，並比較多種本地搜尋演算法。

## 對話與需求紀錄

1. **建立專案資料夾**：新建獨立資料夾 `tsp` 以利模組化。
2. **實作核心要求**：
   - 實作符合定義的 `Solution` 類別。
   - **鄰居函數 `neighbor`**：隨機選擇兩條邊 $(a,b)$ 與 $(c,d)$，重連為 $(a,d)$ 與 $(b,c)$。在單一無向環的 TSP 中，這對應標準的 **2-opt swap**（即反轉該區間內的所有節點順序，否則會分裂成兩個獨立的圈）。
   - **高度函數 `height`**：定義為 $\text{總旅行距離} \times -1$，求解目標為極大化高度。
   - **初始解**：採用順序走訪 $1 \to 2 \to 3 \to \dots \to n \to 1$ 的方式。

---

## 專案結構

- [solution.py](file:///e:/newhwforCJ/tsp/solution.py)：定義 `Solution` 類別，包含路徑表示、`height` 與 2-opt `neighbor` 函數。
- [solver.py](file:///e:/newhwforCJ/tsp/solver.py)：實作三種本地搜尋演算法（Hill Climbing, Steepest Ascent Hill Climbing, Simulated Annealing）。
- [main.py](file:///e:/newhwforCJ/tsp/main.py)：問題初始化與測試入口，隨機產生 15 個城市進行求解對比。

---

## 演算法執行方式

於目錄下執行：
```bash
python main.py
```

### 演算法實作細節

1. **隨機爬山演算法 (Hill Climbing)**：
   隨機產生 2-opt 鄰居，若高度增加（距離變短）則移動。
2. **最陡爬山演算法 (Steepest Ascent Hill Climbing)**：
   遍歷所有可能的 2-opt 鄰居（共 $\frac{n(n-3)}{2}$ 個），並往最優（高度最高）的鄰居移動。
3. **模擬退火演算法 (Simulated Annealing)**：
   利用波茲曼機率分佈，在溫度高時有機會接受高度降低的鄰居以跳脫局部最佳解。
