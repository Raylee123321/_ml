"""
nn.py — 自動微分引擎 (Value) 與 Adam 優化器

提供：
  class Value  — 純 Python autograd 節點
  class Adam   — Adam optimizer
  linear()     — 矩陣乘法
  softmax()    — 數值穩定 softmax
  rmsnorm()    — RMS Normalization
"""

import math

class Value:
    """純 Python 的自動微分節點，支援反向傳播。"""
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data**(other - 1),))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self):
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self):         return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other):  return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other):  return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        """反向傳播：計算所有參數的梯度。"""
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

    def __repr__(self):
        return f"Value({self.data:.4f})"


class Adam:
    """Adam optimizer，支援 learning rate 線性衰減。"""

    def __init__(self, params, lr=0.01, beta1=0.85, beta2=0.99, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = [0.0] * len(params)
        self.v = [0.0] * len(params)
        self.step_count = 0

    def step(self, lr_override=None):
        """執行一步參數更新，並清除梯度。"""
        self.step_count += 1
        lr = lr_override if lr_override is not None else self.lr
        for i, p in enumerate(self.params):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * p.grad
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * p.grad ** 2
            m_hat = self.m[i] / (1 - self.beta1 ** self.step_count)
            v_hat = self.v[i] / (1 - self.beta2 ** self.step_count)
            p.data -= lr * m_hat / (v_hat ** 0.5 + self.eps)
            p.grad = 0


def linear(x, w):
    """矩陣乘法：y = W @ x"""
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    """數值穩定的 softmax。"""
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    """RMS Normalization（取代 LayerNorm）。"""
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def gd(model, optimizer, tokens, step, num_steps):
    """
    一步梯度下降：forward → loss → backward → Adam update。
    回傳 loss 值。
    """
    n = min(model.block_size, len(tokens) - 1)
    keys   = [[] for _ in range(model.n_layer)]
    values = [[] for _ in range(model.n_layer)]

    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = model(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    loss.backward()

    lr_t = optimizer.lr * (1 - step / num_steps)
    optimizer.step(lr_override=lr_t)

    return loss.data

def cross_entropy_simple(logits, target_idx):
    """
    直觀版本的 Cross-Entropy，依賴現有的 softmax 函式。
    """
    probs = softmax(logits)
    # 提取目標類別的機率，並取負對數
    return -probs[target_idx].log()

r"""
真實的深度學習框架（如 PyTorch 的 F.cross_entropy）中，通常不會「先算 Softmax 機率，再算 Log」，
因為機率值如果逼近 0，再取 .log() 時非常容易引發數值錯誤（例如出現 -inf 或 NaN）。

框架會利用數學對數律展開公式：

$$-\log\left(\frac{e^{x_t}}{\sum e^{x_i}}\right) = \log\left(\sum e^{x_i}\right) - x_t$$

結合 softmax 原本避免溢位的 max_val 減法技巧（即 Log-Sum-Exp trick），
我們可以寫出更穩定、計算圖也更短的版本：
"""
def cross_entropy(logits, target_idx):
    """
    融合 Log-Softmax 與 NLLLoss 的數值穩定版本。
    計算公式： log(sum(exp(x_i - max))) - (x_target - max)
    """
    # 1. 找出最大值以確保數值穩定 (純量)
    max_val = max(val.data for val in logits)
    
    # 2. 計算所有元素的 exp(x_i - max)
    exps = [(val - max_val).exp() for val in logits]
    
    # 3. 計算分母的總和，並建立 log 節點
    log_sum_exp = sum(exps).log()
    
    # 4. 計算最終 Loss，直接避開了除法運算與極小機率的 log 運算
    target_logit_shifted = logits[target_idx] - max_val
    return log_sum_exp - target_logit_shifted


if __name__ == "__main__":
    print("=== 1. 自動微分引擎 (Value) 極簡示範 ===")
    a = Value(2.0)
    b = Value(-3.0)
    c = Value(10.0)
    d = a * b + c
    d.backward()
    print(f"公式: d = a * b + c")
    print(f"數值: a={a.data}, b={b.data}, c={c.data} => d={d.data}")
    print(f"梯度: a.grad={a.grad} (應為 -3.0), b.grad={b.grad} (應為 2.0), c.grad={c.grad} (應為 1.0)\n")

    print("=== 2. 使用 Adam 訓練 MLP 解決 XOR 分類問題 ===")
    
    class XOR_MLP:
        def __init__(self):
            import random
            random.seed(42)
            # 隱藏層 (2 -> 4)
            self.W1 = [[Value(random.uniform(-1.0, 1.0)) for _ in range(2)] for _ in range(4)]
            self.b1 = [Value(0.0) for _ in range(4)]
            # 輸出層 (4 -> 2)，對應 softmax 二分類
            self.W2 = [[Value(random.uniform(-1.0, 1.0)) for _ in range(4)] for _ in range(2)]
            self.b2 = [Value(0.0) for _ in range(2)]

        def forward(self, x):
            # 隱藏層
            h_linear = linear(x, self.W1)
            h = [(h_linear[i] + self.b1[i]).relu() for i in range(4)]
            # 輸出層 (logits)
            logits_linear = linear(h, self.W2)
            logits = [logits_linear[i] + self.b2[i] for i in range(2)]
            return logits

        def parameters(self):
            params = []
            for row in self.W1: params.extend(row)
            params.extend(self.b1)
            for row in self.W2: params.extend(row)
            params.extend(self.b2)
            return params

    # XOR 訓練資料 (X, Y)
    X = [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0]
    ]
    Y = [0, 1, 1, 0] # 目標類別

    model = XOR_MLP()
    # parameters 包含 W1, b1, W2, b2 所有的 Value 物件
    optimizer = Adam(model.parameters(), lr=0.1, beta1=0.9, beta2=0.999)

    print("開始訓練 XOR 神經網路...")
    epochs = 300
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        # 逐樣本訓練 (SGD)
        for x, y in zip(X, Y):
            # Forward
            logits = model.forward(x)
            # Loss (使用數值穩定的 cross_entropy)
            loss = cross_entropy(logits, y)
            total_loss += loss.data
            
            # Backward
            loss.backward()
            
            # Update
            optimizer.step()

        # 平均 loss
        avg_loss = total_loss / len(X)
        if epoch % 50 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | Loss: {avg_loss:.6f}")

    print("\n訓練完成！XOR 預測結果測試：")
    for x, y_true in zip(X, Y):
        logits = model.forward(x)
        probs = softmax(logits)
        pred = 0 if probs[0].data > probs[1].data else 1
        probs_data = [p.data for p in probs]
        print(f"輸入: {x} | 預測機率 (0, 1): ({probs_data[0]:.4f}, {probs_data[1]:.4f}) | 預測類別: {pred} | 真實類別: {y_true}")

