"""
gpt.py - 基於 nn0.py 自動微分引擎的模組化 GPT 實作
參考自 Andrej Karpathy 的 microgpt 專案

本模組使用物件導向方式將 GPT 神經網路架構封裝於 GPT 類別中，
並完全複用 nn0.py 的基礎算子、優化器及 gd 訓練迴圈。
"""

import os
import random
import urllib.request
from nn0 import Value, Adam, linear, softmax, rmsnorm, gd

# 設定亂數種子，維持結果一致性
random.seed(42)

# --- 1. 資料集下載與載入 ---
dataset_file = 'input.txt'
if not os.path.exists(dataset_file):
    print("下載訓練數據集...")
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, dataset_file)

docs = [line.strip() for line in open(dataset_file, encoding='utf-8') if line.strip()]
random.shuffle(docs)
print(f"Loaded documents: {len(docs)}")

# --- 2. Tokenizer (字元級分詞器) ---
uchars = sorted(set(''.join(docs)))
BOS = len(uchars)  # Beginning of Sequence (和 End of Sequence 同用此 Token)
vocab_size = len(uchars) + 1
print(f"Vocab size: {vocab_size} (Characters: {len(uchars)} + 1 BOS)")

# --- 3. GPT 模型結構定義 ---
class GPT:
    """
    遵循 GPT-2 設計理念，使用純 Value 自動微分物件構成的 Transformer 解碼器 (Decoder-only) 模型。
    """
    def __init__(self, vocab_size, n_layer=1, n_embd=16, block_size=16, n_head=4):
        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_embd = n_embd
        self.block_size = block_size
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        # 初始化模型的所有參數矩陣
        std = 0.08
        matrix = lambda nout, nin: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
        
        self.state_dict = {
            'wte': matrix(vocab_size, n_embd),      # Token Embedding
            'wpe': matrix(block_size, n_embd),     # Position Embedding
            'lm_head': matrix(vocab_size, n_embd)  # Language Model Head
        }
        
        # 動態生成各個 Transformer 層參數
        for i in range(n_layer):
            self.state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
            self.state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
            self.state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
            self.state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
            self.state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
            self.state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)

    def __call__(self, token_id, pos_id, keys, values):
        """
        單步 Forward 運算：輸入目前 Token ID 與位置 ID，回傳預測下一個 Token 的 Logits。
        """
        # 1) Embedding 層與 RMSNorm 歸一化
        tok_emb = self.state_dict['wte'][token_id]
        pos_emb = self.state_dict['wpe'][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm(x)

        # 2) Transformer 堆疊層運算
        for li in range(self.n_layer):
            # --- Multi-head Self-Attention ---
            x_residual = x
            x = rmsnorm(x)
            
            q = linear(x, self.state_dict[f'layer{li}.attn_wq'])
            k = linear(x, self.state_dict[f'layer{li}.attn_wk'])
            v = linear(x, self.state_dict[f'layer{li}.attn_wv'])
            
            # 儲存與記錄 KV cache 用以因應自迴歸 (Autoregressive) 的注意力計算
            keys[li].append(k)
            values[li].append(v)
            
            x_attn = []
            for h in range(self.n_head):
                hs = h * self.head_dim
                q_h = q[hs : hs + self.head_dim]
                k_h = [ki[hs : hs + self.head_dim] for ki in keys[li]]
                v_h = [vi[hs : hs + self.head_dim] for vi in values[li]]
                
                # 計算 Scaled Dot-Product Attention 權重
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(self.head_dim)) / (self.head_dim ** 0.5)
                    for t in range(len(k_h))
                ]
                attn_weights = softmax(attn_logits)
                
                # Weighted Sum
                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(self.head_dim)
                ]
                x_attn.extend(head_out)
                
            x = linear(x_attn, self.state_dict[f'layer{li}.attn_wo'])
            x = [a + b for a, b in zip(x, x_residual)]  # 殘差連接

            # --- MLP 前饋網絡 ---
            x_residual = x
            x = rmsnorm(x)
            x = linear(x, self.state_dict[f'layer{li}.mlp_fc1'])
            x = [xi.relu() for xi in x]
            x = linear(x, self.state_dict[f'layer{li}.mlp_fc2'])
            x = [a + b for a, b in zip(x, x_residual)]  # 殘差連接

        # 3) LM Head 輸出預測 Logits
        logits = linear(x, self.state_dict['lm_head'])
        return logits

    def parameters(self):
        """
        導出所有的參數變數以便優化器存取。
        """
        params = []
        for mat in self.state_dict.values():
            for row in mat:
                params.extend(row)
        return params

# --- 4. 實例化模型與優化器 ---
model = GPT(vocab_size=vocab_size, n_layer=1, n_embd=16, block_size=16, n_head=4)
print(f"Model parameters: {len(model.parameters())} weights")

# 使用 nn0.py 封裝好的 Adam 優化器
optimizer = Adam(model.parameters(), lr=0.01, beta1=0.85, beta2=0.99)

# --- 5. 訓練迴圈 (Training Loop) ---
num_steps = 1000
print("\n開始訓練 GPT 模型...")
for step in range(num_steps):
    # 取出單個樣本 document 並做 token 轉換，前後包裹 BOS 標記
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    
    # 執行一步 Forward, Loss, Backward 與梯度更新 (使用 nn0.py 中的 gd 函數)
    loss_val = gd(model, optimizer, tokens, step, num_steps)
    
    if (step + 1) % 50 == 0 or step == 0:
        print(f"Step {step + 1:4d} / {num_steps:4d} | Loss: {loss_val:.4f}")

# --- 6. 生成推理 (Inference) ---
temperature = 0.5
print("\n--- 推理生成（新生成的英文名字） ---")
for sample_idx in range(20):
    # 初始化快取 (KV cache)
    keys = [[] for _ in range(model.n_layer)]
    values = [[] for _ in range(model.n_layer)]
    token_id = BOS
    sample = []
    
    for pos_id in range(model.block_size):
        # 預測下一個 Token
        logits = model(token_id, pos_id, keys, values)
        # 依據 Temperature 調節多樣性
        probs = softmax([l / temperature for l in logits])
        
        # 依權重抽樣
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
        
    print(f"Sample {sample_idx + 1:2d}: {''.join(sample)}")
