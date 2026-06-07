import os
import numpy as np
import pandas as pd

# 嘗試載入 fastf1，如果沒有則在執行時捕獲
try:
    import fastf1
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False

# 設定快取目錄
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'fastf1_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
if FASTF1_AVAILABLE:
    try:
        fastf1.Cache.enable_cache(CACHE_DIR)
    except Exception as e:
        print(f"無法啟用 FastF1 快取: {e}，將以無快取模式運行")

def get_real_telemetry(year=2023, gp='Spa', session_type='Q', driver='VER'):
    """
    使用 FastF1 獲取真實的車手單圈遙測數據
    """
    if not FASTF1_AVAILABLE:
        raise RuntimeError("FastF1 套件不可用，無法獲取真實遙測。")
    
    print(f"正在從 FastF1 下載 {year} {gp} GP {session_type} - {driver} 的數據...")
    session = fastf1.get_session(year, gp, session_type)
    session.load()
    
    lap = session.laps.pick_driver(driver).pick_fastest()
    telemetry = lap.get_telemetry()
    
    # 整理欄位
    df = pd.DataFrame({
        'Distance': telemetry['Distance'],
        'Speed': telemetry['Speed'],          # km/h
        'Throttle': telemetry['Throttle'],    # 0-100
        'Brake': telemetry['Brake'].astype(float) * 100.0, # 轉換為 0 或 100
        'Time': telemetry['SessionTime'].dt.total_seconds() # 秒
    })
    return df

def generate_simulated_telemetry():
    """
    高品質物理模擬遙測（Fallback 方案）
    模擬一個長度 7000m 的賽道，包含 5 個彎道，模擬車手 B 的完美單圈。
    """
    print("啟動高品質物理模擬引擎，生成基準車手遙測數據...")
    dist_grid = np.arange(0, 7000, 2.0) # 每 2 公尺一個點
    n_points = len(dist_grid)
    
    speed = np.zeros(n_points)
    throttle = np.zeros(n_points)
    brake = np.zeros(n_points)
    time_sec = np.zeros(n_points)
    
    # 彎道設定: (彎心距離 Apex, 彎心安全速度 km/h, 煞車起始距離, 出彎加速結束距離)
    corners = [
        {'apex': 500,  'v_apex': 90,  'brake_start': 350,  'acc_end': 750},
        {'apex': 1800, 'v_apex': 120, 'brake_start': 1600, 'acc_end': 2100},
        {'apex': 3200, 'v_apex': 80,  'brake_start': 3050, 'acc_end': 3450},
        {'apex': 4800, 'v_apex': 140, 'brake_start': 4650, 'acc_end': 5150},
        {'apex': 6000, 'v_apex': 100, 'brake_start': 5850, 'acc_end': 6350}
    ]
    
    v = 300.0 / 3.6 # 初始速度 300 km/h (m/s)
    current_time = 0.0
    
    for i, d in enumerate(dist_grid):
        # 尋找當前所在的區域
        in_corner = False
        for c in corners:
            if c['brake_start'] <= d < c['apex']:
                # 煞車入彎區
                in_corner = True
                brake[i] = 100.0
                throttle[i] = 0.0
                # 減速到 apex 速度
                dist_to_apex = c['apex'] - d
                total_brake_dist = c['apex'] - c['brake_start']
                v_target = c['v_apex'] / 3.6
                v_start = 300.0 / 3.6
                # 使用二次插值模擬煞車減速曲線
                ratio = dist_to_apex / total_brake_dist
                v = v_target + (v_start - v_target) * (ratio ** 1.5)
                break
            elif c['apex'] <= d < c['acc_end']:
                # 出彎加速區
                in_corner = True
                brake[i] = 0.0
                dist_from_apex = d - c['apex']
                total_acc_dist = c['acc_end'] - c['apex']
                ratio = dist_from_apex / total_acc_dist
                throttle[i] = ratio * 100.0 # 漸進踩油門
                v_apex = c['v_apex'] / 3.6
                v_exit = 280.0 / 3.6
                v = v_apex + (v_exit - v_apex) * (ratio ** 0.8)
                break
        
        if not in_corner:
            # 直線大直道加速
            brake[i] = 0.0
            throttle[i] = 100.0
            v_max = 330.0 / 3.6
            # 漸進接近極速
            v = v + (v_max - v) * 0.015
            
        # 限制速度區間
        v = max(30.0 / 3.6, min(v, 335.0 / 3.6))
        speed[i] = v * 3.6 # 轉回 km/h
        
        # 累加時間 dt = ds / v
        if i > 0:
            ds = 2.0
            dt = ds / v
            current_time += dt
        time_sec[i] = current_time
        
    df = pd.DataFrame({
        'Distance': dist_grid,
        'Speed': speed,
        'Throttle': throttle,
        'Brake': brake,
        'Time': time_sec
    })
    return df

def align_telemetry(df, step=2.0):
    """
    將遙測數據以固定距離間距（如每 2m）重新採樣，確保完全對齊。
    """
    max_dist = df['Distance'].max()
    grid = np.arange(0, max_dist, step)
    
    speed = np.interp(grid, df['Distance'], df['Speed'])
    throttle = np.interp(grid, df['Distance'], df['Throttle'])
    brake = np.interp(grid, df['Distance'], df['Brake'])
    time_sec = np.interp(grid, df['Distance'], df['Time'])
    
    # 限制油門與煞車在合理區間
    throttle = np.clip(throttle, 0.0, 100.0)
    brake = np.clip(brake, 0.0, 100.0)
    
    return pd.DataFrame({
        'Distance': grid,
        'Speed': speed,
        'Throttle': throttle,
        'Brake': brake,
        'Time': time_sec
    })

def inject_driver_a_defects(df_b, corners=None):
    """
    根據基準車手 B (df_b)，在特定彎道區域為車手 A 注入不同類別的駕駛缺陷，
    並計算精確的 behavior deltas (Ground Truth)。
    """
    df_a = df_b.copy()
    
    # 如果沒有指定彎道，則自動檢測或使用預設的 Spa/Sim 彎道位置
    if corners is None:
        # 簡單檢測煞車開始點與 Apex
        # 我們將連續 Brake > 50 的區域視為彎道
        corners = []
        in_brake = False
        start_idx = 0
        for idx in range(len(df_b)):
            if df_b.loc[idx, 'Brake'] > 50 and not in_brake:
                in_brake = True
                start_idx = idx
            elif df_b.loc[idx, 'Brake'] < 10 and in_brake:
                in_brake = False
                end_idx = idx
                # 尋找這段區間 Speed 最小點作為 Apex
                sub_df = df_b.iloc[start_idx:end_idx + 100] # 包含出彎
                if len(sub_df) > 0:
                    apex_idx = sub_df['Speed'].idxmin()
                    corners.append({
                        'start_dist': df_b.loc[start_idx, 'Distance'],
                        'apex_dist': df_b.loc[apex_idx, 'Distance'],
                        'end_dist': df_b.loc[min(len(df_b)-1, apex_idx + 80), 'Distance']
                    })
    
    # 儲存缺陷樣本與標籤
    # 每個彎道樣本長度 seq_len = 100 (對應 200m 區域)
    samples = []
    
    # 對每一個彎道生成多種駕駛缺陷
    for c_idx, corner in enumerate(corners):
        s_dist = corner['start_dist'] - 60 # 提早 60m 開始
        e_dist = corner['end_dist']
        
        # 尋找對應的 df_b 索引
        b_sub = df_b[(df_b['Distance'] >= s_dist) & (df_b['Distance'] <= e_dist)].reset_index(drop=True)
        if len(b_sub) < 100:
            continue
        # 限制長度為 100
        b_sub = b_sub.iloc[:100].reset_index(drop=True)
        
        # 1. 正常行駛樣本 (Class 0: Normal)
        normal_a = b_sub.copy()
        # 加入微小噪聲
        normal_a['Speed'] += np.random.normal(0, 0.5, len(normal_a))
        normal_a['Throttle'] = np.clip(normal_a['Throttle'] + np.random.normal(0, 2, len(normal_a)), 0, 100)
        normal_a['Brake'] = np.clip(normal_a['Brake'] + np.random.normal(0, 2, len(normal_a)), 0, 100)
        # 計算 delta
        samples.append({
            'seq_b': b_sub[['Speed', 'Throttle', 'Brake']].values,
            'seq_a': normal_a[['Speed', 'Throttle', 'Brake']].values,
            'label': 0, # Normal
            'brake_delta': 0.0,
            'exit_speed_delta': 0.0,
            'time_loss': 0.0
        })
        
        # 2. 晚煞車缺陷 (Class 1: Late Braking)
        # 參數隨機化
        for _ in range(5):
            late_dist = np.random.uniform(8.0, 20.0) # 晚煞車 8m 到 20m
            # 對應的時間延遲
            speed_at_brake = b_sub.loc[30, 'Speed'] / 3.6 # m/s (假設約第 30 點開始煞車)
            brake_delta_s = late_dist / max(10.0, speed_at_brake)
            
            exit_speed_loss = np.random.uniform(2.0, 8.0) # 出彎慢 2-8 km/h
            time_loss_s = np.random.uniform(0.15, 0.35)    # 總共損失 0.15-0.35s
            
            a_sub = b_sub.copy()
            # 模擬晚煞車：將煞車信號向後延遲
            # 假設 B 的煞車從 idx 30 開始
            brake_idx_b = (a_sub['Brake'] > 50).idxmax() if (a_sub['Brake'] > 50).any() else 30
            shift_idx = int(late_dist / 2.0) # 每個 step 2m
            
            # A 在延遲期間內不煞車，繼續保持高速
            a_sub.loc[:brake_idx_b + shift_idx, 'Brake'] = 0.0
            a_sub.loc[:brake_idx_b + shift_idx, 'Throttle'] = 100.0 # 繼續給油
            
            # 因為煞車太晚，速度在初期偏高
            a_sub.loc[brake_idx_b:brake_idx_b + shift_idx + 10, 'Speed'] += np.random.uniform(15, 30)
            
            # 但在彎中被迫大力減速，且出彎加油點大幅延遲
            a_sub.loc[brake_idx_b + shift_idx:, 'Brake'] = 100.0
            a_sub.loc[brake_idx_b + shift_idx:brake_idx_b + shift_idx + 40, 'Throttle'] = 0.0
            
            # 出彎油門推遲，油門斜率降低
            throttle_start = brake_idx_b + shift_idx + 30
            a_sub.loc[throttle_start:, 'Throttle'] = np.clip(
                np.linspace(0, 100, len(a_sub) - throttle_start) * 0.6, 0, 100
            )
            
            # 出彎速度降低
            a_sub.loc[throttle_start:, 'Speed'] -= exit_speed_loss
            a_sub['Speed'] = np.clip(a_sub['Speed'], 30.0, 340.0)
            
            samples.append({
                'seq_b': b_sub[['Speed', 'Throttle', 'Brake']].values,
                'seq_a': a_sub[['Speed', 'Throttle', 'Brake']].values,
                'label': 1, # Late Braking
                'brake_delta': brake_delta_s,
                'exit_speed_delta': exit_speed_loss,
                'time_loss': time_loss_s
            })
            
        # 3. 早煞車缺陷 (Class 2: Early Braking)
        for _ in range(5):
            early_dist = np.random.uniform(10.0, 25.0) # 早煞車 10m 到 25m
            speed_at_brake = b_sub.loc[30, 'Speed'] / 3.6
            brake_delta_s = -early_dist / max(10.0, speed_at_brake) # 早煞車為負值
            
            exit_speed_loss = np.random.uniform(-1.0, 1.0) # 出彎速度與 B 差不多
            time_loss_s = np.random.uniform(0.08, 0.22)    # 損失時間
            
            a_sub = b_sub.copy()
            brake_idx_b = (a_sub['Brake'] > 50).idxmax() if (a_sub['Brake'] > 50).any() else 30
            shift_idx = int(early_dist / 2.0)
            
            # A 提早煞車
            early_start = max(0, brake_idx_b - shift_idx)
            a_sub.loc[early_start:brake_idx_b, 'Brake'] = 100.0
            a_sub.loc[early_start:brake_idx_b, 'Throttle'] = 0.0
            
            # 速度提前下降
            a_sub.loc[early_start:brake_idx_b + 30, 'Speed'] -= np.random.uniform(15, 30)
            a_sub['Speed'] = np.clip(a_sub['Speed'], 30.0, 340.0)
            
            samples.append({
                'seq_b': b_sub[['Speed', 'Throttle', 'Brake']].values,
                'seq_a': a_sub[['Speed', 'Throttle', 'Brake']].values,
                'label': 2, # Early Braking
                'brake_delta': brake_delta_s,
                'exit_speed_delta': exit_speed_loss,
                'time_loss': time_loss_s
            })
            
        # 4. 出彎給油慢缺陷 (Class 3: Slow Throttle)
        for _ in range(5):
            brake_delta_s = 0.0 # 煞車點一樣
            exit_speed_loss = np.random.uniform(3.0, 10.0) # 出彎慢 3-10 km/h
            time_loss_s = np.random.uniform(0.12, 0.40)    # 直線損失時間
            
            a_sub = b_sub.copy()
            brake_idx_b = (a_sub['Brake'] > 50).idxmax() if (a_sub['Brake'] > 50).any() else 30
            # 煞車一樣，但踩油門非常緩慢
            # 尋找 B 開始踩油門的位置 (例如在 Apex 後)
            throttle_idx_b = (b_sub.loc[brake_idx_b:, 'Throttle'] > 10).idxmax() + brake_idx_b
            
            # A 的油門上升速度只有 B 的一半或更低
            a_sub.loc[throttle_idx_b:, 'Throttle'] = np.clip(
                np.linspace(0, 100, len(a_sub) - throttle_idx_b) * 0.4, 0, 100
            )
            a_sub.loc[throttle_idx_b:, 'Speed'] -= np.linspace(0, exit_speed_loss, len(a_sub) - throttle_idx_b)
            a_sub['Speed'] = np.clip(a_sub['Speed'], 30.0, 340.0)
            
            samples.append({
                'seq_b': b_sub[['Speed', 'Throttle', 'Brake']].values,
                'seq_a': a_sub[['Speed', 'Throttle', 'Brake']].values,
                'label': 3, # Slow Throttle
                'brake_delta': brake_delta_s,
                'exit_speed_delta': exit_speed_loss,
                'time_loss': time_loss_s
            })
            
    return samples

def prepare_dataset_csv():
    """
    獲取數據，生成缺陷資料集，並將序列保存為 Numpy 或 CSV 格式。
    """
    # 1. 獲取基準單圈數據
    df_b = None
    if FASTF1_AVAILABLE:
        try:
            df_b = get_real_telemetry(2023, 'Spa', 'Q', 'VER')
            df_b = align_telemetry(df_b)
        except Exception as e:
            print(f"從 FastF1 獲取真實數據失敗: {e}。")
            df_b = None
            
    if df_b is None:
        df_b = generate_simulated_telemetry()
        df_b = align_telemetry(df_b)
        
    # 保存對齊後的基礎數據
    base_file = os.path.join(os.path.dirname(__file__), 'base_telemetry.csv')
    df_b.to_csv(base_file, index=False)
    print(f"基準遙測已儲存至 {base_file}")
    
    # 2. 生成缺陷樣本
    samples = inject_driver_a_defects(df_b)
    print(f"成功生成 {len(samples)} 個訓練/測試序列樣本。")
    
    # 3. 整理成 Numpy arrays 並儲存
    # 輸入特徵 X 包含: [Speed_A, Throttle_A, Brake_A, Speed_B, Throttle_B, Brake_B]
    # 維度: (N, seq_len=100, 6)
    X = []
    Y_cls = []
    Y_reg = []
    
    for s in samples:
        # 將 A 與 B 的遙測特徵拼接
        seq_features = np.hstack([s['seq_a'], s['seq_b']]) # (100, 6)
        X.append(seq_features)
        Y_cls.append(s['label'])
        Y_reg.append([s['brake_delta'], s['exit_speed_delta'], s['time_loss']])
        
    X = np.array(X, dtype=np.float32)
    Y_cls = np.array(Y_cls, dtype=np.int64)
    Y_reg = np.array(Y_reg, dtype=np.float32)
    
    # 儲存為 .npy 檔案
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    np.save(os.path.join(data_dir, 'X.npy'), X)
    np.save(os.path.join(data_dir, 'Y_cls.npy'), Y_cls)
    np.save(os.path.join(data_dir, 'Y_reg.npy'), Y_reg)
    print(f"特徵與標籤已儲存至 {data_dir}/ 目錄中。")

if __name__ == '__main__':
    prepare_dataset_csv()
