import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from model import TelemetryAnalyzer

def create_demo_driver_a(df_b):
    """
    動態生成一位測試車手 A 的一整圈遙測數據，
    在第 1 個彎道 (500m Apex) 注入 晚煞車 (Late Braking)，
    在第 3 個彎道 (3200m Apex) 注入 給油猶豫 (Slow Throttle)，
    其餘路段基本與 B 保持一致（加入微小噪聲）。
    """
    df_a = df_b.copy()
    
    # 加入基礎微小噪聲
    df_a['Speed'] += np.random.normal(0, 0.4, len(df_a))
    df_a['Throttle'] = np.clip(df_a['Throttle'] + np.random.normal(0, 1.5, len(df_a)), 0, 100)
    df_a['Brake'] = np.clip(df_a['Brake'] + np.random.normal(0, 1.0, len(df_a)), 0, 100)
    
    # --- 彎道 1 (400m - 700m): 注入 Late Braking ---
    # B 開始煞車約在 350m
    b_start_idx = df_b[df_b['Distance'] >= 350].index[0]
    late_shift = 8 # 延後 8 個 points (16米)
    
    df_a.loc[b_start_idx : b_start_idx + late_shift, 'Brake'] = 0.0
    df_a.loc[b_start_idx : b_start_idx + late_shift, 'Throttle'] = 100.0
    # 速度衝高
    df_a.loc[b_start_idx : b_start_idx + late_shift + 10, 'Speed'] += 20.0
    
    # 隨後在彎中大力重踩煞車
    df_a.loc[b_start_idx + late_shift : b_start_idx + 80, 'Brake'] = 100.0
    df_a.loc[b_start_idx + late_shift : b_start_idx + late_shift + 40, 'Throttle'] = 0.0
    
    # 出彎加油延遲，速度下降
    acc_start = b_start_idx + late_shift + 40
    throttle_slice = df_a.loc[acc_start : b_start_idx + 120, 'Throttle']
    df_a.loc[acc_start : b_start_idx + 120, 'Throttle'] = np.linspace(0, 70, len(throttle_slice))
    df_a.loc[acc_start : b_start_idx + 160, 'Speed'] -= 6.5 # 出彎慢 6.5 km/h
    
    # --- 彎道 3 (3000m - 3300m): 注入 Slow Throttle ---
    # B 開始煞車在 3050m，B Apex 在 3200m，B 開始踩油門約在 3210m
    b_apex_idx = df_b[df_b['Distance'] >= 3200].index[0]
    
    # 煞車一樣，但出彎加油很緩慢
    throttle_slice_c3 = df_a.loc[b_apex_idx : b_apex_idx + 80, 'Throttle']
    df_a.loc[b_apex_idx : b_apex_idx + 80, 'Throttle'] = np.linspace(0, 80, len(throttle_slice_c3)) * 0.4
    speed_slice_c3 = df_a.loc[b_apex_idx : b_apex_idx + 120, 'Speed']
    df_a.loc[b_apex_idx : b_apex_idx + 120, 'Speed'] -= np.linspace(0, 9.0, len(speed_slice_c3)) # 速度上升慢
    
    # 限制物理邊界
    df_a['Speed'] = np.clip(df_a['Speed'], 30.0, 340.0)
    df_a['Throttle'] = np.clip(df_a['Throttle'], 0.0, 100.0)
    df_a['Brake'] = np.clip(df_a['Brake'], 0.0, 100.0)
    
    # 重新計算 A 的累積時間 (T_A = T_start + sum(ds / v))
    time_a = np.zeros(len(df_a))
    curr_time = 0.0
    for idx in range(len(df_a)):
        if idx > 0:
            ds = df_a.loc[idx, 'Distance'] - df_a.loc[idx-1, 'Distance']
            v_ms = (df_a.loc[idx, 'Speed'] / 3.6)
            curr_time += ds / max(5.0, v_ms)
        time_a[idx] = curr_time
    df_a['Time'] = time_a
    
    return df_a

def analyze_telemetry(model_path, base_csv):
    """
    載入模型，掃描全圈遙測數據，尋找並分析 A 車手的瓶頸
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TelemetryAnalyzer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    df_b = pd.read_csv(base_csv)
    df_a = create_demo_driver_a(df_b)
    
    # 計算時間差與速度差
    # 注意：在真實 FastF1 數據對齊後，我們可以直接拿兩者的數據比較
    df_a['TimeDelta'] = df_a['Time'] - df_b['Time'] # A 減 B
    df_a['SpeedDelta'] = df_b['Speed'] - df_a['Speed'] # B 減 A (正值表示 A 較慢)
    
    print("\n" + "="*80)
    print("                      F1 AI TELEMETRY BOTTLENECK REPORT                      ")
    print("="*80)
    print(f"基準車手 B: VER | 目標車手 A: Demo Driver")
    print(f"賽道總長: {df_b['Distance'].max():.1f} m")
    print("-"*80)
    
    # 滾動掃描
    seq_len = 100
    step = 20 # 每次前進 40m
    n_points = len(df_b)
    
    detected_bottlenecks = []
    
    for start_idx in range(0, n_points - seq_len, step):
        end_idx = start_idx + seq_len
        sub_b = df_b.iloc[start_idx:end_idx].reset_index(drop=True)
        sub_a = df_a.iloc[start_idx:end_idx].reset_index(drop=True)
        
        # 準備模型輸入 [Speed_A, Throttle_A, Brake_A, Speed_B, Throttle_B, Brake_B]
        feat_a = sub_a[['Speed', 'Throttle', 'Brake']].values
        feat_b = sub_b[['Speed', 'Throttle', 'Brake']].values
        features = np.hstack([feat_a, feat_b]) # (100, 6)
        
        x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device) # (1, 100, 6)
        
        with torch.no_grad():
            logits, reg_preds, attn_weights = model(x_tensor)
            
            # 取得預測
            cls_pred = torch.argmax(logits, dim=1).item()
            reg_pred = reg_preds[0].cpu().numpy() # [brake_delta, exit_speed_delta, time_loss]
            attn = attn_weights[0, :, 0].cpu().numpy() # (100,)
            
        brake_delta, exit_speed_delta, time_loss = reg_pred
        
        # 篩選條件：如果預測不是 Class 0 (Normal) 且預測的 time_loss > 0.05 秒
        if cls_pred != 0 and time_loss > 0.05:
            # 尋找這個視窗內 Attention 最高的索引，定位關鍵失誤點的距離
            max_attn_idx = np.argmax(attn)
            key_distance = sub_a.loc[max_attn_idx, 'Distance']
            
            detected_bottlenecks.append({
                'start_dist': sub_a['Distance'].min(),
                'end_dist': sub_a['Distance'].max(),
                'key_dist': key_distance,
                'class': cls_pred,
                'brake_delta': brake_delta,
                'exit_speed_delta': exit_speed_delta,
                'time_loss': time_loss,
                'attn_weights': attn,
                'sub_a': sub_a,
                'sub_b': sub_b
            })
            
    # 進行非極大值抑制 (Non-Maximum Suppression) 合併相鄰且重疊的檢測區域
    # 我們根據預測的 time_loss 進行合併，保留 local peak
    merged = []
    if len(detected_bottlenecks) > 0:
        # 按照時間損失排序
        detected_bottlenecks.sort(key=lambda x: x['time_loss'], reverse=True)
        
        for item in detected_bottlenecks:
            overlap = False
            for m in merged:
                # 如果兩個區間中心距離小於 250m，視為同一個瓶頸區段
                if abs((item['start_dist'] + item['end_dist'])/2.0 - (m['start_dist'] + m['end_dist'])/2.0) < 250.0:
                    overlap = True
                    break
            if not overlap:
                merged.append(item)
                
    # 按照距離順序排序輸出結果
    merged.sort(key=lambda x: x['start_dist'])
    
    # 輸出分析報告並繪製圖表
    class_names = {1: "晚煞車 (Late Braking)", 2: "早煞車 (Early Braking)", 3: "出彎給油猶豫 (Slow Throttle)"}
    
    for i, item in enumerate(merged):
        cls_idx = item['class']
        cls_name = class_names.get(cls_idx, "未知缺陷")
        
        print(f"\n[瓶頸分析 #{i+1}] 區域: {item['start_dist']:.0f}m - {item['end_dist']:.0f}m (關鍵點: {item['key_dist']:.0f}m)")
        print(f"> AI 行為診斷: {cls_name}")
        
        if cls_idx == 1: # Late Braking
            print(f"> AI 分析版圖: 「在這個連續彎道中，A 車手因為晚煞車了 {item['brake_delta']:.2f} 秒，"
                  f"導致後續出彎速度比 B 車手慢了 {item['exit_speed_delta']:.1f} km/h，總共損失 {item['time_loss']:.2f} 秒。」")
        elif cls_idx == 2: # Early Braking
            print(f"> AI 分析版圖: 「在這個彎道中，A 車手提早煞車了 {abs(item['brake_delta']):.2f} 秒，"
                  f"導致入彎速度提前滑落，總共損失 {item['time_loss']:.2f} 秒。」")
        elif cls_idx == 3: # Slow Throttle
            print(f"> AI 分析版圖: 「在這個出彎階段，A 車手加油門顯得遲疑，出彎速度比 B 車手慢了 {item['exit_speed_delta']:.1f} km/h，"
                  f"總共損失 {item['time_loss']:.2f} 秒。」")
            
        # 繪製分析對照圖 (WOW 視覺效果)
        plot_bottleneck(item, i+1)
        
    print("\n" + "="*80)
    print("報告與分析對照圖表已全部生成！")
    print("="*80)

def plot_bottleneck(item, b_num):
    """
    為偵測到的瓶頸繪製詳細的遙測對照圖，包含 Attention Weights，
    以視覺化展示神經網路關注的決策特徵。
    """
    sub_a = item['sub_a']
    sub_b = item['sub_b']
    attn = item['attn_weights']
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    
    # 1. 速度對比
    axes[0].plot(sub_b['Distance'], sub_b['Speed'], 'g-', label='Driver B (VER)', linewidth=2)
    axes[0].plot(sub_a['Distance'], sub_a['Speed'], 'r--', label='Driver A (Demo)', linewidth=2)
    axes[0].set_ylabel('Speed (km/h)')
    axes[0].set_title(f'Bottleneck #{b_num} Telemetry Comparison (Key Event near {item["key_dist"]:.0f}m)')
    axes[0].grid(True)
    axes[0].legend()
    
    # 2. 油門對比
    axes[1].plot(sub_b['Distance'], sub_b['Throttle'], 'g-', label='Driver B', linewidth=1.5)
    axes[1].plot(sub_a['Distance'], sub_a['Throttle'], 'r--', label='Driver A', linewidth=1.5)
    axes[1].set_ylabel('Throttle (%)')
    axes[1].grid(True)
    
    # 3. 煞車對比
    axes[2].plot(sub_b['Distance'], sub_b['Brake'], 'g-', label='Driver B', linewidth=1.5)
    axes[2].plot(sub_a['Distance'], sub_a['Brake'], 'r--', label='Driver A', linewidth=1.5)
    axes[2].set_ylabel('Brake (%)')
    axes[2].grid(True)
    
    # 4. Attention 權重分佈 (可解釋性 AI 展現)
    axes[3].fill_between(sub_a['Distance'], attn, color='purple', alpha=0.3, label='AI Attention Weight')
    axes[3].plot(sub_a['Distance'], attn, color='purple', linewidth=1.5)
    axes[3].axvline(x=item['key_dist'], color='blue', linestyle=':', label='Key Fault Detection Point')
    axes[3].set_ylabel('Attention')
    axes[3].set_xlabel('Distance (m)')
    axes[3].grid(True)
    axes[3].legend()
    
    plt.tight_layout()
    plot_path = os.path.join(os.path.dirname(__file__), f'bottleneck_analysis_{b_num}.png')
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"> 遙測診斷圖表已儲存至: {plot_path}")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'telemetry_model.pth')
    base_csv = os.path.join(current_dir, 'base_telemetry.csv')
    
    if not os.path.exists(model_path):
        print(f"錯誤: 找不到已訓練的模型 {model_path}，請先運行 train.py！")
    elif not os.path.exists(base_csv):
        print(f"錯誤: 找不到基準遙測數據 {base_csv}，請先運行 data_prep.py！")
    else:
        analyze_telemetry(model_path, base_csv)
