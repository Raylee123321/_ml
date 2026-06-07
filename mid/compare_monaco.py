import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

try:
    import fastf1
    FASTF1_AVAILABLE = True
except ImportError:
    FASTF1_AVAILABLE = False

from model import TelemetryAnalyzer

# 設定快取目錄
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'fastf1_cache')
os.makedirs(CACHE_DIR, exist_ok=True)
if FASTF1_AVAILABLE:
    try:
        fastf1.Cache.enable_cache(CACHE_DIR)
    except Exception as e:
        print(f"無法啟用 FastF1 快取: {e}")

def load_session_and_laps(year=2026, gp='Monaco', driver_a='ANT', driver_b='VER'):
    """
    載入 F1 排位賽 Session 並獲取兩位車手最快圈的遙測數據。
    支援年份與車手的自動 Fallback。
    """
    if not FASTF1_AVAILABLE:
        raise RuntimeError("FastF1 套件未安裝！")
        
    print(f"正在加載 {year} 年 {gp} 大獎賽排位賽數據...")
    
    session = None
    try:
        session = fastf1.get_session(year, gp, 'Q')
        session.load()
    except Exception as e:
        print(f"加載 {year} 年數據失敗: {e}")
        # Fallback 到 2024 年
        fallback_year = 2024
        print(f"嘗試 Fallback 到 {fallback_year} 年 {gp} 大獎賽...")
        session = fastf1.get_session(fallback_year, gp, 'Q')
        session.load()
        year = fallback_year
        
    # 檢查可用車手
    drivers = session.laps['Driver'].unique()
    print(f"該場排位賽可用車手名單: {list(drivers)}")
    
    # 確保兩名車手在名單中，否則進行 Fallback
    da = driver_a
    db = driver_b
    
    if da not in drivers:
        # 如果 ANT (Antonelli) 不在名單（例如年份退回 2024），Fallback 到 LEC (Leclerc)
        fallback_a = 'LEC' if 'LEC' in drivers else drivers[1]
        print(f"車手 '{da}' 不在名單中，自動 Fallback 到 '{fallback_a}'")
        da = fallback_a
        
    if db not in drivers:
        fallback_b = 'VER' if 'VER' in drivers else drivers[0]
        print(f"車手 '{db}' 不在名單中，自動 Fallback 到 '{fallback_b}'")
        db = fallback_b
        
    print(f"開始提取 {da} 與 {db} 的最快圈...")
    lap_a = session.laps.pick_driver(da).pick_fastest()
    lap_b = session.laps.pick_driver(db).pick_fastest()
    
    telemetry_a = lap_a.get_telemetry()
    telemetry_b = lap_b.get_telemetry()
    
    # 建立與對齊遙測數據
    # 由於車手走線不同，每圈實際行駛總里程不同（ANT: 3269.8m, VER: 3292.2m）。
    # 為了讓終點時間差精確等於排位賽單圈差值 (0.043s)，我們將兩人的距離進行等比例歸一化重映射
    target_max_dist = 3270.0
    
    dist_a_normalized = (telemetry_a['Distance'] / telemetry_a['Distance'].max()) * target_max_dist
    dist_b_normalized = (telemetry_b['Distance'] / telemetry_b['Distance'].max()) * target_max_dist
    
    grid = np.arange(0, target_max_dist, 2.0) # 每 2 米一個點
    
    speed_a = np.interp(grid, dist_a_normalized, telemetry_a['Speed'])
    throttle_a = np.interp(grid, dist_a_normalized, telemetry_a['Throttle'])
    brake_a = np.interp(grid, dist_a_normalized, telemetry_a['Brake'].astype(float) * 100.0)
    time_a = np.interp(grid, dist_a_normalized, telemetry_a['Time'].dt.total_seconds())
    
    speed_b = np.interp(grid, dist_b_normalized, telemetry_b['Speed'])
    throttle_b = np.interp(grid, dist_b_normalized, telemetry_b['Throttle'])
    brake_b = np.interp(grid, dist_b_normalized, telemetry_b['Brake'].astype(float) * 100.0)
    time_b = np.interp(grid, dist_b_normalized, telemetry_b['Time'].dt.total_seconds())
    
    df_aligned_a = pd.DataFrame({
        'Distance': grid, 'Speed': speed_a, 'Throttle': throttle_a, 'Brake': brake_a, 'Time': time_a
    })
    df_aligned_b = pd.DataFrame({
        'Distance': grid, 'Speed': speed_b, 'Throttle': throttle_b, 'Brake': brake_b, 'Time': time_b
    })
    
    return df_aligned_a, df_aligned_b, da, db, year

def analyze_real_comparison(model_path, df_a, df_b, name_a, name_b):
    """
    使用神經網路對兩位真實車手的遙測進行雙向對比掃描，
    分析 A 相比於 B 的瓶頸，以及 B 相比於 A 的瓶頸。
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TelemetryAnalyzer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    n_points = len(df_a)
    seq_len = 100
    step = 15 # 每次前進 30m
    
    # 計算真實的時間差與速度差
    time_delta = df_a['Time'].values - df_b['Time'].values
    speed_delta = df_b['Speed'].values - df_a['Speed'].values
    
    print("\n" + "="*80)
    print(f"               AI TELEMETRY DETAILED ANALYSIS: {name_a} vs {name_b}               ")
    print("="*80)
    print(f"賽道總對齊長度: {df_a['Distance'].max():.1f} m")
    print(f"終點總時間差: {name_a} {'落後' if time_delta[-1] > 0 else '領先'} {name_b} {abs(time_delta[-1]):.3f} 秒")
    print("-"*80)
    
    # 進行雙向診斷
    # 方向 1: 分析 A (ANT) 相比於 B (VER) 的瓶頸 (B 是基準, A 是分析對象)
    bottlenecks_a = scan_bottlenecks(model, df_a, df_b, device, seq_len, step)
    
    # 方向 2: 分析 B (VER) 相比於 A (ANT) 的瓶頸 (A 是基準, B 是分析對象)
    # 這能幫我們找出 A (ANT) 相比於 B (VER) 的「優勢區段」！
    bottlenecks_b = scan_bottlenecks(model, df_b, df_a, device, seq_len, step)
    
    # 合併與輸出 A 的缺陷 (A 損失時間的地方)
    print(f"\n>>> 【{name_a} 表現落後 / 發生瓶頸的彎道區段】:")
    report_results(bottlenecks_a, name_a, name_b, df_a, df_b, prefix="bottleneck_a")
    
    # 合併與輸出 B 的缺陷 (A 贏時間的地方)
    print(f"\n>>> 【{name_a} 表現領先 / 優勢的彎道區段】:")
    report_results(bottlenecks_b, name_b, name_a, df_b, df_a, prefix="advantage_a")
    
    print("\n" + "="*80)
    print("摩納哥排位賽 AI 分析對照報告已全部生成！")
    print("="*80)

def scan_bottlenecks(model, df_target, df_base, device, seq_len, step):
    """
    掃描目標遙測相對於基準遙測的缺陷
    """
    detected = []
    n_points = len(df_target)
    
    for start_idx in range(0, n_points - seq_len, step):
        end_idx = start_idx + seq_len
        sub_t = df_target.iloc[start_idx:end_idx].reset_index(drop=True)
        sub_b = df_base.iloc[start_idx:end_idx].reset_index(drop=True)
        
        # 組合輸入 [Speed_Target, Throttle_Target, Brake_Target, Speed_Base, Throttle_Base, Brake_Base]
        feat_t = sub_t[['Speed', 'Throttle', 'Brake']].values
        feat_b = sub_b[['Speed', 'Throttle', 'Brake']].values
        features = np.hstack([feat_t, feat_b])
        
        x_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            logits, reg_preds, attn_weights = model(x_tensor)
            cls_pred = torch.argmax(logits, dim=1).item()
            reg_pred = reg_preds[0].cpu().numpy() # [brake_delta, exit_speed_delta, time_loss]
            attn = attn_weights[0, :, 0].cpu().numpy()
            
        brake_delta, exit_speed_delta, time_loss = reg_pred
        
        # 如果不是 normal 且預測有時間損失
        if cls_pred != 0 and time_loss > 0.04:
            max_attn_idx = np.argmax(attn)
            key_distance = sub_t.loc[max_attn_idx, 'Distance']
            
            detected.append({
                'start_dist': sub_t['Distance'].min(),
                'end_dist': sub_t['Distance'].max(),
                'key_dist': key_distance,
                'class': cls_pred,
                'brake_delta': brake_delta,
                'exit_speed_delta': exit_speed_delta,
                'time_loss': time_loss,
                'attn_weights': attn,
                'sub_target': sub_t,
                'sub_base': sub_b
            })
            
    # NMS 合併
    merged = []
    if len(detected) > 0:
        detected.sort(key=lambda x: x['time_loss'], reverse=True)
        for item in detected:
            overlap = False
            for m in merged:
                if abs((item['start_dist'] + item['end_dist'])/2.0 - (m['start_dist'] + m['end_dist'])/2.0) < 180.0:
                    overlap = True
                    break
            if not overlap:
                merged.append(item)
    
    merged.sort(key=lambda x: x['start_dist'])
    return merged

def report_results(bottlenecks, name_a, name_b, df_a, df_b, prefix):
    """
    印出報告並繪製圖表
    """
    class_names = {1: "晚煞車 (Late Braking)", 2: "早煞車 (Early Braking)", 3: "出彎給油猶豫 (Slow Throttle)"}
    
    if len(bottlenecks) == 0:
        print("  (未偵測到顯著的行為差異)")
        return
        
    for i, item in enumerate(bottlenecks):
        cls_idx = item['class']
        cls_name = class_names.get(cls_idx, "駕駛操作差異")
        
        # 由於是雙向對比，若是 advantage_a，表示 A 領先 B，那麼報告描述中主角與配角身份需要對換
        is_adv = prefix.startswith("advantage_a")
        
        main_driver = name_a if not is_adv else name_b
        comp_driver = name_b if not is_adv else name_a
        
        print(f"\n  [區段 #{i+1}] {item['start_dist']:.0f}m - {item['end_dist']:.0f}m (關鍵失誤點: {item['key_dist']:.0f}m)")
        print(f"  > AI 診斷: {main_driver} 相比於 {comp_driver} {cls_name}")
        
        # 動態計算這段區間真實的 delta，增加報告的真實物理精確度
        sub_a_segment = df_a[(df_a['Distance'] >= item['start_dist']) & (df_a['Distance'] <= item['end_dist'])]
        sub_b_segment = df_b[(df_b['Distance'] >= item['start_dist']) & (df_b['Distance'] <= item['end_dist'])]
        
        real_time_loss = abs(sub_a_segment['Time'].iloc[-1] - sub_b_segment['Time'].iloc[-1] - 
                             (sub_a_segment['Time'].iloc[0] - sub_b_segment['Time'].iloc[0]))
        
        # 輸出自然語言
        if cls_idx == 1: # Late Braking
            print(f"  > AI 分析: 「在此連續彎道中，{main_driver} 因為晚煞車了 {abs(item['brake_delta']):.2f} 秒，"
                  f"導致後續出彎速度比 {comp_driver} 慢了 {abs(item['exit_speed_delta']):.1f} km/h，該區段損失 {real_time_loss:.3f} 秒。」")
        elif cls_idx == 2: # Early Braking
            print(f"  > AI 分析: 「在此彎道中，{main_driver} 提早煞車了 {abs(item['brake_delta']):.2f} 秒，"
                  f"入彎速度滑落較早，導致此區段損失 {real_time_loss:.3f} 秒。」")
        elif cls_idx == 3: # Slow Throttle
            print(f"  > AI 分析: 「在此出彎處，{main_driver} 踩油門加速較為遲緩，出彎速度比 {comp_driver} 慢了 {abs(item['exit_speed_delta']):.1f} km/h，"
                  f"該區段損失 {real_time_loss:.3f} 秒。」")
            
        # 繪製真實對比圖表
        plot_real_comparison(item, i+1, name_a, name_b, prefix)

def plot_real_comparison(item, num, name_a, name_b, prefix):
    """
    繪製真實 F1 遙測對照圖
    """
    sub_target = item['sub_target']
    sub_base = item['sub_base']
    attn = item['attn_weights']
    
    # 決定 A 與 B 在圖表中的顯示
    # 我們讓 Driver A 始終代表 ANT，Driver B 始終代表 VER
    is_adv = prefix.startswith("advantage_a")
    df_ant = sub_base if is_adv else sub_target
    df_ver = sub_target if is_adv else sub_base
    
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    
    # 1. 速度
    axes[0].plot(df_ver['Distance'], df_ver['Speed'], 'g-', label=f'{name_b} (VER)', linewidth=2)
    axes[0].plot(df_ant['Distance'], df_ant['Speed'], 'r--', label=f'{name_a} (ANT)', linewidth=2)
    axes[0].set_ylabel('Speed (km/h)')
    axes[0].set_title(f"Monaco GP {name_a} vs {name_b} Comparison - Section #{num} ({prefix.upper()})")
    axes[0].grid(True)
    axes[0].legend()
    
    # 2. 油門
    axes[1].plot(df_ver['Distance'], df_ver['Throttle'], 'g-', label=name_b, linewidth=1.5)
    axes[1].plot(df_ant['Distance'], df_ant['Throttle'], 'r--', label=name_a, linewidth=1.5)
    axes[1].set_ylabel('Throttle (%)')
    axes[1].grid(True)
    
    # 3. 煞車
    axes[2].plot(df_ver['Distance'], df_ver['Brake'], 'g-', label=name_b, linewidth=1.5)
    axes[2].plot(df_ant['Distance'], df_ant['Brake'], 'r--', label=name_a, linewidth=1.5)
    axes[2].set_ylabel('Brake (%)')
    axes[2].grid(True)
    
    # 4. Attention
    axes[3].fill_between(df_ant['Distance'], attn, color='purple', alpha=0.3, label='AI Attention Weight')
    axes[3].plot(df_ant['Distance'], attn, color='purple', linewidth=1.5)
    axes[3].axvline(x=item['key_dist'], color='blue', linestyle=':', label='Key Telemetry Delta Point')
    axes[3].set_ylabel('Attention')
    axes[3].set_xlabel('Distance (m)')
    axes[3].grid(True)
    axes[3].legend()
    
    plt.tight_layout()
    output_dir = os.path.join(os.path.dirname(__file__), 'monaco_comparison_plots')
    os.makedirs(output_dir, exist_ok=True)
    plot_name = f"monaco_{prefix}_{num}.png"
    plot_path = os.path.join(output_dir, plot_name)
    plt.savefig(plot_path, dpi=120)
    plt.close()
    print(f"  > 對照圖表已儲存至: {plot_path}")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'telemetry_model.pth')
    
    if not os.path.exists(model_path):
        print(f"錯誤: 找不到已訓練的模型 {model_path}，請先運行 train.py！")
    else:
        # 下載並對齊真實遙測
        try:
            df_a, df_b, actual_a, actual_b, actual_year = load_session_and_laps(
                year=2026, gp='Monaco', driver_a='ANT', driver_b='VER'
            )
            analyze_real_comparison(model_path, df_a, df_b, actual_a, actual_b)
        except Exception as e:
            print(f"\n無法載入真實遙測數據: {e}")
            print("啟動 fallback：使用 2023 賽季排位賽真實遙測 (VER vs HAM) 進行分析展示...")
            # Fallback 到 2023 西班牙站 VER 與 HAM 的最快單圈
            try:
                df_a, df_b, actual_a, actual_b, actual_year = load_session_and_laps(
                    year=2023, gp='Spain', driver_a='HAM', driver_b='VER'
                )
                analyze_real_comparison(model_path, df_a, df_b, actual_a, actual_b)
            except Exception as ex:
                print(f"Fallback 載入真實數據亦失敗: {ex}。將終止分析。")
