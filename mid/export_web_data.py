import os
import json
import numpy as np
import pandas as pd
import torch

from model import TelemetryAnalyzer
from compare_monaco import load_session_and_laps, scan_bottlenecks

def get_corner_name(gp, distance):
    """
    根據賽道 GP 和當前距離點，判斷對應的彎道名稱/編號。
    """
    gp_lower = gp.lower()
    
    if 'monaco' in gp_lower:
        # 摩納哥 (Monaco) 19個彎道對照
        corners = [
            {'name': 'Turn 1 (Sainte Devote)', 'range': [100, 350]},
            {'name': 'Turn 3 (Massenet)', 'range': [650, 820]},
            {'name': 'Turn 4 (Casino)', 'range': [820, 950]},
            {'name': 'Turn 5-7 (Mirabeau/Hairpin)', 'range': [1000, 1340]},
            {'name': 'Turn 8 (Portier)', 'range': [1340, 1450]},
            {'name': 'Turn 9 (Tunnel)', 'range': [1450, 1950]},
            {'name': 'Turn 10-11 (Nouvelle Chicane)', 'range': [1950, 2150]},
            {'name': 'Turn 12 (Tabac)', 'range': [2200, 2450]},
            {'name': 'Turn 13-14 (Louis Chiron)', 'range': [2500, 2750]},
            {'name': 'Turn 15-16 (Swimming Pool)', 'range': [2750, 2900]},
            {'name': 'Turn 17-18 (Rascasse)', 'range': [2900, 3050]},
            {'name': 'Turn 19 (Antony Noghes)', 'range': [3050, 3268]}
        ]
    elif 'spain' in gp_lower or 'catalunya' in gp_lower:
        # 加泰隆尼亞 (Spain) 彎道對照
        corners = [
            {'name': 'Turn 1-2 (Elf)', 'range': [500, 750]},
            {'name': 'Turn 3 (Renault)', 'range': [800, 1100]},
            {'name': 'Turn 4 (Repsol)', 'range': [1200, 1450]},
            {'name': 'Turn 5', 'range': [1550, 1750]},
            {'name': 'Turn 7-8 (Wurth)', 'range': [2000, 2300]},
            {'name': 'Turn 9 (Campsa)', 'range': [2400, 2600]},
            {'name': 'Turn 10 (La Caixa)', 'range': [2800, 3100]},
            {'name': 'Turn 12', 'range': [3300, 3600]},
            {'name': 'Turn 13-14 (New Chicane/Final)', 'range': [3700, 4000]}
        ]
    else:
        return "彎道 (Corner)"
        
    for c in corners:
        if c['range'][0] <= distance <= c['range'][1]:
            return c['name']
            
    return "直線 / 銜接段"

def export_data(model_path, web_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TelemetryAnalyzer().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # 1. 嘗試加載真實遙測（預設使用 2026 Monaco GP ANT vs VER，若無則 fallback）
    print("正在下載/加載遙測數據...")
    try:
        df_a, df_b, driver_a, driver_b, year = load_session_and_laps(
            year=2026, gp='Monaco', driver_a='ANT', driver_b='VER'
        )
        gp_name = 'Monaco'
        race_info = f"{year} Monaco Grand Prix Qualifying - {driver_a} vs {driver_b}"
    except Exception as e:
        print(f"加載 2026 數據失敗，改用西班牙站 fallback: {e}")
        df_a, df_b, driver_a, driver_b, year = load_session_and_laps(
            year=2023, gp='Spain', driver_a='HAM', driver_b='VER'
        )
        gp_name = 'Spain'
        race_info = f"{year} Spanish Grand Prix Qualifying - {driver_a} vs {driver_b}"
        
    print(f"成功加載數據: {race_info}")
    
    # 由於影片播放進度需要對照到 Driver A 的 Time，我們將 Time_A 轉為從 0 開始的相對播放時間
    time_a_start = df_a['Time'].iloc[0]
    df_a['PlayTime'] = df_a['Time'] - time_a_start
    # 同理，Driver B 也是相對於 A 的起點，以維持相對時間差
    df_b['PlayTime'] = df_b['Time'] - time_a_start
    
    # 2. 進行模型瓶頸掃描
    print("正在掃描駕駛行為瓶頸與優勢...")
    seq_len = 100
    step = 15
    
    # A (ANT) 相比於 B (VER) 的瓶頸
    bottlenecks = scan_bottlenecks(model, df_a, df_b, device, seq_len, step)
    # B (VER) 相比於 A (ANT) 的瓶頸 (即 A 的優勢)
    advantages = scan_bottlenecks(model, df_b, df_a, device, seq_len, step)
    
    # 整理瓶頸數據為前端 JSON 格式
    class_names = {1: "Late Braking", 2: "Early Braking", 3: "Slow Throttle"}
    class_names_zh = {1: "晚煞車 (Late Braking)", 2: "早煞車 (Early Braking)", 3: "出彎給油猶豫 (Slow Throttle)"}
    
    events_json = []
    
    # 處理 A 的缺陷 (A 損失時間的地方)
    for i, item in enumerate(bottlenecks):
        cls_idx = item['class']
        # 尋找對應的 PlayTime 戳記，以便影片跳轉
        sub_a = item['sub_target']
        # 關鍵點在 A 遙測中對應的 PlayTime
        key_idx = (df_a['Distance'] - item['key_dist']).abs().idxmin()
        key_playtime = df_a.loc[key_idx, 'PlayTime']
        
        start_idx = (df_a['Distance'] - item['start_dist']).abs().idxmin()
        end_idx = (df_a['Distance'] - item['end_dist']).abs().idxmin()
        
        # 計算此區域真實損失時間
        sub_a_segment = df_a.iloc[start_idx:end_idx+1]
        sub_b_segment = df_b.iloc[start_idx:end_idx+1]
        real_time_loss = abs(sub_a_segment['Time'].iloc[-1] - sub_b_segment['Time'].iloc[-1] - 
                             (sub_a_segment['Time'].iloc[0] - sub_b_segment['Time'].iloc[0]))
        
        # 生成 AI 診斷自然語言描述
        if cls_idx == 1:
            desc = f"在此連續彎道中，{driver_a} 因為晚煞車了 {abs(item['brake_delta']):.2f} 秒，導致出彎速度慢了 {abs(item['exit_speed_delta']):.1f} km/h，損失 {real_time_loss:.3f} 秒。"
        elif cls_idx == 2:
            desc = f"在此彎道中，{driver_a} 提早煞車了 {abs(item['brake_delta']):.2f} 秒，入彎速度滑落較早，損失 {real_time_loss:.3f} 秒。"
        else:
            desc = f"在此出彎處，{driver_a} 踩油門加速較為遲緩，出彎速度慢了 {abs(item['exit_speed_delta']):.1f} km/h，損失 {real_time_loss:.3f} 秒。"
            
        events_json.append({
            'id': f"bottleneck_{i+1}",
            'type': 'bottleneck',
            'driver': driver_a,
            'opposing_driver': driver_b,
            'corner': get_corner_name(gp_name, item['key_dist']),
            'start_dist': float(item['start_dist']),
            'end_dist': float(item['end_dist']),
            'key_dist': float(item['key_dist']),
            'key_playtime': float(key_playtime),
            'class_id': int(cls_idx),
            'class_name': class_names[cls_idx],
            'class_name_zh': class_names_zh[cls_idx],
            'brake_delta': float(item['brake_delta']),
            'exit_speed_delta': float(item['exit_speed_delta']),
            'time_loss': float(real_time_loss),
            'description': desc,
            'attn': item['attn_weights'].tolist() # 自注意力權重
        })
        
    # 處理 A 的優勢 (B 損失時間的地方)
    for i, item in enumerate(advantages):
        cls_idx = item['class']
        key_idx = (df_a['Distance'] - item['key_dist']).abs().idxmin()
        key_playtime = df_a.loc[key_idx, 'PlayTime']
        
        start_idx = (df_a['Distance'] - item['start_dist']).abs().idxmin()
        end_idx = (df_a['Distance'] - item['end_dist']).abs().idxmin()
        
        sub_a_segment = df_a.iloc[start_idx:end_idx+1]
        sub_b_segment = df_b.iloc[start_idx:end_idx+1]
        real_time_loss = abs(sub_a_segment['Time'].iloc[-1] - sub_b_segment['Time'].iloc[-1] - 
                             (sub_a_segment['Time'].iloc[0] - sub_b_segment['Time'].iloc[0]))
        
        if cls_idx == 1:
            desc = f"此區段為 {driver_a} 的優勢區！{driver_b} 在此晚煞車了 {abs(item['brake_delta']):.2f} 秒，導致出彎速度偏慢，被 {driver_a} 追回了 {real_time_loss:.3f} 秒。"
        elif cls_idx == 2:
            desc = f"此區段為 {driver_a} 的優勢區！{driver_b} 提早煞車了 {abs(item['brake_delta']):.2f} 秒，{driver_a} 帶著更多速度入彎，領先 {real_time_loss:.3f} 秒。"
        else:
            desc = f"此區段為 {driver_a} 的優勢區！{driver_b} 踩油門加速較為遲緩，{driver_a} 的加速效率更高，領先 {real_time_loss:.3f} 秒。"
            
        events_json.append({
            'id': f"advantage_{i+1}",
            'type': 'advantage',
            'driver': driver_a,
            'opposing_driver': driver_b,
            'corner': get_corner_name(gp_name, item['key_dist']),
            'start_dist': float(item['start_dist']),
            'end_dist': float(item['end_dist']),
            'key_dist': float(item['key_dist']),
            'key_playtime': float(key_playtime),
            'class_id': int(cls_idx),
            'class_name': class_names[cls_idx],
            'class_name_zh': "優勢: " + class_names_zh[cls_idx],
            'brake_delta': float(item['brake_delta']),
            'exit_speed_delta': float(item['exit_speed_delta']),
            'time_loss': float(real_time_loss),
            'description': desc,
            'attn': item['attn_weights'].tolist()
        })
        
    # 按照距離順序排序所有事件
    events_json.sort(key=lambda x: x['start_dist'])
    
    # 3. 整理整圈遙測數據
    telemetry_json = {
        'info': {
            'race_info': race_info,
            'driver_a': driver_a,
            'driver_b': driver_b,
            'max_distance': float(df_a['Distance'].max()),
            'max_playtime': float(df_a['PlayTime'].max())
        },
        'telemetry': {
            'distance': df_a['Distance'].tolist(),
            'playtime_a': df_a['PlayTime'].tolist(),
            'playtime_b': df_b['PlayTime'].tolist(),
            'speed_a': df_a['Speed'].tolist(),
            'speed_b': df_b['Speed'].tolist(),
            'throttle_a': df_a['Throttle'].tolist(),
            'throttle_b': df_b['Throttle'].tolist(),
            'brake_a': df_a['Brake'].tolist(),
            'brake_b': df_b['Brake'].tolist(),
        },
        'events': events_json
    }
    
    os.makedirs(web_dir, exist_ok=True)
    json_path = os.path.join(web_dir, 'telemetry_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(telemetry_json, f, indent=2, ensure_ascii=False)
        
    print(f"成功導出網頁遙測數據至: {json_path}")

if __name__ == '__main__':
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, 'telemetry_model.pth')
    web_dir = os.path.join(current_dir, 'web')
    
    export_data(model_path, web_dir)
