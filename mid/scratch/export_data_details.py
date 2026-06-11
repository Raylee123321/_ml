import os
import numpy as np
import pandas as pd

def main():
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(current_dir, 'data')
    output_dir = os.path.join(current_dir, 'exported_training_data')
    samples_dir = os.path.join(output_dir, 'samples')
    
    os.makedirs(samples_dir, exist_ok=True)
    
    # 載入 NumPy 數據
    X = np.load(os.path.join(data_dir, 'X.npy'))
    Y_cls = np.load(os.path.join(data_dir, 'Y_cls.npy'))
    Y_reg = np.load(os.path.join(data_dir, 'Y_reg.npy'))
    
    class_names = {
        0: "Normal (正常)",
        1: "Late Braking (晚煞車)",
        2: "Early Braking (早煞車)",
        3: "Slow Throttle (出彎給油慢)"
    }
    
    # 1. 建立 summary 資訊
    summary_data = []
    
    for i in range(len(X)):
        sample_id = f"sample_{i+1:03d}"
        class_id = int(Y_cls[i])
        class_name = class_names.get(class_id, "Unknown")
        brake_delta, exit_speed_delta, time_loss = Y_reg[i]
        
        summary_data.append({
            'Sample_ID': sample_id,
            'Class_ID': class_id,
            'Class_Name': class_name,
            'Brake_Delta_S': brake_delta,
            'Exit_Speed_Delta_KMH': exit_speed_delta,
            'Time_Loss_S': time_loss
        })
        
        # 2. 導出單個樣本的時序遙測數據
        # X[i] 維度: (100, 6)
        # 前 3 行為 Driver A，後 3 行為 Driver B
        sample_df = pd.DataFrame(X[i], columns=[
            'Speed_A', 'Throttle_A', 'Brake_A',
            'Speed_B', 'Throttle_B', 'Brake_B'
        ])
        sample_df.index.name = 'Step'
        sample_csv_path = os.path.join(samples_dir, f"{sample_id}.csv")
        sample_df.to_csv(sample_csv_path)
        
    summary_df = pd.DataFrame(summary_data)
    summary_csv_path = os.path.join(output_dir, 'summary.csv')
    summary_df.to_csv(summary_csv_path, index=False)
    
    print(f"成功導出數據！")
    print(f"總樣本數: {len(X)}")
    print(f"摘要檔已儲存至: {summary_csv_path}")
    print(f"詳細時序樣本已儲存至: {samples_dir}")

if __name__ == '__main__':
    main()
