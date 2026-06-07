import json
import numpy as np

json_path = r"e:\co\_ml\mid\web\telemetry_data.json"
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

telemetry = data['telemetry']
dist = np.array(telemetry['distance'])
speed_a = np.array(telemetry['speed_a'])
speed_b = np.array(telemetry['speed_b'])

# Find local minima in Speed A to find corner apexes
from scipy.signal import find_peaks
# Invert speed to find minima
peaks, _ = find_peaks(-speed_a, distance=25, prominence=5)

print("=== Detected Corner Apexes (Speed Minima) ===")
for p in peaks:
    d = dist[p]
    s_a = speed_a[p]
    s_b = speed_b[p]
    print(f"Apex at Dist: {d:.1f} m | Speed A: {s_a:.1f} km/h | Speed B: {s_b:.1f} km/h")
