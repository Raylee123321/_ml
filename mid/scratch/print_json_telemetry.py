import json

json_path = r"e:\co\_ml\mid\web\telemetry_data.json"
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

telemetry = data['telemetry']
dist = telemetry['distance']
throttle_a = telemetry['throttle_a']
brake_a = telemetry['brake_a']
throttle_b = telemetry['throttle_b']
brake_b = telemetry['brake_b']

for i in range(len(dist)):
    d = dist[i]
    if 2040 <= d <= 2120:
        print(f"Dist: {d:.1f} | Throttle_A: {throttle_a[i]:.2f}, Brake_A: {brake_a[i]:.2f} | Throttle_B: {throttle_b[i]:.2f}, Brake_B: {brake_b[i]:.2f}")
