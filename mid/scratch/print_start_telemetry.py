import json

json_path = r"e:\co\_ml\mid\web\telemetry_data.json"
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

telemetry = data['telemetry']
dist = telemetry['distance']
speed_a = telemetry['speed_a']
throttle_a = telemetry['throttle_a']
brake_a = telemetry['brake_a']

for i in range(len(dist)):
    d = dist[i]
    if d <= 100:
        print(f"Dist: {d:.1f} | Speed_A: {speed_a[i]:.1f} | Throttle_A: {throttle_a[i]:.1f} | Brake_A: {brake_a[i]:.1f}")
