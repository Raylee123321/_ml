import json

json_path = r"e:\co\_ml\mid\web\telemetry_data.json"
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

telemetry = data['telemetry']
dist = telemetry['distance']
brake_a = telemetry['brake_a']
brake_b = telemetry['brake_b']

for i in range(len(dist)):
    d = dist[i]
    if 1950 <= d <= 2050:
        print(f"Dist: {d:.1f} | Brake_A: {brake_a[i]:.2f} | Brake_B: {brake_b[i]:.2f}")
