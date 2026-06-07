import json

json_path = r"e:\co\_ml\mid\web\telemetry_data.json"
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

telemetry = data['telemetry']
playtime_a = telemetry['playtime_a']
playtime_b = telemetry['playtime_b']

print("First points:")
print(f"PlayTime A[0]: {playtime_a[0]:.6f} | PlayTime B[0]: {playtime_b[0]:.6f} | Delta: {playtime_a[0]-playtime_b[0]:.6f}")

print("\nLast points:")
print(f"PlayTime A[-1]: {playtime_a[-1]:.6f} | PlayTime B[-1]: {playtime_b[-1]:.6f} | Delta: {playtime_a[-1]-playtime_b[-1]:.6f}")
