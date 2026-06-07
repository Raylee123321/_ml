import json

json_path = r"e:\co\_ml\mid\web\telemetry_data.json"
with open(json_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

print(data['info'])
