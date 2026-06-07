import os
import fastf1

CACHE_DIR = r"e:\co\_ml\mid\fastf1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

session = fastf1.get_session(2026, 'Monaco', 'Q')
session.load()

lap_a = session.laps.pick_driver('ANT').pick_fastest()
lap_b = session.laps.pick_driver('VER').pick_fastest()

print("=== Lap Metadata ===")
print(f"Driver A (ANT) LapTime: {lap_a.LapTime}")
print(f"Driver B (VER) LapTime: {lap_b.LapTime}")
print(f"Real LapTime Delta: {(lap_a.LapTime - lap_b.LapTime).total_seconds():.6f} seconds")

telemetry_a = lap_a.get_telemetry()
telemetry_b = lap_b.get_telemetry()

print("\n=== Raw Telemetry Time Range ===")
print(f"Driver A (ANT) Telemetry Time Max: {telemetry_a['Time'].max()} | Min: {telemetry_a['Time'].min()}")
print(f"Driver B (VER) Telemetry Time Max: {telemetry_b['Time'].max()} | Min: {telemetry_b['Time'].min()}")
print(f"Raw Telemetry Delta at Max Time: {(telemetry_a['Time'].max() - telemetry_b['Time'].max()).total_seconds():.6f} seconds")

print("\n=== Raw Telemetry Distance Range ===")
print(f"Driver A (ANT) Distance Max: {telemetry_a['Distance'].max():.1f} | Min: {telemetry_a['Distance'].min():.1f}")
print(f"Driver B (VER) Distance Max: {telemetry_b['Distance'].max():.1f} | Min: {telemetry_b['Distance'].min():.1f}")
