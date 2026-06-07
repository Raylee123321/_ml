import os
import numpy as np
import pandas as pd
import fastf1

# Setup Cache
CACHE_DIR = r"e:\co\_ml\mid\fastf1_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
fastf1.Cache.enable_cache(CACHE_DIR)

# Load Q3 fastest laps
session = fastf1.get_session(2026, 'Monaco', 'Q')
session.load()

lap_a = session.laps.pick_driver('ANT').pick_fastest()
lap_b = session.laps.pick_driver('VER').pick_fastest()

telemetry_a = lap_a.get_telemetry()
telemetry_b = lap_b.get_telemetry()

# Print telemetry around 2000m - 2150m for ANT (Driver A)
print("=== ANT Raw Telemetry around 2000m - 2150m ===")
ant_sub = telemetry_a[(telemetry_a['Distance'] >= 2000) & (telemetry_a['Distance'] <= 2150)]
print(ant_sub[['Distance', 'Speed', 'Throttle', 'Brake']].to_string())

# Print telemetry around 2000m - 2150m for VER (Driver B)
print("\n=== VER Raw Telemetry around 2000m - 2150m ===")
ver_sub = telemetry_b[(telemetry_b['Distance'] >= 2000) & (telemetry_b['Distance'] <= 2150)]
print(ver_sub[['Distance', 'Speed', 'Throttle', 'Brake']].to_string())
