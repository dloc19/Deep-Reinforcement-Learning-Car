#!/usr/bin/env python3
"""Trích xuất tọa độ (x, y, z, yaw) của tất cả spawn points dùng trong các kịch bản
đánh giá route. Yêu cầu CARLA server đang chạy.

Cách dùng:
    1. Mở CARLA server: CarlaUE4.exe
    2. Chạy script này:
       python export_spawn_coords.py --output spawn_coordinates.csv

Script sẽ tự nạp từng Town (01-05), đọc spawn points, và ghi ra file CSV
với tọa độ chính xác.
"""
import argparse
import csv
import os
import sys
import time

# Thêm path CARLA Python API
CARLA_EGG = r'C:\Users\dloc\Desktop\Do_An\CARLA_0.9.10\WindowsNoEditor\PythonAPI\carla\dist'
eggs = [f for f in os.listdir(CARLA_EGG) if f.endswith('.egg')]
if eggs:
    sys.path.insert(0, os.path.join(CARLA_EGG, eggs[0]))

try:
    import carla
except ImportError:
    print("Khong import duoc 'carla'. Kiem tra CARLA Python API path.")
    sys.exit(1)

# Spawn point indices dùng trong đánh giá route (từ file route_Town*.csv)
SPAWN_INDICES = {
    'Town01': [6, 8, 34, 56, 73, 80, 87, 112, 113, 153, 155, 169, 173, 187, 199, 210, 228, 237, 238, 244],
    'Town02': [13, 26, 27, 33, 36, 44, 47, 49, 59, 61, 64, 71, 75, 76, 80, 83],
    'Town03': [36, 37, 38, 42, 44, 77, 101, 112, 122, 124, 126, 142, 194, 201, 206, 210, 218, 236, 240, 264],
    'Town04': [35, 116, 125, 126, 133, 167, 169, 178, 179, 182, 193, 196, 254, 263, 272, 275, 276, 336, 341, 362],
    'Town05': [31, 34, 40, 55, 56, 70, 85, 103, 122, 132, 138, 143, 164, 187, 202, 215, 242, 276, 285],
}

# Route lengths from CSV
ROUTE_LENGTHS = {
    'Town01': {6:82.0,8:203.3,34:178.1,56:97.6,73:138.0,80:82.0,87:218.6,112:166.6,113:135.8,153:90.0,155:135.4,169:161.4,173:156.2,187:107.3,199:204.5,210:178.7,228:172.8,237:171.5,238:142.4,244:119.0},
    'Town02': {13:192.1,26:203.0,27:164.4,33:179.1,36:151.7,44:94.5,47:110.5,49:163.8,59:180.7,61:168.4,64:167.3,71:212.5,75:176.6,76:168.6,80:191.0,83:211.2},
    'Town03': {36:198.1,37:175.7,38:102.7,42:91.6,44:195.1,77:127.6,101:160.0,112:214.0,122:157.9,124:177.1,126:217.5,142:219.1,194:156.0,201:173.7,206:127.7,210:193.1,218:134.0,236:88.7,240:177.2,264:97.5},
    'Town04': {35:132.3,116:84.6,125:148.1,126:125.1,133:84.3,167:195.9,169:202.8,178:210.8,179:145.1,182:182.1,193:216.9,196:91.5,254:210.0,263:163.0,272:157.6,275:104.0,276:120.1,336:183.4,341:134.5,362:183.4},
    'Town05': {31:213.4,34:183.5,40:211.2,55:194.5,56:181.5,70:194.3,85:117.8,103:96.7,122:152.9,132:181.1,138:93.4,143:82.5,164:151.4,187:148.3,202:117.9,215:162.8,242:99.6,276:158.2,285:180.8},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--output', default='spawn_coordinates.csv')
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)

    rows = []
    for town in ['Town01', 'Town02', 'Town03', 'Town04', 'Town05']:
        print(f"Dang nap {town}...")
        client.load_world(town)
        time.sleep(3.0)  # Doi world nap xong

        world = client.get_world()
        world_map = world.get_map()
        spawn_points = world_map.get_spawn_points()

        print(f"  {town}: {len(spawn_points)} spawn points")
        indices = SPAWN_INDICES.get(town, [])
        for idx in indices:
            if idx < len(spawn_points):
                t = spawn_points[idx]
                route_len = ROUTE_LENGTHS.get(town, {}).get(idx, 0)
                rows.append({
                    'town': town,
                    'spawn_index': idx,
                    'x': round(t.location.x, 2),
                    'y': round(t.location.y, 2),
                    'z': round(t.location.z, 2),
                    'yaw': round(t.rotation.yaw, 2),
                    'route_length_m': route_len,
                })
                print(f"    [{idx:3d}] x={t.location.x:8.2f} y={t.location.y:8.2f} "
                      f"z={t.location.z:6.2f} yaw={t.rotation.yaw:7.2f}  route={route_len:.0f}m")
            else:
                print(f"    [{idx:3d}] KHONG HOP LE (chi co {len(spawn_points)} spawn points)")

    # Ghi CSV
    with open(args.output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['town','spawn_index','x','y','z','yaw','route_length_m'])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDa ghi {len(rows)} dong -> {args.output}")


if __name__ == '__main__':
    main()
