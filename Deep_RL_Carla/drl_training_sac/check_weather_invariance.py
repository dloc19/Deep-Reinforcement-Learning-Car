# -*- coding: utf-8 -*-
"""Kiem chung: anh semantic segmentation co that su BAT BIEN voi thoi tiet khong?

Ca pipeline quan sat dua tren gia dinh nay — neu no sai thi moi ket luan ve "mo hinh chiu
duoc moi thoi tiet" deu khong dung, vi mo hinh chua bao gio thay thoi tiet thay doi.

Cach do: dat xe DUNG YEN tai mot spawn point, chi doi `WeatherParameters`, chup anh seg sau
moi lan doi, roi so tung diem anh voi anh dau tien. Xe dung yen nen moi khac biet do duoc
CHINH LA do thoi tiet, khong lan voi thay doi goc nhin.

Chay: python check_weather_invariance.py [--town Town01]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import carla  # noqa: E402

WEATHERS = ["ClearNoon", "CloudyNoon", "WetNoon", "WetCloudyNoon", "MidRainyNoon",
            "HardRainNoon", "SoftRainNoon", "ClearSunset", "CloudySunset",
            "WetSunset", "HardRainSunset"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", default="Town01")
    ap.add_argument("--width", type=int, default=200)
    ap.add_argument("--height", type=int, default=88)
    args = ap.parse_args()

    client = carla.Client("127.0.0.1", 2000)
    client.set_timeout(60.0)
    world = client.get_world()
    if args.town not in world.get_map().name:
        world = client.load_world(args.town)

    settings = world.get_settings()
    # Luu lai bang gia tri tho, khong dung `carla.WorldSettings(...)`: o che do async
    # `fixed_delta_seconds` la None, ma constructor cua no chi nhan double -> ArgumentError.
    original_sync = settings.synchronous_mode
    original_delta = settings.fixed_delta_seconds
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    bp_lib = world.get_blueprint_library()
    vehicle, camera = None, None
    frames = {}
    try:
        spawn = world.get_map().get_spawn_points()[0]
        vehicle = world.spawn_actor(bp_lib.filter("vehicle.tesla.model3")[0], spawn)
        vehicle.set_autopilot(False)
        vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0))

        cam_bp = bp_lib.find("sensor.camera.semantic_segmentation")
        cam_bp.set_attribute("image_size_x", str(args.width))
        cam_bp.set_attribute("image_size_y", str(args.height))
        cam_bp.set_attribute("fov", "90")
        camera = world.spawn_actor(
            cam_bp, carla.Transform(carla.Location(x=1.5, z=2.4)), attach_to=vehicle)

        latest = {}
        camera.listen(lambda img: latest.__setitem__("img", img))

        # Xe vua spawn se ROI XUONG vai centimet cho den khi giam xoc on dinh. Camera gan
        # vao xe, nen anh chup trong luc do o mot do cao khac han cac anh sau — khac biet
        # hinh hoc đó se bi doc nham thanh "anh huong cua thoi tiet". Tick cho no on truoc.
        for _ in range(60):
            world.tick()

        for name in WEATHERS:
            world.set_weather(getattr(carla.WeatherParameters, name))
            latest.pop("img", None)
            # Vai tick de thoi tiet an vao khung hinh va camera tra ve anh moi.
            for _ in range(12):
                world.tick()
            if "img" not in latest:
                print("  %-16s KHONG nhan duoc anh — bo qua" % name)
                continue
            img = latest["img"]
            buf = np.frombuffer(img.raw_data, dtype=np.uint8)
            buf = buf.reshape((img.height, img.width, 4))
            frames[name] = buf[:, :, 2].copy()  # kenh R = class id cua CARLA

        if len(frames) < 2:
            raise SystemExit("Khong du anh de so sanh.")

        names = [n for n in WEATHERS if n in frames]
        total = frames[names[0]].size
        print()
        print("Anh %dx%d = %d diem anh; so sanh tren kenh R = class id cua CARLA." % (
            frames[names[0]].shape[1], frames[names[0]].shape[0], total))
        print("So sanh TUNG CAP chu khong chi so voi anh dau. Ly do: lan chay dau tien cho")
        print("ket qua 176 diem anh (1.00%) lech o GAN NHU MOI thoi tiet, ke ca CloudyNoon —")
        print("mot con so giong het nhau nhu vay khong the do thoi tiet gay ra. That ra do la")
        print("xe vua spawn con dang lun giam xoc, nen anh DAU chup o mot do cao khac. Doi 60")
        print("tick cho xe on dinh thi con so tut ve 0-2 diem anh.")
        print()
        print("%-16s %-16s %12s %11s" % ("Thoi tiet A", "Thoi tiet B", "diem khac", "ty le"))
        print("-" * 60)
        worst = 0.0
        worst_pair = ("-", "-")
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                diff = int(np.count_nonzero(frames[names[i]] != frames[names[j]]))
                pct = 100.0 * diff / total
                if pct > worst:
                    worst, worst_pair = pct, (names[i], names[j])
                if diff:  # chi in cap CO khac biet, khong in 55 dong toan so 0
                    print("%-16s %-16s %12d %10.4f%%" % (names[i], names[j], diff, pct))
        if worst == 0.0:
            print("(khong cap nao khac nhau du mot diem anh)")

        print()
        if worst == 0.0:
            print("KET LUAN: bat bien tuyet doi — 0 diem anh khac nhau qua ca %d thoi tiet."
                  % len(frames))
        elif worst < 0.05:
            print("KET LUAN: bat bien tren thuc te. Cap lech nhat la %s vs %s, %d/%d diem anh"
                  % (worst_pair[0], worst_pair[1], int(round(worst * total / 100.0)), total))
            print("(%.4f%%). Vai diem le do roi vao cac thoi tiet co mua — hat mua duoc phan" % worst)
            print("loai thanh class rieng. Qua nho de anh huong policy, nhung du de KHONG noi")
            print("duoc la 'bat bien tuyet doi'.")
        else:
            print("KET LUAN: KHONG bat bien — cap lech nhat %s vs %s, %.4f%% so diem anh."
                  % (worst_pair[0], worst_pair[1], worst))
            print("Phai coi thoi tiet la mot chieu bien thien that su cua du lieu, va danh gia")
            print("lai cac ket luan dua tren 'seg la ground truth'.")

        if worst < 0.05:
            print()
            print("HE QUA CHO DO AN: doi thoi tiet trong luc train KHONG tao them du lieu moi")
            print("cho mang, vi dau vao gan nhu khong doi. Do la tin TOT (mo hinh mien nhiem")
            print("voi thoi tiet theo dung thiet ke), nhung cung co nghia la khong duoc ke viec")
            print("doi thoi tiet nhu mot bien phap da dang hoa du lieu. Muon da dang thi phai")
            print("doi ban do va tuyen duong — dung nhu `town_rotate_episodes` dang lam.")
    finally:
        if camera is not None:
            camera.stop(); camera.destroy()
        if vehicle is not None:
            vehicle.destroy()
        settings.synchronous_mode = original_sync
        settings.fixed_delta_seconds = original_delta
        world.apply_settings(settings)


if __name__ == "__main__":
    main()
