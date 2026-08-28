"""Tim ego va cho mot ego DUNG DUOC xuat hien.

Dung chung boi hai cho:

- `CarlaCollector`: khi ego dang thu bi ket / bi huy / camera chet, collector go
  sensor ra va cho mot chiec xe khac de gan lai NGAY TRONG session dang chay.
- `SessionRunner`: khi ca session da dong lai, no cho ego chay lai roi mo session
  moi.

Dieu kien "dung duoc" kiem tra CA HAI thu. Neu world dung han (truong hop
`no_samples_timeout`) thi `get_velocity()` van tra ve gia tri cu cua tick cuoi
cung, nen chi nhin van toc thoi se tuong xe dang chay va gan sensor vao mot the
gioi da chet. Frame number tang len giua hai lan poll moi la bang chung world
con song.
"""

import time


def speed_of(actor):
    velocity = actor.get_velocity()
    return (velocity.x ** 2 + velocity.y ** 2 + velocity.z ** 2) ** 0.5


def wheel_count(vehicle):
    try:
        return int(vehicle.attributes.get("number_of_wheels", 4))
    except (TypeError, ValueError):
        return 4


def ego_candidates(world, args):
    """Moi xe co the la ego, theo thu tu actor list cua server.

    Loc xe 2 banh: `automatic_control.py` boc blueprint ngau nhien tu
    `vehicle.*`, nen chiec "hero" no tao ra co the la xe dap
    (`vehicle.gazelle.omafiets`) hay moto (`vehicle.kawasaki.ninja`). Mot session
    Town03 that da co 992/4906 mau (20%) quay tu xe dap va moto: dong hoc khac
    han, camera dat o z=2.4 khong con nam tren mui xe, va IL/DRL hoc phai ca hai
    kieu dieu khien tron vao nhau. Dat `--min-wheels 0` de tat bo loc nay.
    """
    vehicles = list(world.get_actors().filter("vehicle.*"))
    if args.vehicle_id:
        # Nguoi dung ghim cung mot id thi khong loc gi them - ho da chon roi.
        return [item for item in vehicles if item.id == args.vehicle_id]
    min_wheels = getattr(args, "min_wheels", 4)
    if min_wheels:
        vehicles = [item for item in vehicles if wheel_count(item) >= min_wheels]
    matches = [item for item in vehicles
               if item.attributes.get("role_name", "") == args.role_name]
    if not matches and len(vehicles) == 1:
        # Chi co dung mot xe tren map: chac chan la no, du role_name khong khop.
        matches = vehicles
    return matches


def find_ego_actor(world, args):
    """Mot lan quet actor list de tim ego. None neu chua co."""
    candidates = ego_candidates(world, args)
    return candidates[0] if candidates else None


def pick_ego(world, args, previous_id=0, threshold=0.1):
    """Chon xe de (gan lai) sensor vao. None neu chua co xe nao dung duoc.

    Uu tien mot chiec xe KHAC voi `previous_id`: khi nguoi dung chay lai
    `automatic_control.py`, chiec hero cu co the con nam do them vai giay truoc
    khi bi huy, va `find_ego_actor` se tra ve dung no - tuc la gan sensor vao lai
    chiec xe vua ket. Xe moi khong can dang chay: no vua spawn nen dung yen vai
    frame dau la binh thuong, watchdog se bat lai neu no khong bao gio chay.

    Con neu chi con lai chinh chiec xe cu (hoac `--vehicle-id` ghim cung mot id)
    thi phai thay no THUC SU di chuyen moi nhan - khong thi lai gan sensor vao
    dung cai xe dang phanh khan cap.
    """
    candidates = ego_candidates(world, args)
    if previous_id:
        fresh = [item for item in candidates if item.id != previous_id]
        if fresh:
            return fresh[-1]  # actor list xep theo thu tu spawn: cuoi = moi nhat
    for vehicle in candidates:
        if speed_of(vehicle) >= threshold:
            return vehicle
    return None


def wait_for_ego(client, args, timeout_s, should_stop, previous_id=0,
                 message=None, poll_s=1.0):
    """Cho den khi world tick tro lai VA co mot ego dung duoc.

    Tra ve `(ego, world)`, hoac `(None, None)` neu het `timeout_s` giay THUC
    hoac `should_stop()` bao dung.
    """
    if timeout_s <= 0:
        return None, None
    deadline = time.time() + timeout_s
    threshold = max(args.stall_speed, 0.1)
    last_frame = None
    announced = False
    while time.time() < deadline:
        if should_stop():
            return None, None
        world = None
        frame = None
        ego = None
        try:
            world = client.get_world()
            frame = world.get_snapshot().frame
            ego = pick_ego(world, args, previous_id, threshold)
        except RuntimeError:
            # World dang reload / ket noi chop chop: coi nhu chua co gi, thu lai.
            pass
        if frame is not None:
            ticking = last_frame is not None and frame != last_frame
            last_frame = frame
            if ticking and ego is not None:
                return ego, world
        if not announced and message:
            print(message % timeout_s)
            announced = True
        time.sleep(poll_s)
    return None, None
