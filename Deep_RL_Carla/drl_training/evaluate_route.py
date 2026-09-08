# -*- coding: utf-8 -*-
"""Danh gia CO TUYEN DUONG: chay cung mot tuyen A* bang ba bo dieu khien khac nhau va do
xem viec ghep A* vao policy co that su bot va cham o nga tu khong.

VI SAO CAN SCRIPT NAY, khong dung `evaluate.py`:

`evaluate.py` tha xe chay tu do, khong dich den — no do "policy bam lan tot the nao". Nhung
diem yeu da truy duoc cua PPO nam o cho khac: trong nga tu PHAN NHANH, observation khong co
truong nao mang y dinh di lai (`raw_action_cols` rong, chi co speed + speed_limit + den
giao thong), nen policy khong the biet nen re trai hay phai — no chi thay mot vung duong mo
ra ba huong. Do la gioi han cua HOP DONG QUAN SAT chu khong phai cua thuat toan hoc, va
khong the do duoc bang mot bai danh gia khong co dich den.

THIET KE GHEP CAP (paired), khong phai so voi con so cu:

Ba che do chay tren DUNG cung mot danh sach tuyen (cung diem xuat phat, cung dich, cung thu
tu):

    policy   policy lai toan bo, ke ca trong nga tu. Tracker van chay de DO tien do nhung
             khong bao gio cam lai. Day la PPO nguyen ban.
    astar    pure-pursuit lai toan bo. Khong dung policy. Cho biet mot bo dieu khien hinh
             hoc thuan tuy di duoc bao xa.
    hybrid   policy lai duong thuong, pure-pursuit lai nga tu re va doi lan — dung logic
             chuyen giao cua `CarlaDashBoard/.../modes/route_learned_autopilot.py`.

So ghep cap manh hon nhieu so voi doi chieu voi lo danh gia truoc do (Town04 33,3% va cham):
o do moi episode mot tuyen ngau nhien khac nhau, nen chenh lech do duoc lan voi chenh lech
tuyen de/kho. O day ba che do gap DUNG nhung tinh huong nhu nhau, nen chenh lech con lai la
cua bo dieu khien.

NHIP DIEU KHIEN — cho nay de sai:

Dashboard chay pure-pursuit MOI tick (0.05s) con policy moi 0.2s, vi pure-pursuit can nhip
day de om cua muot. Neu de `action_repeat=4` nhu luc train thi pure-pursuit bi ep xuong
0.2s, tuc lam yeu dung cai dang muon do. Nen script chay env o `action_repeat=1` va TU GIU
lenh cua policy 4 buoc de van dung `control_dt=0.2s` ma checkpoint duoc train. An toan vi
`raw_action_cols` rong — `previous_steer` khong nam trong observation cua policy, no chi
vao cong thuc reward, ma danh gia thi khong dung reward.

Doi lai, `off_lane_patience_steps` va `max_episode_steps` dem theo BUOC nen phai nhan 4 de
giu nguyen nguong tinh theo giay.

Chay:
    python evaluate_route.py --algorithm ppo --config ppo_config_v4.json \\
        --resume runs/best/ppo_latest.pt --town Town04 --trials 30
"""

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve()
sys.path.insert(0, str(_THIS.parent))
sys.path.insert(0, str(_THIS.parents[1]))

import carla  # noqa: E402

from config import load_config, peek_algorithm  # noqa: E402
from csv_logger import CsvLogger  # noqa: E402
from envs.carla_lane_keep_env import CarlaLaneKeepEnv  # noqa: E402
from evaluate import build_agent, resolve_target_speed  # noqa: E402
from policy.checkpoint_io import load_il_checkpoint  # noqa: E402
from policy.observation import ObservationContract  # noqa: E402
from router_plan.Global_Route_Planner import GlobalRoutePlanner, RouteNotFoundError  # noqa: E402
from router_plan.controller import RoutePurePursuitController  # noqa: E402

MODES = ("policy", "astar", "hybrid")

# Giong `_PLANNER_COMMANDS` cua mode tren dashboard: chi nhung lenh MANG Y DINH moi phai
# giao cho pure-pursuit. "STRAIGHT" khong nam trong day — di thang qua nga tu chinh la thu
# policy lam tot, giao cho pure-pursuit chi lam mat do muot.
PLANNER_COMMANDS = ("LEFT", "RIGHT", "CHANGELANELEFT", "CHANGELANERIGHT")

# Giu pure-pursuit them bao nhieu buoc sau khi tin hieu nga tu tat. Dashboard dung 20 tick o
# sim_fps 20 = 1 giay; o day mot buoc cung la mot tick 0.05s nen giu nguyen con so.
HANDOFF_HOLD_STEPS = 20


def parse_own_args(argv):
    """Tach cac tham so RIENG cua script nay ra khoi argv truoc khi `load_config()` doc phan
    con lai — `load_config` dung argparse rieng va se bao loi neu gap co la."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--modes", default=",".join(MODES))
    parser.add_argument("--route-min-m", type=float, default=80.0)
    parser.add_argument("--route-max-m", type=float, default=220.0)
    parser.add_argument("--max-seconds", type=float, default=90.0)
    parser.add_argument("--target-speed-kmh", type=float, default=30.0)
    parser.add_argument("--graph-resolution", type=float, default=2.0)
    parser.add_argument("--lane-change-cost", type=float, default=3.0)
    parser.add_argument("--route-tolerance-m", type=float, default=3.0)
    parser.add_argument("--route-seed", type=int, default=20260906)
    parser.add_argument("--out", default=None)
    own, rest = parser.parse_known_args(argv)
    return own, rest


def chon_tuyen(planner, spawn_points, so_tuyen, min_m, max_m, seed):
    """Chon truoc danh sach (chi_so_xuat_phat, chi_so_dich, tuyen) DUNG MOT LAN.

    Phai chon truoc chu khong boc ngau nhien trong luc chay: ba che do bat buoc phai gap
    cung bo tinh huong, neu khong thi khong con la so sanh ghep cap nua. Lo cac cap khong
    co duong di, hoac co duong nhung qua ngan/qua dai (tuyen 20m khong di qua nga tu nao
    thi khong do duoc dieu gi; tuyen 500m lam moi trial ton vai phut).
    """
    rng = np.random.RandomState(seed)
    n = len(spawn_points)
    ket_qua = []
    da_thu = set()
    for _ in range(so_tuyen * 200):
        if len(ket_qua) >= so_tuyen:
            break
        i, j = int(rng.randint(n)), int(rng.randint(n))
        if i == j or (i, j) in da_thu:
            continue
        da_thu.add((i, j))
        try:
            route = planner.plan(spawn_points[i].location, spawn_points[j].location)
        except RouteNotFoundError:
            continue
        if len(route) < 2:
            continue
        tracker = planner.tracker_for(route)
        if not (min_m <= tracker.total_m <= max_m):
            continue
        ket_qua.append((i, j, route, tracker.total_m))
    return ket_qua


def planner_nen_lai(state, route_state, hold):
    """Tra ve (co_giao_cho_pure_pursuit, hold_moi). Giong het `_planner_should_drive` cua
    mode tren dashboard, chep lai chu khong import vi bridge_server nam ngoai repo nay."""
    command = (route_state or {}).get("route_command", "LANEFOLLOW")
    if command in PLANNER_COMMANDS or (state or {}).get("is_junction"):
        return True, HANDOFF_HOLD_STEPS
    if hold > 0:
        return True, hold - 1
    return False, 0


def chay_mot_trial(env, agent, planner, controller_factory, mode, start_idx, goal_loc,
                   toler_m, max_steps, action_hold):
    """Chay MOT tuyen bang MOT che do. Tra ve dict cac chi so cua lan chay do."""
    env.cfg["fixed_spawn_index"] = start_idx
    obs, _info = env.reset()

    vehicle = env.vehicle
    try:
        # Truyen ca huong xe — xem GlobalRoutePlanner.snap_to_graph().
        start_transform = vehicle.get_transform()
        route = planner.plan(start_transform.location, goal_loc,
                             start_heading_deg=start_transform.rotation.yaw)
    except RouteNotFoundError:
        return None
    tracker = planner.tracker_for(route, target_tolerance_m=toler_m)
    controller = controller_factory()

    hold = 0
    giu_lenh, con_giu = None, 0
    lech_lan, so_buoc_planner = [], 0
    va_cham, den_dich = False, False
    state, route_state = {}, {}
    tien_do_m = 0.0

    for _buoc in range(max_steps):
        # Tracker chay TRUOC va MOI buoc bat ke ai lai: `target_index` cua no vua la dau vao
        # cua pure-pursuit vua la can cu chuyen giao. Cap nhat le nhip la chuyen giao sai cho.
        route_state = tracker.update(vehicle.get_transform())
        tien_do_m = max(tien_do_m, float(route_state.get("route_progress_m", 0.0) or 0.0))
        if route_state.get("route_completed"):
            den_dich = True
            break

        if mode == "astar":
            dung_planner = True
        elif mode == "policy":
            dung_planner = False
        else:
            dung_planner, hold = planner_nen_lai(state, route_state, hold)

        if dung_planner:
            so_buoc_planner += 1
            # Tra lai quyen lai thi policy phai quyet dinh NGAY, khong phat lai lenh cu tu
            # truoc khi vao nga tu.
            giu_lenh, con_giu = None, 0
            ctrl = controller.compute_control(
                planner.graph, route, tracker.target_index, vehicle,
                speed_limit_kmh=vehicle.get_speed_limit())
            action = np.array([ctrl.steer, ctrl.throttle - ctrl.brake], dtype=np.float32)
        else:
            if giu_lenh is None or con_giu >= action_hold:
                giu_lenh = agent.select_action(obs["seg"], obs["scalar"], deterministic=True)
                giu_lenh = np.asarray(giu_lenh, dtype=np.float32).reshape(-1)
                con_giu = 1
            else:
                con_giu += 1
            action = giu_lenh

        obs, _r, terminated, truncated, info = env.step(action)
        state = info.get("state", {}) or {}
        if not state.get("is_junction"):
            lech_lan.append(abs(float(state.get("lane_offset_m", 0.0))))
        if info.get("terminate_reason") == "collision":
            va_cham = True
        if terminated or truncated:
            break

    tong_m = tracker.total_m or 1.0
    return {
        "che_do": mode,
        "diem_xuat_phat": start_idx,
        "dai_tuyen_m": round(tracker.total_m, 2),
        "den_dich": den_dich,
        "va_cham": va_cham,
        "tien_do_pct": round(100.0 * min(tien_do_m, tong_m) / tong_m, 2),
        "so_buoc": _buoc + 1,
        "lech_lan_m": round(float(np.mean(lech_lan)), 4) if lech_lan else float("nan"),
        "ty_le_buoc_planner": round(100.0 * so_buoc_planner / max(_buoc + 1, 1), 1),
        "ket_thuc": info.get("terminate_reason", "het_gio") if not den_dich else "den_dich",
    }


def tom_tat(rows, mode):
    r = [x for x in rows if x["che_do"] == mode]
    if not r:
        return None
    off = [x["lech_lan_m"] for x in r if not np.isnan(x["lech_lan_m"])]
    return {
        "n": len(r),
        "va_cham_pct": 100.0 * sum(1 for x in r if x["va_cham"]) / len(r),
        "den_dich_pct": 100.0 * sum(1 for x in r if x["den_dich"]) / len(r),
        "tien_do_pct": float(np.mean([x["tien_do_pct"] for x in r])),
        "lech_lan_m": float(np.mean(off)) if off else float("nan"),
        "buoc_planner_pct": float(np.mean([x["ty_le_buoc_planner"] for x in r])),
    }


def mcnemar(rows, mode_a, mode_b, khoa):
    """Kiem dinh McNemar cho hai che do tren CUNG bo tuyen (du lieu ghep cap).

    Dung McNemar chu khong dung z-test hai ty le: hai nhanh khong doc lap, chung chay tren
    dung cung nhung tuyen duong. Chi cac cap BAT DONG (mot ben hong, ben kia khong) mang
    thong tin; cac cap cung ket qua chi phan anh do kho cua tuyen, khong phan biet duoc hai
    bo dieu khien.
    """
    a = {x["diem_xuat_phat"]: x for x in rows if x["che_do"] == mode_a}
    b = {x["diem_xuat_phat"]: x for x in rows if x["che_do"] == mode_b}
    chung = sorted(set(a) & set(b))
    n01 = sum(1 for k in chung if a[k][khoa] and not b[k][khoa])
    n10 = sum(1 for k in chung if not a[k][khoa] and b[k][khoa])
    n = n01 + n10
    if n == 0:
        return n01, n10, float("nan"), 1.0
    # Xac suat hai phia cua nhi thuc B(n, 0.5) — dung chinh xac thay vi xap xi chi binh
    # phuong, vi n o day thuong nho (duoi 25).
    #
    # Tu tinh to hop chu khong dung `math.comb`: moi truong chay la Python 3.7 (conda env
    # `carla_rl`, bam theo CARLA 0.9.10) con `math.comb` chi co tu 3.8.
    from math import factorial

    def to_hop(a, b):
        return factorial(a) // (factorial(b) * factorial(a - b))

    k = min(n01, n10)
    p = min(1.0, 2.0 * sum(to_hop(n, i) for i in range(k + 1)) / (2.0 ** n))
    return n01, n10, (n01 - n10) / np.sqrt(n), p


def main():
    own, rest = parse_own_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest

    algorithm = peek_algorithm()
    config = load_config(algorithm)
    if not config["_resume"]:
        raise SystemExit("Can --resume <checkpoint .pt>")

    modes = [m.strip() for m in own.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            raise SystemExit("--modes chi nhan: %s" % ", ".join(MODES))

    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available()
                          else "cpu")
    checkpoint = torch.load(config["_resume"], map_location=device)
    if checkpoint.get("algorithm") and checkpoint["algorithm"] != algorithm:
        algorithm = checkpoint["algorithm"]
        config = load_config(algorithm)

    contract = ObservationContract(
        load_il_checkpoint(Path(config["il_checkpoint"]).expanduser().resolve(),
                           map_location="cpu"))

    if isinstance(config.get("town"), (list, tuple)):
        config["town"] = config["town"][0]
        print("[!] config liet ke nhieu ban do — danh gia can MOT, dung '%s'." % config["town"])
    config["town_rotate_episodes"] = 10 ** 9
    config["target_speed_mps"] = resolve_target_speed(config, contract)

    # Xem muc "NHIP DIEU KHIEN" o docstring: ha action_repeat ve 1 de pure-pursuit chay day
    # nhip, va nhan bu cac nguong dem theo buoc de chung giu nguyen y nghia tinh theo giay.
    ar_goc = int(config.get("action_repeat", 4))
    config["action_repeat"] = 1
    config["off_lane_patience_steps"] = int(config.get("off_lane_patience_steps", 10)) * ar_goc
    dt_buoc = float(config.get("fixed_delta_seconds", 0.05))
    max_steps = int(own.max_seconds / max(dt_buoc, 1e-6))
    config["max_episode_steps"] = max_steps + 10   # de vong lap cua ta quyet dinh, khong phai env

    agent = build_agent(algorithm, contract, config, device)
    if algorithm == "sac":
        agent.load_state_dict(checkpoint, actor_only=True)
    else:
        agent.load_state_dict(checkpoint)
    agent.actor.eval()
    agent.critic.eval()

    print("Danh gia CO TUYEN: %s | %s | %d tuyen | che do: %s" % (
        algorithm.upper(), config["town"], own.trials, ", ".join(modes)))
    print("action_repeat %d -> 1 (pure-pursuit day nhip); policy van giu lenh %d buoc = %.2fs"
          % (ar_goc, ar_goc, ar_goc * dt_buoc))

    env = CarlaLaneKeepEnv(config, contract)
    rows = []
    try:
        planner = GlobalRoutePlanner(env.map, resolution_m=own.graph_resolution,
                                     lane_change_cost=own.lane_change_cost)
        print("Dang chon %d tuyen dai %.0f-%.0f m tren %s..." % (
            own.trials, own.route_min_m, own.route_max_m, config["town"]))
        tuyen = chon_tuyen(planner, env.spawn_points, own.trials,
                           own.route_min_m, own.route_max_m, own.route_seed)
        if len(tuyen) < own.trials:
            print("[!] Chi tim duoc %d/%d tuyen thoa dieu kien — chay voi so nay."
                  % (len(tuyen), own.trials))
        print("Da chon %d tuyen, dai trung binh %.0f m\n" % (
            len(tuyen), float(np.mean([t[3] for t in tuyen])) if tuyen else 0.0))

        def controller_factory():
            return RoutePurePursuitController(
                target_speed_kmh=own.target_speed_kmh, dt=dt_buoc)

        for so, (i, j, _route, dai_m) in enumerate(tuyen):
            for mode in modes:
                kq = chay_mot_trial(
                    env, agent, planner, controller_factory, mode, i,
                    env.spawn_points[j].location, own.route_tolerance_m, max_steps, ar_goc)
                if kq is None:
                    continue
                rows.append(kq)
                print("  tuyen %2d/%d (%3.0fm) %-7s: %-9s tien do %5.1f%% lech %.3f "
                      "planner %4.1f%% %s" % (
                          so + 1, len(tuyen), dai_m, mode, kq["ket_thuc"], kq["tien_do_pct"],
                          kq["lech_lan_m"], kq["ty_le_buoc_planner"],
                          "VA CHAM" if kq["va_cham"] else ""))
    finally:
        env.close()

    if not rows:
        print("Khong co ket qua nao.")
        return

    print("\n=== Tong ket %d tuyen tren %s ===" % (
        len(set(r["diem_xuat_phat"] for r in rows)), config["town"]))
    print("%-8s %4s %10s %10s %11s %11s %11s" % (
        "che do", "n", "va cham", "den dich", "tien do", "lech lan", "buoc planner"))
    print("-" * 70)
    for mode in modes:
        s = tom_tat(rows, mode)
        if s:
            print("%-8s %4d %9.1f%% %9.1f%% %10.1f%% %11.3f %10.1f%%" % (
                mode, s["n"], s["va_cham_pct"], s["den_dich_pct"], s["tien_do_pct"],
                s["lech_lan_m"], s["buoc_planner_pct"]))

    if "policy" in modes and "hybrid" in modes:
        print("\n=== McNemar ghep cap: policy doi chung hybrid ===")
        for khoa, ten in (("va_cham", "va cham"), ("den_dich", "den dich")):
            n01, n10, z, p = mcnemar(rows, "policy", "hybrid", khoa)
            print("  %-9s chi policy %2d | chi hybrid %2d | p = %.4f  %s" % (
                ten, n01, n10, p,
                "khac biet co y nghia" if p < 0.05 else "chua du de ket luan"))

    if own.out:
        path = Path(own.out).expanduser().resolve()
        fields = ["che_do", "diem_xuat_phat", "dai_tuyen_m", "den_dich", "va_cham",
                  "tien_do_pct", "so_buoc", "lech_lan_m", "ty_le_buoc_planner", "ket_thuc",
                  "ban_do", "checkpoint", "chay_luc_utc"]
        logger = CsvLogger(path, fields, mode="w")
        luc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        for r in rows:
            logger.log(dict(r, ban_do=config["town"],
                            checkpoint=str(Path(config["_resume"]).resolve()),
                            chay_luc_utc=luc))
        logger.close()
        print("\nDa ghi:", path)


if __name__ == "__main__":
    main()
