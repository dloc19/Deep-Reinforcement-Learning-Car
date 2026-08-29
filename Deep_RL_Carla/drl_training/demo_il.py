"""Chay THANG checkpoint IL (`behavior_cloning/best_il_model.pth`) tren CARLA song —
buoc kiem tra VONG KIN bat buoc TRUOC khi bat dau fine-tune DRL.

Vi sao can file rieng ma `evaluate.py` khong lam duoc: `evaluate.py` doi mot checkpoint
PPO/SAC (`--resume ....pt`), tuc chi dung duoc SAU khi da train DRL. Nhung cau hoi quan
trong nhat lai den TRUOC do: "policy IL co tu lai duoc khong, hay no chi dep tren bang MAE
offline?". MAE offline duoc do tren quy dao CUA AUTOPILOT (open-loop). Trong vong kin, sai
so cua chinh model quyet dinh khung hinh ke tiep — day la cho moi policy IL chet, va la ly
do §12 cua notebook canh bao ve "MAE vung lech lan gap N lan MAE vung giua lan".

Script nay dung DUNG duong di ma DRL se dung: `CarlaLaneKeepEnv` + `GaussianActor` da
warm-start bang `load_il_actor_weights` + `mean_action` (deterministic, khong lay mau
exploration). Nho vay ket qua o day chinh la diem xuat phat that su cua PPO/SAC, chu khong
phai mot ban inference viet lai.

Cach dung:
    python demo_il.py --config ppo_config.json --episodes 3
    python demo_il.py --config ppo_config.json --episodes 3 --preview
    python demo_il.py --config ppo_config.json --episodes 5 --town Town05 \\
        --log-steps runs/il_demo/steps.csv

Luu y ve ban do: Town03 co 44% waypoint nam trong nga tu, noi `lane_offset_m` /
`heading_error_rad` la phep do rac (xem policy/observation.py). Tong ket ben duoi vi vay
in RIENG con so tren duong thuong, va cham diem PASS bang chinh con so do.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config  # noqa: E402
from csv_logger import CsvLogger  # noqa: E402
from envs.carla_lane_keep_env import CarlaLaneKeepEnv  # noqa: E402
from policy.actor_critic import GaussianActor, load_il_actor_weights  # noqa: E402
from policy.checkpoint_io import load_il_checkpoint  # noqa: E402
from policy.observation import ObservationContract  # noqa: E402

# Nguong "dat/khong dat" cho ket luan cuoi script. Khong phai con so tuy y:
#   - lane_offset 0.35m: lan CARLA rong ~3.5m, xe rong ~2m -> con ~0.75m du dia moi ben.
#     Vuot 0.35m la da an nua khoang du dia, tuc bat dau nguy hiem chu chua phai ra khoi lan.
#   - off_lane 5%: |lane_offset| > nua be rong lan, tuc banh da qua vach.
#   - speed 1.0 m/s: duoi muc nay xe coi nhu DUNG. Mot policy dung yen co lane_offset dep
#     tuyet doi nhung vo dung — bay ma bang metric bam lan khong tu bat duoc.
PASS_MEAN_OFFSET_M = 0.35
PASS_OFF_LANE_RATE = 0.05
PASS_MEAN_SPEED_MPS = 1.0
# Duoi nguong nay thi steer gan nhu hang so: model dang bo qua anh segmentation chu khong
# phai dang "lai muot". Doi chieu: steer MAE cua chinh tap val vao khoang 0.005-0.027.
PASS_STEER_STD = 0.01

# Mau BGR theo TRAIN id (giong PALETTE cua notebook IL / SEG_CLASS_COLORS cua schema.py).
PREVIEW_COLORS = np.array([[60, 60, 60], [128, 64, 128], [50, 234, 157], [232, 35, 244]],
                          dtype=np.uint8)


def split_argv(argv):
    """Tach co rieng cua script nay ra khoi argv truoc khi dua phan con lai cho
    `config.load_config()` — parser cua no dung `parse_args()` nen bao loi voi co la."""
    own = {"preview": False, "log_steps": None}
    with_value = {"--log-steps": "log_steps"}
    rest = []
    index = 0
    while index < len(argv):
        arg = argv[index]
        if arg == "--preview":
            own["preview"] = True
        elif arg in with_value:
            own[with_value[arg]] = argv[index + 1]
            index += 1
        else:
            # --config / --il-checkpoint / --episodes / --town / ... -> load_config
            rest.append(arg)
        index += 1
    return own, rest


def draw_preview(seg, steer, longitudinal, state):
    import cv2
    frame = PREVIEW_COLORS[np.clip(seg, 0, len(PREVIEW_COLORS) - 1)]
    frame = cv2.resize(frame, (frame.shape[1] * 2, frame.shape[0] * 2),
                       interpolation=cv2.INTER_NEAREST)
    height, width = frame.shape[:2]
    mid = width // 2
    # Thanh steer: goc o giua, keo sang trai/phai. Thanh longitudinal: xanh = ga, do = phanh.
    cv2.rectangle(frame, (mid, height - 30), (int(mid + steer * mid * 0.9), height - 18),
                  (255, 255, 0), -1)
    long_color = (0, 255, 0) if longitudinal >= 0 else (0, 0, 255)
    cv2.rectangle(frame, (mid, height - 16), (int(mid + longitudinal * mid * 0.9), height - 4),
                  long_color, -1)
    cv2.line(frame, (mid, height - 32), (mid, height - 2), (255, 255, 255), 1)
    cv2.putText(frame, "steer %+.3f  long %+.3f" % (steer, longitudinal), (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "v %.1f m/s  offset %+.2f m  head %+.3f rad%s" % (
        state.get("speed_mps", 0.0), state.get("lane_offset_m", 0.0),
        state.get("heading_error_rad", 0.0), "  OFF-LANE" if state.get("off_lane") else ""),
        (6, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imshow("IL demo - quan sat segmentation cua model", frame)
    return cv2.waitKey(1) & 0xFF


def mean_of(rows, key):
    values = [row[key] for row in rows]
    return float(np.mean(values)) if values else float("nan")


def print_checkpoint_header(il_path, il_checkpoint, contract, config):
    control_dt_env = config.get("action_repeat", 1) / float(config.get("fps", 20.0))
    print("Checkpoint IL      :", il_path)
    print("  run_name         : %s | epoch %s | weights %s" % (
        il_checkpoint.get("run_name"), il_checkpoint.get("epoch"),
        il_checkpoint.get("weights_source")))
    print("  val steer/long MAE: %.5f / %.5f   (baseline 'chep prev': %.5f / %.5f)" % (
        il_checkpoint.get("val_steer_mae", float("nan")),
        il_checkpoint.get("val_long_mae", float("nan")),
        il_checkpoint.get("copycat_steer_mae", float("nan")),
        il_checkpoint.get("copycat_long_mae", float("nan"))))
    print("  scalar (%d)       : %s" % (contract.scalar_feature_dim,
                                        il_checkpoint.get("scalar_feature_order")))
    print("  control_dt       : %s s  |  env: fps %.1f x action_repeat %d = %.3f s" % (
        contract.control_dt, config.get("fps", 20.0), config.get("action_repeat", 1),
        control_dt_env))
    print("  towns            : train %s | val %s" % (il_checkpoint.get("train_towns"),
                                                      il_checkpoint.get("val_towns")))
    if not il_checkpoint.get("trained_on_predicted_seg", False):
        print("  [i] IL train tren mask GROUND-TRUTH. Env nay cung doc camera segmentation")
        print("      ground-truth cua CARLA, nen o day KHONG co lech phan phoi. Lech chi")
        print("      xuat hien khi thay bang mask do model segmentation du doan.")


def main():
    own, rest = split_argv(sys.argv[1:])
    config = load_config("ppo", rest)

    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available()
                          else "cpu")
    il_path = Path(config["il_checkpoint"]).expanduser().resolve()
    il_checkpoint = load_il_checkpoint(il_path, map_location="cpu")
    contract = ObservationContract(il_checkpoint)
    print_checkpoint_header(il_path, il_checkpoint, contract, config)

    actor = GaussianActor(contract.scalar_feature_dim, contract.num_classes)
    load_il_actor_weights(actor, il_checkpoint)
    actor.to(device).eval()

    # `--town` do CarlaLaneKeepEnv xu ly (xem `_load_town`), dung chung voi train_ppo.py /
    # train_sac.py / evaluate.py thay vi moi script mot ban.
    env = CarlaLaneKeepEnv(config, contract)
    episodes = config["_episodes"] or 3
    step_logger = None
    if own["log_steps"]:
        step_logger = CsvLogger(
            Path(own["log_steps"]).expanduser().resolve(),
            ["episode", "step", "steer", "longitudinal", "speed_mps", "lane_offset_m",
             "heading_error_rad", "off_lane", "is_junction", "reward"], mode="w")

    results = []
    aborted = False
    try:
        for episode in range(episodes):
            obs, info = env.reset()
            done = False
            record = {"episode": episode, "reward": 0.0, "steps": 0, "collided": False,
                      "off_lane_steps": 0, "junction_steps": 0,
                      "terminate_reason": "time_limit"}
            # `offsets_road` bo qua buoc trong nga tu: o do lane_offset_m la phep do rac nen
            # gop vao trung binh chi lam ban ket luan sai ve nang luc bam lan. Do tren
            # il_demo_v9: |lech| TB 0.579m tren TAT CA buoc, nhung 0.225m neu chi tinh duong
            # thuong — hai con so ke hai cau chuyen khac han.
            offsets, offsets_road, speeds, steers, longs = [], [], [], [], []
            while not done:
                seg_t = torch.as_tensor(obs["seg"], device=device).unsqueeze(0)
                scalar_t = torch.as_tensor(obs["scalar"], device=device).unsqueeze(0)
                with torch.no_grad():
                    # mean_action = ky vong cua policy, KHONG lay mau. IL khong co khai niem
                    # do lech exploration; `log_std` cua GaussianActor la tham so moi tinh
                    # cua PPO, khong den tu checkpoint IL.
                    action = actor.mean_action(seg_t, scalar_t).squeeze(0).cpu().numpy()

                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = info.get("state", {})
                steer = float(np.clip(action[0], -1.0, 1.0))
                longitudinal = float(np.clip(action[1], -1.0, 1.0))

                record["reward"] += reward
                record["steps"] += 1
                offsets.append(abs(state.get("lane_offset_m", 0.0)))
                if state.get("is_junction"):
                    record["junction_steps"] += 1
                else:
                    offsets_road.append(abs(state.get("lane_offset_m", 0.0)))
                speeds.append(state.get("speed_mps", 0.0))
                steers.append(steer)
                longs.append(longitudinal)
                if state.get("off_lane"):
                    record["off_lane_steps"] += 1
                if step_logger is not None:
                    step_logger.log({
                        "episode": episode, "step": record["steps"], "steer": steer,
                        "longitudinal": longitudinal, "speed_mps": state.get("speed_mps", 0.0),
                        "lane_offset_m": state.get("lane_offset_m", 0.0),
                        "heading_error_rad": state.get("heading_error_rad", 0.0),
                        "off_lane": int(bool(state.get("off_lane"))),
                        "is_junction": int(bool(state.get("is_junction"))), "reward": reward})
                if own["preview"] and draw_preview(obs["seg"], steer, longitudinal, state) == 27:
                    aborted = True
                    done = True

            record["terminate_reason"] = info.get("terminate_reason", "time_limit")
            record["collided"] = record["terminate_reason"] == "collision"
            record["mean_abs_offset"] = float(np.mean(offsets)) if offsets else float("nan")
            record["mean_abs_offset_road"] = (float(np.mean(offsets_road)) if offsets_road
                                              else float("nan"))
            record["p95_abs_offset"] = float(np.percentile(offsets, 95)) if offsets else float("nan")
            record["mean_speed"] = float(np.mean(speeds)) if speeds else 0.0
            record["distance_m"] = (record["mean_speed"] * record["steps"]
                                    * config.get("action_repeat", 1) / float(config.get("fps", 20.0)))
            record["steer_std"] = float(np.std(steers)) if steers else 0.0
            record["long_mean"] = float(np.mean(longs)) if longs else 0.0
            record["brake_rate"] = float(np.mean([1.0 if value < 0 else 0.0 for value in longs])) if longs else 0.0
            results.append(record)
            print("ep=%d reward=%8.1f steps=%4d %6.0fm v=%.1fm/s |offset|=%.3f(duong "
                  "thuong %.3f) nga_tu=%3.0f%% off_lane=%4.1f%% steer_std=%.4f "
                  "long=%+.2f(phanh %2.0f%%) -> %s" % (
                      episode, record["reward"], record["steps"], record["distance_m"],
                      record["mean_speed"], record["mean_abs_offset"],
                      record["mean_abs_offset_road"],
                      100.0 * record["junction_steps"] / max(record["steps"], 1),
                      100.0 * record["off_lane_steps"] / max(record["steps"], 1),
                      record["steer_std"], record["long_mean"], 100 * record["brake_rate"],
                      record["terminate_reason"]))
            if aborted:
                break
    finally:
        env.close()
        if step_logger is not None:
            step_logger.close()
        if own["preview"]:
            try:
                import cv2
                cv2.destroyAllWindows()
            except Exception:
                pass

    if not results:
        return

    mean_offset = mean_of(results, "mean_abs_offset")
    # Cham bang so do tren DUONG THUONG. Trong nga tu, lane_offset_m khong phai "bam lan
    # kem" ma la mot phep do khong con y nghia — cham no la cham nham (xem observation.py).
    mean_offset_road = mean_of(results, "mean_abs_offset_road")
    mean_speed = mean_of(results, "mean_speed")
    steer_std = mean_of(results, "steer_std")
    total_steps = sum(row["steps"] for row in results)
    off_lane_rate = sum(row["off_lane_steps"] for row in results) / float(max(total_steps, 1))
    collisions = sum(1 for row in results if row["collided"])

    print("\n=== Tong ket %d episode — policy IL thuan (chua co DRL) ===" % len(results))
    print("Reward TB           : %.1f" % mean_of(results, "reward"))
    print("Quang duong TB      : %.0f m (%.0f buoc/episode)" % (
        mean_of(results, "distance_m"), total_steps / float(len(results))))
    print("Toc do TB           : %.2f m/s (%.1f km/h)" % (mean_speed, mean_speed * 3.6))
    total_junction = sum(row["junction_steps"] for row in results)
    print("|lech lan| TB       : %.3f m  (chi duong thuong: %.3f m)" % (
        mean_offset, mean_offset_road))
    print("Thoi gian trong nga tu: %.1f%%  (o day lane_offset/heading la phep do rac)" % (
        100.0 * total_junction / float(max(total_steps, 1))))
    print("Ty le ra khoi lan   : %.1f%%" % (100.0 * off_lane_rate))
    print("Va cham             : %d/%d episode" % (collisions, len(results)))
    print("Do lech chuan steer : %.4f" % steer_std)
    print("Longitudinal TB     : %+.2f (ty le phanh %.0f%%)" % (
        mean_of(results, "long_mean"), 100 * mean_of(results, "brake_rate")))

    problems = []
    if mean_speed < PASS_MEAN_SPEED_MPS:
        # Nguyen nhan khac han tuy hop dong observation, va hai cach sua cung khac han.
        if "previous_longitudinal" in contract.raw_action_cols:
            cause = ("day la vong lap phan hoi cua `previous_longitudinal`: model phanh o "
                     "buoc dau, gia tri do quay lai lam dau vao, va no khoa cung o do. Bo "
                     "cot nay khoi observation (USE_PREV_ACTIONS = False).")
        else:
            cause = ("`previous_longitudinal` da bi bo, nen nghi can con lai la `speed_mps`: "
                     "neu tap train day frame dung den do (speed~0, long=-1) thi model hoc "
                     "'speed ~ 0 -> phanh', ma xe LUON spawn o toc do 0. Bat "
                     "DROP_STATIONARY_RUNS o §2 cua notebook, va xem §11c.")
        problems.append(
            "XE GAN NHU KHONG CHAY (%.2f m/s). Xem cot longitudinal trong --log-steps: neu "
            "no bi keo ve am roi o lai do thi %s" % (mean_speed, cause))
    if mean_offset_road > PASS_MEAN_OFFSET_M:
        problems.append("Bam lan kem tren duong thuong (|lech| TB %.3f m > %.2f m)."
                        % (mean_offset_road, PASS_MEAN_OFFSET_M))
    if off_lane_rate > PASS_OFF_LANE_RATE:
        problems.append("Ra khoi lan %.1f%% thoi gian (nguong %.0f%%)." % (
            100 * off_lane_rate, 100 * PASS_OFF_LANE_RATE))
    if collisions:
        problems.append("Co %d va cham." % collisions)
    if steer_std < PASS_STEER_STD:
        problems.append("steer gan nhu hang so (std %.4f) — model dang bo qua anh "
                        "segmentation chu khong phai 'lai muot'." % steer_std)

    print("\n" + "=" * 78)
    if problems:
        print("KET LUAN: policy IL CHUA du de demo doc lap. Van de:")
        for item in problems:
            print("  - " + item)
        print("\nVan dung duoc lam warm-start cho DRL, nhung dat ky vong dung: PPO/SAC se phai")
        print("HOC LAI phan nay, khong chi tinh chinh.")
    else:
        print("KET LUAN: policy IL chay on trong vong kin. San sang warm-start DRL.")
    print("=" * 78)

    out_path = Path(config["output"]).expanduser().resolve() / "il_demo_results.csv"
    fields = ["episode", "reward", "steps", "distance_m", "mean_speed", "mean_abs_offset",
              "mean_abs_offset_road", "p95_abs_offset", "off_lane_steps", "junction_steps",
              "steer_std", "long_mean", "brake_rate",
              "collided", "terminate_reason", "checkpoint", "run_utc"]
    logger = CsvLogger(out_path, fields, mode="w")
    run_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    for row in results:
        logger.log(dict([(key, row.get(key)) for key in fields],
                        checkpoint=str(il_path), run_utc=run_utc))
    logger.close()
    print("Da ghi:", out_path)


if __name__ == "__main__":
    main()
