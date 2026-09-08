"""Chay mot checkpoint SAC da train tren CARLA server that N episode va bao cao cac chi so
bam lan - doi ung ben DRL cua muc 11 (danh gia) trong notebook IL.

Danh gia chi goi `agent.select_action(seg, scalar, deterministic) -> action`, nen no khong
dung toi critic hay replay buffer.

Chay (MOT ban do moi lo danh gia - xem chu thich ve `--town` ben duoi):
    python evaluate.py --config sac_config.json --town Town05 \
        --resume runs/sac_f/sac_latest.pt --episodes 30 --deterministic \
        --eval-csv-out runs/sac_f/eval_Town05.csv
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
from policy.checkpoint_io import load_il_checkpoint  # noqa: E402
from policy.observation import ObservationContract  # noqa: E402


def resolve_target_speed(config, contract):
    """`target_speed_mps=None` -> trung binh `speed_mps` cua tap train IL.

    Doc tu chinh checkpoint thay vi go mot hang so, de neu train lai IL tren du lieu co
    toc do khac thi reward tu bam theo, khong lech am tham."""
    if config.get("target_speed_mps"):
        return float(config["target_speed_mps"])
    stats = getattr(contract, "norm_stats", {}) or {}
    mean = stats.get("speed_mps", (8.0, 1.0))[0]
    return float(max(mean, 1.0))

def build_agent(contract, config, device):
    from sac.networks import GaussianPolicy, TwinQNetwork
    from sac.sac_agent import SACAgent
    actor = GaussianPolicy(contract.scalar_feature_dim, contract.num_classes)
    critic = TwinQNetwork(contract.scalar_feature_dim, action_dim=2,
                          num_classes=contract.num_classes)
    return SACAgent(actor, critic, config, device)


def main():
    config = load_config()
    if not config["_resume"]:
        raise SystemExit("Can --resume <checkpoint .pt> de danh gia "
                         "(vd runs/sac_f/sac_latest.pt).")

    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config["_resume"], map_location=device)

    # Chan nap nham checkpoint PPO (`../drl_training/runs/ppo_v*`): kien truc actor khac
    # han (khong co `log_std_head`, `log_std` la vector doc lap trang thai), nen neu khong
    # kiem tra o day thi phep nap se bo qua phan lech va cho ra mot policy sai am tham.
    checkpoint_algo = checkpoint.get("algorithm")
    if checkpoint_algo and checkpoint_algo != "sac":
        raise SystemExit(
            "Checkpoint duoc luu boi thuat toan '%s', khong phai SAC. Thu muc nay chi danh "
            "gia SAC - dung ../drl_training/evaluate.py cho checkpoint PPO." % checkpoint_algo)

    il_checkpoint_path = Path(config["il_checkpoint"]).expanduser().resolve()
    il_checkpoint = load_il_checkpoint(il_checkpoint_path, map_location="cpu")
    contract = ObservationContract(il_checkpoint)

    # `town` mac dinh gio la DANH SACH bon town (de train khong quen ban do nao). Danh gia
    # thi phai co dinh MOT ban do: neu de nguyen, env se xoay vong sau moi
    # `town_rotate_episodes` episode va tron ket qua cua nhieu ban do vao mot lo danh gia —
    # con so trung binh khi do khong ung voi ban do nao ca.
    if isinstance(config.get("town"), (list, tuple)):
        chosen = config["town"][0]
        print("[!] --town khong duoc dat va config liet ke %d ban do %s. Danh gia can MOT "
              "ban do -> dung '%s'. Dat --town <TenBanDo> de chon ro." % (
                  len(config["town"]), list(config["town"]), chosen))
        config["town"] = chosen
    # Chot chan lan hai: du sao cung tat xoay vong trong luc danh gia.
    config["town_rotate_episodes"] = 10 ** 9

    agent = build_agent(contract, config, device)
    # Danh gia chi goi `select_action` -> chi can actor. Nap "actor_only" de mot checkpoint
    # train truoc khi kien truc critic doi van danh gia duoc: critic khong tham gia phep
    # tinh nao o day, nen tu choi nap no chi lam mat kha nang so sanh cac lan train cu.
    agent.load_state_dict(checkpoint, actor_only=True)
    agent.actor.eval()
    agent.critic.eval()

    # Mac dinh lay tu `eval_episodes` cua config (30), khong phai 5: xem chu thich o
    # config.py — o n=10 ti le va cham khong phan biet duoc hai checkpoint bat ky.
    episodes = config["_episodes"] or config.get("eval_episodes", 30)
    deterministic = bool(config["_deterministic"])
    print("Danh gia SAC: %d episode, deterministic=%s" % (episodes, deterministic))

    config["target_speed_mps"] = resolve_target_speed(config, contract)
    print("target_speed = %.2f m/s (%.0f km/h) — moc \"day du diem toc do\"" % (
        config["target_speed_mps"], config["target_speed_mps"] * 3.6))
    env = CarlaLaneKeepEnv(config, contract)
    results = []
    try:
        for episode in range(episodes):
            obs, _info = env.reset()
            done = False
            ep_reward, ep_len, collided, off_lane_steps = 0.0, 0, False, 0
            junction_steps = 0
            # Hai danh sach, khong phai mot. Trong nga tu, `lane_offset_m` duoc suy ra tu
            # "lan duong gan nhat", ma cac nhanh cat nhau nen tham chieu do nhay sang nhanh
            # vuong goc chi sau vai met — con so vo nghia (do duoc: heading_error nhay 175
            # do trong MOT buoc 0.2s). Gop no vao trung binh lam nang luc bam lan trong te
            # hon THUC TE. `demo_il.py` da tach nhu vay; o day phai tach GIONG HET, neu
            # khong thi bieu do 09 dem baseline IL (duong thuong) so voi DRL (lan nga tu) —
            # hai thang do khac nhau, ket luan se sai.
            lane_offsets, lane_offsets_road = [], []
            info = {}
            while not done:
                action = agent.select_action(obs["seg"], obs["scalar"], deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_len += 1
                state = info.get("state", {})
                lane_offsets.append(abs(state.get("lane_offset_m", 0.0)))
                if state.get("is_junction"):
                    junction_steps += 1
                else:
                    lane_offsets_road.append(abs(state.get("lane_offset_m", 0.0)))
                if state.get("off_lane"):
                    off_lane_steps += 1
                if info.get("terminate_reason") == "collision":
                    collided = True

            reason = info.get("terminate_reason", "time_limit")
            mean_offset = float(np.mean(lane_offsets)) if lane_offsets else 0.0
            mean_offset_road = (float(np.mean(lane_offsets_road)) if lane_offsets_road
                                else float("nan"))
            results.append({
                "algorithm": "sac", "episode": episode, "reward": ep_reward, "length": ep_len,
                "collided": collided, "off_lane_steps": off_lane_steps,
                "junction_steps": junction_steps,
                "mean_abs_lane_offset": mean_offset,
                "mean_abs_lane_offset_road": mean_offset_road,
                "terminate_reason": reason, "town": env.current_town,
            })
            print("episode=%d reward=%.2f len=%d collided=%s mean|lane_offset|=%.3fm "
                  "(duong thuong %.3fm, nga tu %.0f%%) reason=%s" % (
                      episode, ep_reward, ep_len, collided, mean_offset, mean_offset_road,
                      100.0 * junction_steps / max(ep_len, 1), reason))
    finally:
        env.close()

    if not results:
        return

    if config["_eval_csv_out"]:
        csv_path = Path(config["_eval_csv_out"]).expanduser().resolve()
        if csv_path.exists():
            print("[!] %s da ton tai — se GHI DE (khong noi vao) vi day la 1 lo danh gia "
                  "doc lap moi; noi vao se tron lan nay voi lan truoc khi ve bieu do." % csv_path)
        # mode="w": moi lan chay evaluate.py la MOT lo danh gia doc lap (N episode tren MOT
        # checkpoint) — xem giai thich chi tiet trong CsvLogger.__init__. Ghi kem checkpoint
        # + thoi diem chay de con biet ket qua nay ung voi checkpoint/lan chay nao.
        checkpoint_path = str(Path(config["_resume"]).expanduser().resolve())
        eval_run_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        fieldnames = ["algorithm", "episode", "reward", "length", "collided",
                      "off_lane_steps", "junction_steps", "mean_abs_lane_offset",
                      "mean_abs_lane_offset_road", "terminate_reason", "town",
                      "checkpoint", "eval_run_utc"]
        logger = CsvLogger(csv_path, fieldnames, mode="w")
        for row in results:
            logger.log(dict(row, checkpoint=checkpoint_path, eval_run_utc=eval_run_utc))
        logger.close()
        print("Da ghi ket qua tung episode ra (ghi de file cu neu co):", csv_path)

    rewards = [r["reward"] for r in results]
    collision_rate = 100.0 * sum(1 for r in results if r["collided"]) / len(results)
    mean_offset = float(np.mean([r["mean_abs_lane_offset"] for r in results]))
    mean_offset_road = float(np.nanmean([r["mean_abs_lane_offset_road"] for r in results]))
    junction_rate = (100.0 * sum(r["junction_steps"] for r in results)
                     / max(sum(r["length"] for r in results), 1))
    print("\n=== Tong ket %d episode (SAC) ===" % len(results))
    print("Reward trung binh: %.2f +/- %.2f" % (float(np.mean(rewards)), float(np.std(rewards))))
    print("Ty le va cham: %.1f%%" % collision_rate)
    print("Lech lan trung binh (|m|): %.3f  (chi duong thuong: %.3f)" % (
        mean_offset, mean_offset_road))
    print("Thoi gian trong nga tu: %.1f%%  (o do lane_offset la phep do rac)" % junction_rate)


if __name__ == "__main__":
    main()
