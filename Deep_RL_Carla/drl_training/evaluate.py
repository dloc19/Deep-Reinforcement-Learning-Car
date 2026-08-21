"""Run a trained PPO or SAC checkpoint against a live CARLA server for N episodes and report
lane-keeping metrics — the DRL-side counterpart to the IL notebook's section 11 evaluation.

Shared across both algorithms because evaluation only needs
`agent.select_action(seg, scalar, deterministic) -> action`, a call shape both
`ppo.ppo_agent.PPOAgent` and `sac.sac_agent.SACAgent` expose identically (see their
docstrings) — no algorithm-specific evaluation logic needed.

Usage:
    python evaluate.py --algorithm ppo --config ppo_config.json \\
        --resume runs/ppo_lane_keep/ppo_latest.pt --episodes 10 --deterministic
    python evaluate.py --algorithm sac --config sac_config.json \\
        --resume runs/sac_lane_keep/sac_latest.pt --episodes 10 --deterministic
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config, peek_algorithm  # noqa: E402
from csv_logger import CsvLogger  # noqa: E402
from envs.carla_lane_keep_env import CarlaLaneKeepEnv  # noqa: E402
from policy.checkpoint_io import load_il_checkpoint  # noqa: E402
from policy.observation import ObservationContract  # noqa: E402


def build_agent(algorithm, contract, config, device):
    if algorithm == "ppo":
        from policy.actor_critic import GaussianActor, ValueCritic
        from ppo.ppo_agent import PPOAgent
        actor = GaussianActor(contract.scalar_feature_dim, contract.num_classes)
        critic = ValueCritic(contract.scalar_feature_dim, contract.num_classes)
        return PPOAgent(actor, critic, config, device)
    if algorithm == "sac":
        from sac.networks import GaussianPolicy, TwinQNetwork
        from sac.sac_agent import SACAgent
        actor = GaussianPolicy(contract.scalar_feature_dim, contract.num_classes)
        critic = TwinQNetwork(contract.scalar_feature_dim, action_dim=2, num_classes=contract.num_classes)
        return SACAgent(actor, critic, config, device)
    raise ValueError("algorithm phai la 'ppo' hoac 'sac', nhan duoc '%s'" % algorithm)


def main():
    algorithm = peek_algorithm()
    config = load_config(algorithm)
    if not config["_resume"]:
        raise SystemExit("Can --resume <checkpoint .pt> de danh gia (vd ppo_latest.pt / sac_latest.pt).")

    device = torch.device("cuda" if config["device"] == "cuda" and torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(config["_resume"], map_location=device)

    checkpoint_algo = checkpoint.get("algorithm")
    if checkpoint_algo and checkpoint_algo != algorithm:
        print("[!] --algorithm=%s nhung checkpoint duoc luu boi thuat toan '%s' — dung '%s'." % (
            algorithm, checkpoint_algo, checkpoint_algo))
        algorithm = checkpoint_algo
        config = load_config(algorithm)  # nap lai config voi dung ALGO_DEFAULTS (an toan: parse lai cung argv)

    il_checkpoint_path = Path(config["il_checkpoint"]).expanduser().resolve()
    il_checkpoint = load_il_checkpoint(il_checkpoint_path, map_location="cpu")
    contract = ObservationContract(il_checkpoint)

    agent = build_agent(algorithm, contract, config, device)
    agent.load_state_dict(checkpoint)
    agent.actor.eval()
    agent.critic.eval()

    episodes = config["_episodes"] or 5
    deterministic = bool(config["_deterministic"])
    print("Danh gia %s: %d episode, deterministic=%s" % (algorithm.upper(), episodes, deterministic))

    env = CarlaLaneKeepEnv(config, contract)
    results = []
    try:
        for episode in range(episodes):
            obs, _info = env.reset()
            done = False
            ep_reward, ep_len, collided, off_lane_steps = 0.0, 0, False, 0
            lane_offsets = []
            info = {}
            while not done:
                action = agent.select_action(obs["seg"], obs["scalar"], deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                ep_reward += reward
                ep_len += 1
                state = info.get("state", {})
                lane_offsets.append(abs(state.get("lane_offset_m", 0.0)))
                if state.get("off_lane"):
                    off_lane_steps += 1
                if info.get("terminate_reason") == "collision":
                    collided = True

            reason = info.get("terminate_reason", "time_limit")
            mean_offset = float(np.mean(lane_offsets)) if lane_offsets else 0.0
            results.append({
                "algorithm": algorithm, "episode": episode, "reward": ep_reward, "length": ep_len,
                "collided": collided, "off_lane_steps": off_lane_steps,
                "mean_abs_lane_offset": mean_offset, "terminate_reason": reason,
            })
            print("episode=%d reward=%.2f len=%d collided=%s mean|lane_offset|=%.3fm reason=%s" % (
                episode, ep_reward, ep_len, collided, mean_offset, reason))
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
                      "off_lane_steps", "mean_abs_lane_offset", "terminate_reason",
                      "checkpoint", "eval_run_utc"]
        logger = CsvLogger(csv_path, fieldnames, mode="w")
        for row in results:
            logger.log(dict(row, checkpoint=checkpoint_path, eval_run_utc=eval_run_utc))
        logger.close()
        print("Da ghi ket qua tung episode ra (ghi de file cu neu co):", csv_path)

    rewards = [r["reward"] for r in results]
    collision_rate = 100.0 * sum(1 for r in results if r["collided"]) / len(results)
    mean_offset = float(np.mean([r["mean_abs_lane_offset"] for r in results]))
    print("\n=== Tong ket %d episode (%s) ===" % (len(results), algorithm.upper()))
    print("Reward trung binh: %.2f +/- %.2f" % (float(np.mean(rewards)), float(np.std(rewards))))
    print("Ty le va cham: %.1f%%" % collision_rate)
    print("Lech lan trung binh (|m|): %.3f" % mean_offset)


if __name__ == "__main__":
    main()
