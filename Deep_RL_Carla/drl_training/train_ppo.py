"""PPO fine-tuning entrypoint — warm-starts the actor from an IL checkpoint and fine-tunes
against a live CARLA server. See `sac/` + `train_sac.py` for the off-policy alternative
(trade-offs explained in `README.md`).

Usage (mirrors `docs/manual_thu_thap_du_lieu.md`'s terminal layout):
    Terminal 1: CarlaUE4.exe -quality-level=Low
    Terminal 2: (optional) load a Town + weather via a small script/notebook
    Terminal 3: python train_ppo.py --config ppo_config.json

This is an ACTIVE client: it spawns and drives its own ego vehicle in synchronous mode.
Do NOT run it at the same time as `data_collection/` (passive collector) or
`automatic_control.py` against the same CARLA world — they will fight over vehicle control
and world settings (synchronous_mode in particular).

I could not run this against a live CARLA server myself while writing it (no CARLA/Python
in this environment) — please treat the first run as a smoke test: start with a small
`--n-steps` / `--total-steps` and watch `episode_log.csv` / console output before a long run.
"""

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import load_config  # noqa: E402
from csv_logger import CsvLogger  # noqa: E402
from envs.carla_lane_keep_env import CarlaLaneKeepEnv  # noqa: E402
from policy.actor_critic import GaussianActor, ValueCritic, load_il_actor_weights  # noqa: E402
from policy.checkpoint_io import load_il_checkpoint  # noqa: E402
from policy.observation import ObservationContract  # noqa: E402
from ppo.ppo_agent import PPOAgent  # noqa: E402
from ppo.rollout_buffer import RolloutBuffer  # noqa: E402


def resolve_device(config):
    if config["device"] == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(path, agent, update, config, contract):
    torch.save(dict(
        agent.state_dict(), algorithm="ppo", update=update, config=config,
        scalar_feature_dim=contract.scalar_feature_dim, num_classes=contract.num_classes,
    ), path)


def main():
    config = load_config("ppo")
    output_dir = Path(config["output"]).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(config)
    print("Device:", device)

    il_checkpoint_path = Path(config["il_checkpoint"]).expanduser().resolve()
    il_checkpoint = load_il_checkpoint(il_checkpoint_path, map_location="cpu")
    contract = ObservationContract(il_checkpoint)
    feature_names = (contract.continuous_cols + contract.raw_action_cols +
                      ["traffic_light_%s" % v for v in contract.traffic_light_vocab])
    print("Observation contract: %d scalar features: %s" % (contract.scalar_feature_dim, feature_names))

    actor = GaussianActor(contract.scalar_feature_dim, contract.num_classes,
                          log_std_init=config.get("log_std_init", (-3.0, -1.5)))
    critic = ValueCritic(contract.scalar_feature_dim, contract.num_classes)
    print("log_std khoi tao: %s -> std %s" % (
        [round(v, 2) for v in actor.log_std.detach().tolist()],
        [round(v, 4) for v in actor.log_std.detach().exp().tolist()]))

    if config["warm_start"]:
        load_il_actor_weights(actor, il_checkpoint)
        print("Da warm-start actor tu:", il_checkpoint_path)
    else:
        print("[!] Bo qua warm-start — actor khoi tao ngau nhien (chi nen dung de doi chung/debug).")

    agent = PPOAgent(actor, critic, config, device)
    print("actor_lr=%.1e | critic_lr=%.1e | critic_warmup_updates=%d | BatchNorm dong bang: %d lop" % (
        agent.actor_lr, agent.critic_lr, agent.critic_warmup_updates, agent.frozen_bn))

    update = config_start_update = 0
    if config["_resume"]:
        state = torch.load(config["_resume"], map_location=device)
        agent.load_state_dict(state)
        update = config_start_update = state.get("update", 0)
        print("Resume tu:", config["_resume"], "| update =", config_start_update)

    env = CarlaLaneKeepEnv(config, contract)

    obs_h = config.get("obs_height", config["height"])
    obs_w = config.get("obs_width", config["width"])
    buffer = RolloutBuffer(config["n_steps"], (obs_h, obs_w), contract.scalar_feature_dim, 2, device)

    episode_log = CsvLogger(output_dir / "episode_log.csv",
                             ["update", "global_step", "episode_reward", "episode_len", "terminate_reason"])
    update_log = CsvLogger(output_dir / "update_log.csv",
                            ["update", "global_step", "policy_loss", "value_loss", "entropy",
                             "approx_kl", "clip_fraction", "steps_per_sec", "mean_episode_reward"])

    obs, _info = env.reset()
    episode_reward, episode_len = 0.0, 0
    global_step = config_start_update * config["n_steps"]
    n_updates = max(1, config["total_steps"] // config["n_steps"])
    recent_episode_rewards = []

    try:
        for update in range(config_start_update, n_updates):
            update_start = time.time()
            buffer.reset()
            for _ in range(config["n_steps"]):
                env_action, raw_action, log_prob, value = agent.act(obs["seg"], obs["scalar"])
                next_obs, reward, terminated, truncated, info = env.step(env_action)
                done = terminated or truncated

                # Time-limit bootstrap: a truncated (not terminated) episode is cut off by
                # `max_episode_steps`, not by the environment actually ending — treat the
                # cut-off transition's reward as if training continued, by adding the
                # critic's estimate of the value beyond the cutoff. Without this, PPO learns
                # to treat "ran out of time" as equivalent to "crashed", which is wrong and
                # measurably hurts training in any time-limited env (well-documented issue,
                # e.g. Pardo et al., "Time Limits in Reinforcement Learning").
                if truncated and not terminated:
                    bootstrap_value = agent.value_of(next_obs["seg"], next_obs["scalar"])
                    reward = reward + config["gamma"] * bootstrap_value

                buffer.add(obs["seg"], obs["scalar"], raw_action, log_prob, reward, value, done)
                episode_reward += reward
                episode_len += 1
                global_step += 1
                obs = next_obs

                if done:
                    episode_log.log({
                        "update": update, "global_step": global_step,
                        "episode_reward": episode_reward, "episode_len": episode_len,
                        "terminate_reason": info.get("terminate_reason", "time_limit"),
                    })
                    recent_episode_rewards.append(episode_reward)
                    recent_episode_rewards = recent_episode_rewards[-20:]
                    episode_reward, episode_len = 0.0, 0
                    obs, _info = env.reset()

            # Safe unconditionally: if the last stored transition was terminal, `dones[-1]`
            # zeroes this out inside `compute_gae` regardless of what `last_value` is.
            last_value = agent.value_of(obs["seg"], obs["scalar"])
            advantages, returns = buffer.compute_gae(last_value, config["gamma"], config["gae_lambda"])
            # Trong `critic_warmup_updates` dau, actor bi dong bang: critic hoc ham gia tri
            # quanh chinh hanh vi IL truoc khi bat ky gradient policy nao duoc phep chay
            # (§12 notebook IL). Bo qua buoc nay thi advantage tu critic ngau nhien se xoa
            # trong so warm-start trong vai tram update dau.
            freeze_actor = update < agent.critic_warmup_updates
            stats = agent.update(buffer, advantages, returns, freeze_actor=freeze_actor)

            elapsed = max(time.time() - update_start, 1e-6)
            mean_reward = float(np.mean(recent_episode_rewards)) if recent_episode_rewards else float("nan")
            update_log.log({
                "update": update, "global_step": global_step,
                "policy_loss": stats["policy_loss"], "value_loss": stats["value_loss"],
                "entropy": stats["entropy"], "approx_kl": stats["approx_kl"],
                "clip_fraction": stats["clip_fraction"],
                "steps_per_sec": config["n_steps"] / elapsed, "mean_episode_reward": mean_reward,
            })
            print("update=%d step=%d%s policy_loss=%.4f value_loss=%.4f kl=%.4f mean_ep_reward=%.2f (%.1f steps/s)" % (
                update, global_step, " [critic-warmup]" if freeze_actor else "",
                stats["policy_loss"], stats["value_loss"], stats["approx_kl"],
                mean_reward, config["n_steps"] / elapsed))

            is_last_update = update == n_updates - 1
            if (update + 1) % config["save_every_updates"] == 0 or is_last_update:
                ckpt_path = output_dir / ("ppo_update_%06d.pt" % (update + 1))
                save_checkpoint(ckpt_path, agent, update + 1, config, contract)
                save_checkpoint(output_dir / "ppo_latest.pt", agent, update + 1, config, contract)
                print("Da luu checkpoint:", ckpt_path)
    except KeyboardInterrupt:
        print("\nDung boi Ctrl+C — luu checkpoint truoc khi thoat...")
        save_checkpoint(output_dir / "ppo_interrupted.pt", agent, update, config, contract)
    finally:
        env.close()
        episode_log.close()
        update_log.close()


if __name__ == "__main__":
    main()
