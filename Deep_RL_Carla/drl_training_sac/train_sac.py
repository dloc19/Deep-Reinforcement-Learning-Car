"""SAC fine-tuning entrypoint — warm-starts the actor from an IL checkpoint and fine-tunes
against a live CARLA server using an off-policy replay buffer. See `train_ppo.py` for the
on-policy alternative and `README.md` for when to pick which.

Usage (same terminal layout as `train_ppo.py` — see its docstring for the full walkthrough):
    python train_sac.py --config sac_config.json

Same ACTIVE-client caveat as `train_ppo.py`: do not run alongside `data_collection/` or
`automatic_control.py` against the same CARLA world.

I could not run this against a live CARLA server myself while writing it (no CARLA/Python in
this environment) — treat the first run as a smoke test with a small `--total-steps` and
`learning_starts` (edit `sac_config.json`) before a long run.
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
from policy.checkpoint_io import load_il_checkpoint  # noqa: E402
from policy.observation import ObservationContract  # noqa: E402
from sac.networks import GaussianPolicy, TwinQNetwork, load_il_actor_weights  # noqa: E402
from sac.replay_buffer import ReplayBuffer  # noqa: E402
from sac.sac_agent import SACAgent  # noqa: E402

_EMPTY_STATS = {"critic_loss": float("nan"), "actor_loss": float("nan"), "alpha_loss": float("nan"),
                "alpha": float("nan"), "mean_q": float("nan"), "entropy": float("nan"),
                "explained_variance": float("nan"), "freeze_actor": True}


_EPISODE_FIELDS_EXTRA = ["mean_abs_lane_offset", "mean_abs_lane_offset_road",
                         "junction_steps", "town"]
_EMPTY_EPISODE_STATS = {"mean_abs_lane_offset": float("nan"),
                        "mean_abs_lane_offset_road": float("nan"),
                        "junction_steps": 0, "town": ""}


def resolve_target_speed(config, contract):
    """`target_speed_mps=None` -> trung binh `speed_mps` cua tap train IL.

    Doc tu chinh checkpoint thay vi go mot hang so, de neu train lai IL tren du lieu co
    toc do khac thi reward tu bam theo, khong lech am tham."""
    if config.get("target_speed_mps"):
        return float(config["target_speed_mps"])
    stats = getattr(contract, "norm_stats", {}) or {}
    mean = stats.get("speed_mps", (8.0, 1.0))[0]
    return float(max(mean, 1.0))

def resolve_device(config):
    if config["device"] == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def save_checkpoint(path, agent, step, config, contract):
    torch.save(dict(
        agent.state_dict(), algorithm="sac", step=step, config=config,
        scalar_feature_dim=contract.scalar_feature_dim, num_classes=contract.num_classes,
    ), path)


def main():
    config = load_config("sac")
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

    log_std_init = config.get("log_std_init")
    if config.get("log_std_init_from_checkpoint", True) and contract.action_std:
        # Truyen CA CAP [steer, longitudinal] — `log_std_head` la Linear(32, 2) nen bias cua
        # no dat duoc rieng cho tung chieu. Ban truoc lay trung binh hai chieu thanh mot vo
        # huong; xem chu thich trong sac/networks.py de biet hau qua do duoc.
        log_std_init = list(contract.default_log_std())
        print("log_std_head khoi tao %s -> std %s (tu action_std cua checkpoint IL)" % (
            [round(v, 3) for v in log_std_init],
            [round(2.718281828 ** v, 4) for v in log_std_init]))
    if log_std_init is None:
        log_std_init = -2.5
    actor = GaussianPolicy(contract.scalar_feature_dim, contract.num_classes,
                           log_std_init=log_std_init)
    critic = TwinQNetwork(contract.scalar_feature_dim, action_dim=2, num_classes=contract.num_classes)

    if config["warm_start"]:
        load_il_actor_weights(actor, il_checkpoint)
        print("Da warm-start actor tu:", il_checkpoint_path)
    else:
        print("[!] Bo qua warm-start — actor khoi tao ngau nhien (chi nen dung de doi chung/debug).")

    # Thang chuan hoa BC lay tu chinh checkpoint IL, khong go tay.
    if config.get("bc_action_scale") is None and contract.action_std:
        config["bc_action_scale"] = list(contract.action_std)
    agent = SACAgent(actor, critic, config, device)
    print("actor_lr=%.1e | critic_lr=%.1e | alpha khoi tao=%.3f | critic_warmup_steps=%d | "
          "BatchNorm dong bang: %d lop" % (
              config.get("actor_lr", 3e-4), config.get("critic_lr", 3e-4),
              float(agent.alpha.item()), agent.critic_warmup_steps, agent.frozen_bn))

    global_step = 0
    if config["_resume"]:
        state = torch.load(config["_resume"], map_location=device)
        agent.load_state_dict(state)
        global_step = state.get("step", 0)
        print("Resume tu:", config["_resume"], "| step =", global_step)

        # `reset_log_std`: dat lai `log_std_head` ve dung `action_std` cua checkpoint IL.
        #
        # Can rieng mot co vi `freeze_log_std` chi NGUNG cap nhat — no dong bang o gia tri
        # HIEN TAI, ma neu gia tri do da phinh thi dong bang lai chinh cai hong. Do tren
        # runs/sac_a: std lai max di 0.0787 -> 0.2515 (step 20k) -> 0.6452 (step 30k), tuc
        # gap 8.2 lan muc IL, trong khi policy trung binh van duoc rang buoc BC giu nguyen
        # (troi lai chi 0.0012). Rang buoc BC neo `mean_action`, khong cham toi log_std, nen
        # SAC van tu do bom them nhieu vao cac trang thai bat dinh — dung nhung tinh huong
        # kho, noi xe it chiu duoc nhieu nhat.
        if config.get("reset_log_std"):
            import torch as _t
            with _t.no_grad():
                agent.actor.log_std_head.bias.copy_(
                    _t.tensor([float(v) for v in log_std_init], device=device))
                _t.nn.init.uniform_(agent.actor.log_std_head.weight, -1e-3, 1e-3)
            print("Da dat lai log_std_head ve %s -> std %s" % (
                [round(v, 3) for v in log_std_init],
                [round(2.718281828 ** v, 4) for v in log_std_init]))

    config["target_speed_mps"] = resolve_target_speed(config, contract)
    print("target_speed = %.2f m/s (%.0f km/h) — moc \"day du diem toc do\"" % (
        config["target_speed_mps"], config["target_speed_mps"] * 3.6))
    env = CarlaLaneKeepEnv(config, contract)

    obs_h = config.get("obs_height", config["height"])
    obs_w = config.get("obs_width", config["width"])
    buffer = ReplayBuffer(config["buffer_capacity"], (obs_h, obs_w), contract.scalar_feature_dim, 2, device)
    if global_step > 0:
        # Resume: buffer KHONG nam trong checkpoint (no la 1.4-2.3 GB, xem README). Nen sau
        # moi lan resume, SAC phai nap lai `learning_starts` transition truoc khi hoc tiep —
        # mat khoang learning_starts/14.6 giay. Do la cai gia phai tra cho viec khong luu
        # buffer, va no re hon nhieu so voi mot critic bi pha hong.
        print("Resume: replay buffer bat dau RONG — se nap lai %d transition (~%.1f phut) "
              "truoc khi hoc tiep." % (config["learning_starts"],
                                       config["learning_starts"] / 14.6 / 60.0))
    print("Replay buffer: %d transitions x %dx%d px ~= %.2f GB (seg only, xem sac/replay_buffer.py)" % (
        config["buffer_capacity"], obs_h, obs_w,
        config["buffer_capacity"] * obs_h * obs_w / (1024.0 ** 3)))

    # Xem chu thich cung cho trong train_ppo.py.
    episode_log = CsvLogger(output_dir / "episode_log.csv",
                             ["step", "episode_reward", "episode_len",
                              "terminate_reason"] + _EPISODE_FIELDS_EXTRA)
    update_log = CsvLogger(output_dir / "update_log.csv",
                            ["step", "critic_loss", "actor_loss", "alpha_loss", "alpha",
                             "mean_q", "entropy", "explained_variance", "steps_per_sec",
                             "mean_episode_reward"])

    # Standard SAC always explores with pure-random actions for the first `learning_starts`
    # steps to seed the replay buffer with diverse, unbiased transitions before any critic/
    # actor update happens. That default assumes training from scratch — here the actor is
    # already warm-started from IL, so throwing that away for random exploration would (a)
    # waste the whole point of warm-starting and (b) likely fill the buffer mostly with
    # short, low-quality episodes (random steering crashes fast in CARLA). When warm-started,
    # use the actor's own *stochastic* sample instead — it already drives reasonably and
    # still explores via its Gaussian noise. Fall back to pure-random only when
    # `warm_start=False` (training from scratch), matching textbook SAC.
    use_random_warmup = not config["warm_start"]

    obs, _info = env.reset()
    episode_reward, episode_len = 0.0, 0
    recent_episode_rewards = []
    stats = dict(_EMPTY_STATS)
    last_log_time = time.time()
    last_log_step = global_step

    try:
        while global_step < config["total_steps"]:
            if len(buffer) < config["learning_starts"] and use_random_warmup:
                action = np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)
            else:
                action = agent.select_action(obs["seg"], obs["scalar"], deterministic=False)
                # --- Bom DA DANG HANH DONG cho critic, chi trong giai doan actor dong bang --
                #
                # Do duoc tren runs/sac_a va sac_b: quet lenh lai toan dai -1..+1 chi lam Q
                # doi 0.3-1.7% so voi bien thien giua cac trang thai. Tuc critic hoc duoc
                # Q(s,a) ~ V(s) — no biet dang o tinh huong nao nhung khong phan biet lam gi
                # trong tinh huong do. Actor chi hoc qua dQ/da, nen khong co gi de hoc, va moi
                # tinh chinh bc_coef deu vo nghia (ha 2.5 -> 1.0 ma do troi khong doi).
                #
                # Nguyen nhan la vong luan quan du lieu: buffer chi chua hanh dong tu IL cong
                # nhieu std 0.078, mot dai qua hep de critic hoc duoc su phu thuoc hanh dong.
                # Muon da dang thi phai tham do rong, ma tham do rong lai pha warm-start.
                #
                # Cua so an toan de pha vong nay la giai doan CRITIC-WARMUP: actor bi dong
                # bang nen KHONG THE bi pha du nhieu lon co nao; moi thu bom vao chi chay vao
                # buffer cho critic hoc. Dung epsilon-greedy thay vi tang std deu: giu phan
                # lon quy dao on-distribution (episode du dai de co trang thai da dang) ma van
                # chen hanh dong ngau nhien toan dai de critic thay hau qua cua chung.
                eps = float(config.get("explore_epsilon", 0.0))
                if eps > 0 and agent.updates_done < agent.critic_warmup_steps:
                    if np.random.rand() < eps:
                        action = np.random.uniform(-1.0, 1.0, size=2).astype(np.float32)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # See sac/replay_buffer.py docstring. Transition bi CAT vi het gio van duoc
            # luu (neu khong, transition lien truoc no se lay `next_obs` tu episode moi),
            # nhung duoc danh dau `truncated=True` de khong bao gio bi sample. `done` luu
            # vao buffer la `terminated` — chi va cham/ra khoi lan that su moi cat bootstrap.
            buffer.add(obs["seg"], obs["scalar"], action, reward, terminated,
                       truncated=truncated and not terminated)

            episode_reward += reward
            episode_len += 1
            global_step += 1
            obs = next_obs

            if done:
                stats_ep = info.get("episode_stats", _EMPTY_EPISODE_STATS)
                episode_log.log(dict(stats_ep,
                    step=global_step, episode_reward=episode_reward, episode_len=episode_len,
                    terminate_reason=info.get("terminate_reason", "time_limit"),
                ))
                recent_episode_rewards.append(episode_reward)
                recent_episode_rewards = recent_episode_rewards[-20:]
                episode_reward, episode_len = 0.0, 0
                obs, _info = env.reset()

            # Dieu kien tinh theo SO TRANSITION DANG CO, khong theo global_step. Hai cai
            # nay bang nhau o mot lan chay lien tuc, nhung KHAC HAN khi resume: checkpoint
            # khong mang theo replay buffer, nen `global_step` phuc hoi ve 40 000 trong khi
            # buffer rong tinh. Tinh theo global_step thi SAC bat dau update ngay khi buffer
            # co 128 mau va — voi gradient_steps=1 moi env step — chay hang tram lan gradient
            # tren gan nhu cung mot nhum du lieu, du de pha hong critic vua resume ve.
            if len(buffer) >= config["learning_starts"] and global_step % config["train_freq"] == 0:
                for _ in range(config["gradient_steps"]):
                    batch = buffer.sample(config["batch_size"])
                    stats = agent.update(batch)

            if global_step % 1000 == 0:
                elapsed = max(time.time() - last_log_time, 1e-6)
                steps_per_sec = (global_step - last_log_step) / elapsed
                mean_reward = float(np.mean(recent_episode_rewards)) if recent_episode_rewards else float("nan")
                update_log.log({
                    "step": global_step, "critic_loss": stats["critic_loss"],
                    "actor_loss": stats["actor_loss"], "alpha_loss": stats["alpha_loss"],
                    "alpha": stats["alpha"], "mean_q": stats["mean_q"], "entropy": stats["entropy"],
                    "explained_variance": stats.get("explained_variance", float("nan")),
                    "steps_per_sec": steps_per_sec, "mean_episode_reward": mean_reward,
                })
                print("step=%d%s critic_loss=%.4f actor_loss=%.4f alpha=%.4f ev=%.3f "
                      "mean_ep_reward=%.2f (%.1f steps/s)" % (
                          global_step, " [critic-warmup]" if stats.get("freeze_actor") else "",
                          stats["critic_loss"], stats["actor_loss"], stats["alpha"],
                          stats.get("explained_variance", float("nan")),
                          mean_reward, steps_per_sec))
                last_log_time, last_log_step = time.time(), global_step

            if global_step % config["save_every_steps"] == 0:
                ckpt_path = output_dir / ("sac_step_%08d.pt" % global_step)
                save_checkpoint(ckpt_path, agent, global_step, config, contract)
                save_checkpoint(output_dir / "sac_latest.pt", agent, global_step, config, contract)
                print("Da luu checkpoint:", ckpt_path)
    except KeyboardInterrupt:
        print("\nDung boi Ctrl+C — luu checkpoint truoc khi thoat...")
        save_checkpoint(output_dir / "sac_interrupted.pt", agent, global_step, config, contract)
    finally:
        env.close()
        episode_log.close()
        update_log.close()


if __name__ == "__main__":
    main()
