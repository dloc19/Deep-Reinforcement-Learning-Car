"""Do nang luc phan cung roi DE XUAT cau hinh train — chay TRUOC khi bat dau train dai.

Vi sao can: tren may phat trien (GTX 1650 Max-Q 4 GB) do duoc PPO chi dung GPU 10% thoi
gian moi env-step (4.4 ms / 42.9 ms) — nut that la CARLA tick, nen GPU manh hon gan nhu
khong giup gi cho PPO. SAC thi nguoc han: `update()` chiem 97% (1665 ms / 1704 ms), nen no
la thu DUY NHAT thuc su huong loi tu may thue.

Ty le do phu thuoc phan cung, khong the ngoai suy tu may nay sang may khac. Script nay do
lai tren CHINH may ban dang dung roi in ra cau hinh nen dat + thoi gian du kien, de ban
quyet dinh bang so lieu thay vi doan.

    python bench_hardware.py                 # chi do GPU (khong can CARLA)
    python bench_hardware.py --with-carla    # do them thoi gian tick that cua CARLA

Khong ghi gi vao config — chi in ra de ban tu doi.
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from policy.actor_critic import GaussianActor, ValueCritic  # noqa: E402
from policy.checkpoint_io import load_il_checkpoint  # noqa: E402
from policy.observation import ObservationContract  # noqa: E402
from ppo.ppo_agent import PPOAgent  # noqa: E402
from ppo.rollout_buffer import RolloutBuffer  # noqa: E402
from sac.networks import GaussianPolicy, TwinQNetwork  # noqa: E402
from sac.replay_buffer import ReplayBuffer  # noqa: E402
from sac.sac_agent import SACAgent  # noqa: E402

# Muc tieu thuc dung: duoi nguong nay thi mot lan train qua dem la du, tren nguong thi phai
# chia nhieu phien hoac ha batch.
TARGET_HOURS = 8.0


def _time(fn, repeats, warmup=2):
    """Warmup that su can thiet, khong phai hinh thuc: lan goi dau gom ca cuDNN autotune va
    cap phat bo nho, do duoc cham gap ~17 lan lan thu hai tren cung mot may."""
    for _ in range(warmup):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(repeats):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.time() - start) / repeats


def bench_carla(config, contract):
    """Thoi gian THAT cua mot quyet dinh policy o phia moi truong (action_repeat tick +
    doc frame + dung observation), do bang chinh env se dung luc train."""
    from envs.carla_lane_keep_env import CarlaLaneKeepEnv
    env = CarlaLaneKeepEnv(config, contract)
    try:
        obs, _ = env.reset()
        action = np.zeros(2, dtype=np.float32)
        for _ in range(5):
            env.step(action)
        start, steps = time.time(), 40
        for _ in range(steps):
            _obs, _r, terminated, truncated, _info = env.step(action)
            if terminated or truncated:
                env.reset()
        return (time.time() - start) / steps
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--il-checkpoint", default="../behavior_cloning/best_il_model.pth")
    parser.add_argument("--with-carla", action="store_true",
                        help="Do them thoi gian tick that (can CARLA server dang chay)")
    parser.add_argument("--carla-step-ms", type=float, default=38.5,
                        help="Thoi gian moi quyet dinh o phia CARLA (ms) khi khong do duoc. "
                             "Mac dinh 38.5 = do tren may phat trien, GTX 1650 Max-Q")
    args = parser.parse_args()

    if torch.cuda.is_available():
        prop = torch.cuda.get_device_properties(0)
        device = torch.device("cuda")
        print("GPU     : %s  |  VRAM %.1f GB" % (prop.name, prop.total_memory / 1e9))
    else:
        device = torch.device("cpu")
        print("GPU     : KHONG CO — dang chay CPU, moi con so duoi day se rat cham")
    print("torch   : %s  |  python %s" % (torch.__version__, sys.version.split()[0]))

    checkpoint = load_il_checkpoint(Path(args.il_checkpoint).expanduser().resolve())
    contract = ObservationContract(checkpoint)
    height, width = checkpoint["image_height"], checkpoint["image_width"]
    dim, classes = contract.scalar_feature_dim, contract.num_classes
    print("quan sat: %dx%d, %d dac trung scalar, %d lop\n" % (width, height, dim, classes))
    torch.backends.cudnn.benchmark = True
    rng = np.random.default_rng(0)

    # ---- CARLA -------------------------------------------------------------------------
    carla_ms = args.carla_step_ms
    if args.with_carla:
        from config import load_config
        cfg = load_config("ppo", ["--config", "ppo_config.json"])
        print("Dang do CARLA (spawn xe + 40 quyet dinh)...")
        carla_ms = 1000.0 * bench_carla(cfg, contract)
        print("  do duoc: %.1f ms moi quyet dinh\n" % carla_ms)
    else:
        print("CARLA   : dung gia tri mac dinh %.1f ms/quyet dinh "
              "(chay lai voi --with-carla de do that)\n" % carla_ms)

    # ---- PPO ---------------------------------------------------------------------------
    n_steps = 1024
    ppo_cfg = {"actor_lr": 2e-5, "critic_lr": 3e-4, "clip_range": 0.2, "value_clip_range": 0.2,
               "entropy_coef": 0.0, "value_coef": 0.5, "max_grad_norm": 0.5, "epochs": 10,
               "batch_size": 128, "target_kl": 0.02, "log_std_init": (-3.0, -1.5)}
    ppo = PPOAgent(GaussianActor(dim, classes), ValueCritic(dim, classes), ppo_cfg, device)
    rollout = RolloutBuffer(n_steps, (height, width), dim, 2, device)
    for _ in range(n_steps):
        rollout.add(rng.integers(0, 4, (height, width), dtype=np.uint8),
                    rng.normal(0, 1, dim).astype(np.float32),
                    rng.normal(0, 1, 2).astype(np.float32), -1.0, 1.0, 0.5, False)
    adv, ret = rollout.compute_gae(0.0, 0.99, 0.95)
    ppo_update_s = _time(lambda: ppo.update(rollout, adv, ret, freeze_actor=False), 2, warmup=1)
    ppo_ms = carla_ms + 1000.0 * ppo_update_s / n_steps

    print("=" * 74)
    print("PPO  (n_steps=1024, epochs=10, batch=128)")
    print("=" * 74)
    print("  update mot rollout : %8.0f ms  -> khau hao %.1f ms/buoc" % (
        1000 * ppo_update_s, 1000 * ppo_update_s / n_steps))
    print("  moi env-step       : %8.1f ms  (CARLA %.1f + GPU %.1f)" % (
        ppo_ms, carla_ms, 1000 * ppo_update_s / n_steps))
    print("  200 000 buoc       : %8.1f gio   [%.0f%% thoi gian la GPU]" % (
        200000 * ppo_ms / 3.6e6, 100 * (1000 * ppo_update_s / n_steps) / ppo_ms))
    if (1000 * ppo_update_s / n_steps) / ppo_ms < 0.25:
        print("  -> PPO bi chan boi CARLA, khong phai GPU. Thue GPU manh hon gan nhu")
        print("     KHONG rut ngan duoc PPO; tien nen dat vao CPU/GPU render nhanh hon.")

    # ---- SAC ---------------------------------------------------------------------------
    sac_cfg = {"actor_lr": 3e-4, "critic_lr": 3e-4, "alpha_lr": 3e-4, "tau": 0.005,
               "gamma": 0.99, "target_entropy": -4.0, "log_std_init": -2.5,
               "max_grad_norm": 0.5}
    sac = SACAgent(GaussianPolicy(dim, classes), TwinQNetwork(dim, 2, classes), sac_cfg, device)
    replay = ReplayBuffer(400, (height, width), dim, 2, device)
    for _ in range(300):
        replay.add(rng.integers(0, 4, (height, width), dtype=np.uint8),
                   rng.normal(0, 1, dim).astype(np.float32),
                   rng.normal(0, 1, 2).astype(np.float32), 1.0, False)

    print("\n" + "=" * 74)
    print("SAC  (100 000 buoc) — tim cau hinh dat duoi %.0f gio" % TARGET_HOURS)
    print("=" * 74)
    print("  %-7s %-11s %12s %11s %12s" % ("batch", "train_freq", "update ms", "buoc/giay", "100k (gio)"))
    best = None
    for batch in (128, 64, 32):
        update_s = _time(lambda b=batch: sac.update(replay.sample(b)), 4, warmup=2)
        for freq in (1, 2, 3):
            step_ms = carla_ms + 1000.0 * update_s / freq
            hours = 100000 * step_ms / 3.6e6
            flag = ""
            # Uu tien cau hinh GIU nguyen train_freq=1 va batch lon: do la SAC "dung sach",
            # va ty le update/data = 1.0 chinh la luan diem hieu qua mau de so voi PPO.
            if hours <= TARGET_HOURS and best is None:
                best, flag = (batch, freq, hours), "   <== nen dung"
            print("  %-7d %-11d %12.0f %11.1f %12.1f%s" % (
                batch, freq, 1000 * update_s, 1000.0 / step_ms, hours, flag))

    print("\n" + "=" * 74)
    if best is None:
        print("KHONG cau hinh nao dat duoi %.0f gio. Lua chon:" % TARGET_HOURS)
        print("  - ha total_steps cua SAC (vd 50 000) va noi ro trong bao cao, hoac")
        print("  - thue GPU co bang thong bo nho cao hon (update SAC bi chan boi bang thong:")
        print("    moi lan chay ~6 luot backbone tren tensor one-hot batch x 4 x %d x %d)" % (height, width))
    else:
        batch, freq, hours = best
        print("DE XUAT: batch_size=%d, train_freq=%d  ->  100k buoc ~ %.1f gio" % (batch, freq, hours))
        if freq > 1:
            print("  LUU Y cho bao cao: train_freq=%d ha ty le update/data cua SAC tu 1.0 xuong" % freq)
            print("  %.2f. SAC van hieu qua mau hon PPO (0.078) nhung khoang cach thu hep —" % (1.0 / freq))
            print("  neu bao cao lay 'hieu qua mau' lam luan diem thi phai ghi ro con so nay.")
        print("\n  Sua trong sac_config.json:  \"batch_size\": %d, \"train_freq\": %d" % (batch, freq))
    print("=" * 74)


if __name__ == "__main__":
    main()
