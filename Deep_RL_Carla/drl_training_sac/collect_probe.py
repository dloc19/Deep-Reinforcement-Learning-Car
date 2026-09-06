# -*- coding: utf-8 -*-
"""Thu mot bo QUAN SAT THAT lam chuan do do troi cua policy.

Vi sao can: ban truoc cua phep do troi danh gia mang tren `randint(0,4)` + `randn` — anh
nhieu ngau nhien, hoan toan ngoai phan phoi. No bao "troi 0.017" trong khi hieu nang thuc
te sup tu 50% xuong 93% va cham (runs/sac_v2_smoke5). Mot mang co the gan nhu khong doi
tren nhieu ma doi rat nhieu tren anh lai xe that, nen con so do khong dung de ket luan.

Bo probe nay lay tu chinh env, bang chinh policy IL, tren nhieu ban do — tuc dung phan
phoi ma policy se gap khi chay.
"""
import sys, os
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import load_config                                    # noqa: E402
from policy.checkpoint_io import load_il_checkpoint                # noqa: E402
from policy.observation import ObservationContract                # noqa: E402
from sac.networks import GaussianPolicy, load_il_actor_weights       # noqa: E402
from policy.backbone import freeze_batchnorm                      # noqa: E402
from envs.carla_lane_keep_env import CarlaLaneKeepEnv             # noqa: E402
from train_sac import resolve_target_speed                        # noqa: E402


def main():
    out = "probe_obs.npz"
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 256
    towns = sys.argv[2].split(",") if len(sys.argv) > 2 else ["Town01", "Town04"]

    cfg = load_config(["--config", "sac_config.json"])
    cfg["town"] = towns
    cfg["town_rotate_episodes"] = 3
    cfg["max_episode_steps"] = 200
    ck = load_il_checkpoint(cfg["il_checkpoint"], "cpu")
    contract = ObservationContract(ck)
    cfg["target_speed_mps"] = resolve_target_speed(cfg, contract)

    actor = GaussianPolicy(contract.scalar_feature_dim, contract.num_classes,
                           log_std_init=list(contract.default_log_std()))
    load_il_actor_weights(actor, ck)
    freeze_batchnorm(actor)
    actor.eval()

    env = CarlaLaneKeepEnv(cfg, contract)
    segs, scas, towns_seen = [], [], []
    try:
        while len(segs) < n_target:
            obs, _ = env.reset()
            done = False
            while not done and len(segs) < n_target:
                seg_t = torch.as_tensor(obs["seg"]).unsqueeze(0)
                sca_t = torch.as_tensor(obs["scalar"]).unsqueeze(0)
                with torch.no_grad():
                    action = actor.select_action(seg_t, sca_t, deterministic=True)
                obs, _r, term, trunc, _i = env.step(action.squeeze(0).numpy())
                done = term or trunc
                # Lay mau thua ra de bo probe trai rong tren quy dao, khong phai 256 khung
                # lien tiep gan nhu giong het nhau.
                if len(segs) % 1 == 0:
                    segs.append(obs["seg"].copy())
                    scas.append(obs["scalar"].copy())
                    towns_seen.append(env.current_town)
            print("  da thu %d/%d quan sat (%s)" % (len(segs), n_target, env.current_town))
    finally:
        env.close()

    np.savez_compressed(out, seg=np.array(segs, dtype=np.uint8),
                        scalar=np.array(scas, dtype=np.float32),
                        town=np.array(towns_seen))
    import collections
    print("Da luu %s: %d quan sat that, phan bo ban do %s" % (
        out, len(segs), dict(collections.Counter(towns_seen))))


if __name__ == "__main__":
    main()
