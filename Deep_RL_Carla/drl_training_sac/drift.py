# -*- coding: utf-8 -*-
"""Do do troi cua policy tren QUAN SAT THAT (probe_obs.npz), khong phai nhieu ngau nhien."""
import sys, os
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from policy.checkpoint_io import load_il_checkpoint
from policy.observation import ObservationContract
from policy.backbone import freeze_batchnorm
from sac.networks import GaussianPolicy, load_il_actor_weights

d = np.load("probe_obs.npz")
seg = torch.as_tensor(d["seg"]); sca = torch.as_tensor(d["scalar"])
ck = load_il_checkpoint("../behavior_cloning/best_il_model.pth", "cpu")
c = ObservationContract(ck); pair = list(c.default_log_std())

base = GaussianPolicy(c.scalar_feature_dim, c.num_classes, log_std_init=pair)
load_il_actor_weights(base, ck); freeze_batchnorm(base); base.eval()
with torch.no_grad():
    m0, ls0 = base.forward(seg, sca)
    a0 = torch.tanh(m0)

print("Bo probe: %d quan sat that | ban do %s" % (
    len(seg), dict(zip(*np.unique(d["town"], return_counts=True)))))
print("IL: |lai| TB %.4f  |ga| TB %.4f" % (a0[:,0].abs().mean(), a0[:,1].abs().mean()))
print()
print("%-46s %10s %10s %10s" % ("checkpoint", "troi lai", "troi ga", "std lai max"))
for p in sys.argv[1:]:
    if not os.path.exists(p):
        print("%-46s KHONG TON TAI" % p); continue
    a = GaussianPolicy(c.scalar_feature_dim, c.num_classes, log_std_init=pair)
    a.load_state_dict(torch.load(p, map_location="cpu")["actor"])
    freeze_batchnorm(a); a.eval()
    with torch.no_grad():
        m1, ls1 = a.forward(seg, sca)
        a1 = torch.tanh(m1)
    print("%-46s %10.4f %10.4f %10.4f" % (
        p.replace("\\", "/").split("/")[-2] + "/" + p.replace("\\", "/").split("/")[-1],
        (a1[:,0]-a0[:,0]).abs().mean(), (a1[:,1]-a0[:,1]).abs().mean(), ls1[:,0].exp().max()))
