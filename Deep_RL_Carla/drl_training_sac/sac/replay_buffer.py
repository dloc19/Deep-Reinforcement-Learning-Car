"""Fixed-capacity circular off-policy replay buffer, memory-optimized the way
Stable-Baselines3's `ReplayBuffer(optimize_memory_usage=True)` is: each observation is
stored exactly ONCE, not once as `obs` and again as the next transition's `next_obs`.
`next_obs` for index `idx` is simply `obs[idx + 1]` at sample time.

Why this matters here specifically: at even a modest 128x160 env resolution, storing
`obs`+`next_obs` separately would double the (already the dominant cost) segmentation-map
memory for a buffer this size — e.g. 100k transitions x 128x160 bytes x2 ~= 4.1 GB vs ~2.0 GB
this way. This exact trade-off is why PPO (no replay buffer at all) was offered as the
default in PPO (`../drl_training/train_ppo.py`); SAC needs one for off-policy sample
efficiency, so keeping it as
lean as possible matters.

Time-limit truncation: the caller (`train_sac.py`) DOES `add()` the transition that ends in
*truncation* (`truncated and not terminated`), but passes `truncated=True` so this buffer
marks that slot UNSAMPLABLE (`self.invalid`). Reason: this buffer derives `next_obs` from the
following slot, and a truncated episode is followed by `env.reset()`, so that slot holds an
unrelated new episode's first observation.

Why store-and-mask rather than simply skipping the `add()` (what the previous version did):
skipping does not remove the bad pair, it moves it one step back. If the last transition of
the episode is never written, then the observation it carried (`obs[L-1]`) never enters the
buffer, so the PREVIOUS transition (`L-2`, which is perfectly valid and IS sampled, with
`done = 0`) ends up deriving its `next_obs` from the next episode's reset observation.
Storing it and masking only the index keeps `L-2` correctly paired and costs one unusable
slot per episode (~1 in 500 here).

`done` stored here means *true* termination only (collision / sustained off-lane), never
"episode ended for any reason" — it is used as-is as the Bellman-backup mask. A truly
terminated transition IS sampled: its bogus `next_obs` is multiplied by `1 - done = 0`.
"""

import numpy as np
import torch


class ReplayBuffer(object):
    def __init__(self, capacity, seg_shape, scalar_dim, action_dim, device,
                 n_step=1, gamma=0.99):
        """`n_step` > 1 -> tra ve n-step return thay vi 1-step.

        VI SAO CAN (do duoc tren runs/sac_d va sac_f): do nhay hanh dong cua critic — quet
        lenh lai toan dai lam Q doi bao nhieu so voi bien thien Q giua cac trang thai — BAO
        HOA quanh 8-10% tu step 25 000 va khong nhuc nhich trong 30 000 step sau do. Actor
        chi hoc qua dQ/da nen do la tran cua ca thuat toan.

        Nguyen nhan la ti le giua mot hanh dong va chan troi. Moi hanh dong keo dai 0.2s
        (`action_repeat` 4). Voi gamma 0.95, chan troi ~20 buoc, nen MOT hanh dong chiem 5%
        chan troi — phan con lai la V(s). Ha gamma tiep thi chan troi ngan hon ca thoi gian
        mot cu be lai the hien hau qua (1-3 giay). n-step di duong khac: gop n phan thuong
        THUC TE vao muc tieu truoc khi bootstrap, nen tin hieu ve hanh dong dau tien manh
        len ma chan troi hieu dung KHONG ngan lai. Day dung la co che GAE cua PPO dung de
        cong don hang chuc hanh dong doc quy dao.

        DANH DOI (phai noi ro): n-step return khong hieu chinh importance sampling la CO
        THIEN LECH voi du lieu off-policy — n-1 hanh dong giua duong duoc lay boi mot policy
        cu hon. Day la lua chon chuan trong tai lieu (Rainbow, D4PG deu dung n-step khong
        hieu chinh) va thien lech nho khi policy doi cham, ma o day policy doi RAT cham:
        actor_lr 1e-5 cong rang buoc BC neo ve checkpoint IL.

        *** XUNG KHAC VOI `explore_epsilon` — DO DUOC tren runs/sac_g, 6/9/2026 ***

        Lap luan "policy doi cham nen thien lech nho" o tren SAI khi `explore_epsilon` > 0.
        Trong giai doan critic-warmup, eps = 0.25 chen hanh dong NGAU NHIEN TOAN DAI vao 25%
        so buoc. Voi n = 3, xac suat it nhat mot trong ba hanh dong cua cua so la ngau nhien:
            1 - 0.75^3 = 58%
        Tuc 58% muc tieu n-step bi nhiem phan thuong sinh boi hanh dong ma policy dich khong
        bao gio chon. O ban 1-step, thien lech chi cham dung MOT transition.

        Do duoc, sac_g vs sac_f (giong het nhau MOI tham so tru n_step), theo gradient step:
            grad 250:  sac_f +0.581  |  sac_g -0.431
            grad 500:  sac_f +0.527  |  sac_g -1.730
            grad 750:  sac_f +0.931  |  sac_g -0.246
        `runs/smoke_nstep` (cung n=3) lai dat ev 0.94 vi `critic_warmup_steps` chi 1000 thay
        vi 4000 va episode ngan hon (200 vs 500) nen nhieu truncation -> n-step tu bi cat.

        CACH DUNG DUNG: giu n_step = 1 trong suot giai doan warmup (luc eps con bat), roi
        moi bat n-step len sau do — dat thang `buffer.n_step` tu vong lap train. Hai co che
        nham hai muc dich khac nhau va khong chong len nhau: eps la de critic thay hau qua
        cua hanh dong da dang; n-step la de actor co gradient manh hon, ma actor chi hoc SAU
        warmup.
        """
        self.capacity = capacity
        self.device = device
        self.n_step = max(1, int(n_step))
        self.gamma = float(gamma)
        self.seg = np.zeros((capacity,) + tuple(seg_shape), dtype=np.uint8)
        self.scalar = np.zeros((capacity, scalar_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)  # true termination only, see module docstring
        # Slot co `next_obs` suy ra tu slot ke tiep la VO NGHIA (transition cuoi cung cua mot
        # episode bi cat vi het gio) -> khong duoc sample. Xem docstring module.
        self.invalid = np.zeros(capacity, dtype=np.bool_)
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, seg, scalar, action, reward, done, truncated=False):
        """`done`: TRUE termination only (collision / off-lane) — the Bellman mask.
        `truncated`: episode was cut off by the time limit at this transition. Van luu (de
        transition lien truoc con lay dung `next_obs`), nhung danh dau khong sample duoc.
        """
        idx = self.pos
        self.seg[idx] = seg
        self.scalar[idx] = scalar
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = float(done)
        # Ghi DE co cu khi vong lai — slot nay vua duoc dung lai cho transition khac.
        self.invalid[idx] = bool(truncated)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size < 2:
            raise ValueError("Buffer can >=2 transition de sample (hien co %d)." % self.size)

        last_idx = (self.pos - 1) % self.capacity  # most recently added — see module docstring
        if self.size < self.capacity:
            # Chưa wrap: chỉ [0, pos-1) là hợp lệ — slot `pos` chưa từng được ghi, không thể
            # dùng làm next_obs.
            high = self.size - 1
        else:
            high = self.capacity
        idx = np.random.randint(0, high, size=batch_size)
        # Loai ca `last_idx` (slot ke tiep chua duoc ghi) lan cac slot bi danh dau `invalid`
        # (transition cuoi cua mot episode bi cat gio — xem docstring module).
        bad = (idx == last_idx) | self.invalid[idx]
        for _attempt in range(100):
            if not bad.any():
                break
            idx[bad] = np.random.randint(0, high, size=int(bad.sum()))
            bad = (idx == last_idx) | self.invalid[idx]
        else:
            raise RuntimeError(
                "Khong lay du %d transition hop le sau 100 lan thu (size=%d, invalid=%d). "
                "Buffer gan nhu chi chua transition bi cat gio — kiem tra "
                "max_episode_steps/learning_starts." % (batch_size, self.size, int(self.invalid[:high].sum())))

        boot_idx, n_reward, n_done, gamma_n = self._rollout(idx, last_idx)

        to_tensor = lambda arr: torch.as_tensor(arr).to(self.device)  # noqa: E731
        return {
            "seg": to_tensor(self.seg[idx]),
            "scalar": to_tensor(self.scalar[idx]),
            "action": to_tensor(self.action[idx]),
            "reward": to_tensor(n_reward),
            "done": to_tensor(n_done),
            "gamma_n": to_tensor(gamma_n),
            "next_seg": to_tensor(self.seg[boot_idx]),
            "next_scalar": to_tensor(self.scalar[boot_idx]),
        }

    def _rollout(self, idx, last_idx):
        """Cong don n-step tu moi chi so trong `idx`.

        Tra ve (boot_idx, reward, done, gamma_n) sao cho muc tieu Bellman la
            y = reward + gamma_n * (1 - done) * Q_target(obs[boot_idx], a')
        Voi n_step = 1 thi ket qua trung KHIT ban 1-step cu (gamma_n = gamma o moi phan tu).

        Vong lap dung SOM tai mot slot khi khong the di tiep qua no, va bootstrap ngay tai
        quan sat cua slot do — luon cho ra mot k-step return HOP LE voi k <= n:
          * `invalid[j]`  : slot cuoi cua episode bi cat vi het gio. `next_obs` cua no thuoc
                            episode khac, nen khong the di xuyen qua; dung lai va bootstrap
                            tai obs[j] la dung (day chi la n-step ngan hon).
          * `j == last_idx`: slot vua ghi gan nhat — obs[j+1] chua ton tai.
          * `done[j]`     : ket thuc that su. Cong phan thuong roi dat done=1; muc tieu bi
                            che boi (1 - done) nen `boot_idx` khong con quan trong.
        """
        n = self.n_step
        boot = idx.copy()
        reward = np.zeros(len(idx), dtype=np.float32)
        done = np.zeros(len(idx), dtype=np.float32)
        discount = np.ones(len(idx), dtype=np.float32)
        active = np.ones(len(idx), dtype=np.bool_)

        for _k in range(n):
            can = active & (boot != last_idx) & (~self.invalid[boot])
            if not can.any():
                break
            here = boot[can]
            reward[can] += discount[can] * self.reward[here]
            discount[can] *= self.gamma
            boot[can] = (here + 1) % self.capacity
            terminal = np.zeros(len(idx), dtype=np.bool_)
            terminal[can] = self.done[here] > 0
            done[terminal] = 1.0
            active = can & (~terminal)

        return boot, reward, done, discount
