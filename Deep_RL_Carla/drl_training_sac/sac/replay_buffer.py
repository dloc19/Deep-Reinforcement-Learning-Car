"""Fixed-capacity circular off-policy replay buffer, memory-optimized the way
Stable-Baselines3's `ReplayBuffer(optimize_memory_usage=True)` is: each observation is
stored exactly ONCE, not once as `obs` and again as the next transition's `next_obs`.
`next_obs` for index `idx` is simply `obs[idx + 1]` at sample time.

Why this matters here specifically: at even a modest 128x160 env resolution, storing
`obs`+`next_obs` separately would double the (already the dominant cost) segmentation-map
memory for a buffer this size — e.g. 100k transitions x 128x160 bytes x2 ~= 4.1 GB vs ~2.0 GB
this way. This exact trade-off is why PPO (no replay buffer at all) was offered as the
default in `train_ppo.py`; SAC needs one for off-policy sample efficiency, so keeping it as
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
    def __init__(self, capacity, seg_shape, scalar_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
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

        next_idx = (idx + 1) % self.capacity
        to_tensor = lambda arr: torch.as_tensor(arr).to(self.device)  # noqa: E731
        return {
            "seg": to_tensor(self.seg[idx]),
            "scalar": to_tensor(self.scalar[idx]),
            "action": to_tensor(self.action[idx]),
            "reward": to_tensor(self.reward[idx]),
            "done": to_tensor(self.done[idx]),
            "next_seg": to_tensor(self.seg[next_idx]),
            "next_scalar": to_tensor(self.scalar[next_idx]),
        }
