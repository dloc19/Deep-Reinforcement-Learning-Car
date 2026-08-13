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

Time-limit truncation: the caller (`train_sac.py`) must NOT call `add()` for a transition
that ends in *truncation* (`truncated and not terminated`) — see that file's loop comment.
Reason: this buffer derives `next_obs` from the following slot, but a truncated episode is
followed by `env.reset()`, so the following slot would hold an unrelated new episode's first
observation, not this transition's true continuation. Dropping that one transition per
episode (out of typically ~1000) is negligible; silently pairing it with the wrong next_obs
would inject real, hard-to-notice bootstrap error into every SAC update that samples it.
`done` stored here therefore means *true* termination only (collision / sustained off-lane),
never "episode ended for any reason" — it is used as-is as the Bellman-backup mask.
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
        self.pos = 0
        self.size = 0

    def __len__(self):
        return self.size

    def add(self, seg, scalar, action, reward, done):
        idx = self.pos
        self.seg[idx] = seg
        self.scalar[idx] = scalar
        self.action[idx] = action
        self.reward[idx] = reward
        self.done[idx] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        if self.size < 2:
            raise ValueError("Buffer can >=2 transition de sample (hien co %d)." % self.size)

        last_idx = (self.pos - 1) % self.capacity  # most recently added — see module docstring
        if self.size < self.capacity:
            # Chưa wrap: chỉ [0, pos-1) là hợp lệ — slot `pos` chưa từng được ghi, không thể
            # dùng làm next_obs.
            idx = np.random.randint(0, self.size - 1, size=batch_size)
        else:
            idx = np.random.randint(0, self.capacity, size=batch_size)
            collision = idx == last_idx
            while np.any(collision):
                idx[collision] = np.random.randint(0, self.capacity, size=int(collision.sum()))
                collision = idx == last_idx

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
