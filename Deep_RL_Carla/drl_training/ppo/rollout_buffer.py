"""Fixed-size on-policy rollout buffer with GAE(lambda) advantage estimation.

Stores the segmentation observation as a raw (H, W) uint8 class-ID map — NOT one-hot — to
keep memory bounded: one-hot at `num_classes=13` would be 13x the memory for no benefit,
since `PolicyBackbone.forward` one-hot-encodes on the GPU right before the conv stack (see
`policy/backbone.py`). This is the whole reason PPO (on-policy, buffer sized ~n_steps) was
chosen over an off-policy method here: even at 1 byte/pixel, an off-policy replay buffer
large enough for sample-efficient training would dwarf this buffer by 100x+.
"""

import numpy as np
import torch


class RolloutBuffer:
    def __init__(self, capacity, seg_shape, scalar_dim, action_dim, device):
        self.capacity = capacity
        self.device = device
        self.seg = np.zeros((capacity,) + tuple(seg_shape), dtype=np.uint8)
        self.scalar = np.zeros((capacity, scalar_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.log_probs = np.zeros(capacity, dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.values = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0

    def add(self, seg, scalar, action, log_prob, reward, value, done):
        i = self.ptr
        self.seg[i] = seg
        self.scalar[i] = scalar
        self.actions[i] = action
        self.log_probs[i] = log_prob
        self.rewards[i] = reward
        self.values[i] = value
        self.dones[i] = float(done)
        self.ptr += 1

    def is_full(self):
        return self.ptr >= self.capacity

    def reset(self):
        self.ptr = 0

    def compute_gae(self, last_value, gamma, gae_lambda):
        """`last_value`: critic estimate of the state *after* the last stored transition
        (0.0 if that transition was a true terminal — the caller is responsible for that,
        see `train_ppo.py`'s truncation-bootstrap handling before `buffer.add`).
        `dones[t] == 1` marks an episode boundary at step t (terminated OR truncated): GAE
        must not bootstrap/propagate advantage across an episode reset, regardless of why
        the episode ended.
        """
        advantages = np.zeros(self.capacity, dtype=np.float32)
        last_gae = 0.0
        for t in reversed(range(self.capacity)):
            if t == self.capacity - 1:
                next_value = last_value
            else:
                next_value = self.values[t + 1]
            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
        returns = advantages + self.values
        return advantages, returns

    def iter_minibatches(self, batch_size, advantages, returns):
        indices = np.random.permutation(self.capacity)
        for start in range(0, self.capacity, batch_size):
            idx = indices[start:start + batch_size]
            yield {
                "seg": torch.as_tensor(self.seg[idx]).to(self.device),
                "scalar": torch.as_tensor(self.scalar[idx]).to(self.device),
                "actions": torch.as_tensor(self.actions[idx]).to(self.device),
                "old_log_probs": torch.as_tensor(self.log_probs[idx]).to(self.device),
                "old_values": torch.as_tensor(self.values[idx]).to(self.device),
                "advantages": torch.as_tensor(advantages[idx]).to(self.device),
                "returns": torch.as_tensor(returns[idx]).to(self.device),
            }
