"""Bridges to Deep_RL_Carla/drl_training — Phase 4 (design doc §07: IL / DRL Autopilot).

drl_training is a sibling repo folder (`Do_An/Deep_RL_Carla/drl_training`), not part of
bridge_server — same situation as `router_plan` (see route_planning.py's docstring), so this
module follows the exact same shape: `ensure_on_path()`/`load()` get the package importable
and re-export the pieces the bridge needs unchanged, deferred to call time so
`BridgeConfig.deep_rl_carla_root` can override the path first.

On top of that re-export, this module also owns the two things that are specific to *live
inference* rather than training (`evaluate.py`'s job) or offline learning (the notebooks'
job): turning an IL checkpoint into a ready-to-call `predict(seg, scalar) -> action`
function, and the same for a PPO/SAC checkpoint. `bridge/modes/learned_autopilot.py` and
`bridge/sim_loop.py` only ever see the return values of `build_il_predictor`/
`build_drl_predictor` — never torch tensors directly.
"""

import sys
from pathlib import Path

_BRIDGE_SERVER_ROOT = Path(__file__).resolve().parents[1]          # .../CarlaDashBoard/bridge_server
_DEFAULT_DEEP_RL_CARLA_ROOT = _BRIDGE_SERVER_ROOT.parent.parent / "Deep_RL_Carla"  # .../Do_An/Deep_RL_Carla


def resolve_root(deep_rl_carla_root: str = "") -> Path:
    root = Path(deep_rl_carla_root) if deep_rl_carla_root else _DEFAULT_DEEP_RL_CARLA_ROOT
    return root.resolve()


def ensure_on_path(deep_rl_carla_root: str = ""):
    drl_training_dir = resolve_root(deep_rl_carla_root) / "drl_training"
    if not drl_training_dir.is_dir():
        raise RuntimeError(
            "Khong tim thay 'drl_training' trong %s — truyen --deep-rl-carla-root neu "
            "Deep_RL_Carla khong nam canh CarlaDashBoard." % drl_training_dir.parent)
    if str(drl_training_dir) not in sys.path:
        sys.path.insert(0, str(drl_training_dir))


def load(deep_rl_carla_root: str = ""):
    """Import + return drl_training's policy building blocks as a small namespace — the
    same re-export pattern `route_planning.load()` uses for router_plan."""
    ensure_on_path(deep_rl_carla_root)
    from policy.actor_critic import GaussianActor, load_il_actor_weights
    from policy.checkpoint_io import load_il_checkpoint
    from policy.observation import ObservationContract, resize_class_map, build_vehicle_state

    class _DrlTraining:
        pass

    ns = _DrlTraining()
    ns.GaussianActor = GaussianActor
    ns.load_il_actor_weights = load_il_actor_weights
    ns.load_il_checkpoint = load_il_checkpoint
    ns.ObservationContract = ObservationContract
    ns.resize_class_map = resize_class_map
    ns.build_vehicle_state = build_vehicle_state
    return ns


def resolve_device(device_name: str):
    import torch
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_il_predictor(ns, checkpoint_path, device_name="cpu"):
    """Loads an IL checkpoint (`behavior_cloning/train_il_v9.ipynb` output) and returns
    `(contract, predict)`. `predict(seg, scalar) -> np.ndarray[2]` (steer, throttle/brake in
    [-1, 1]) is deterministic — IL has no notion of an exploration distribution, it's a
    straight regression to the demonstrated action, so there's no `deterministic` flag to
    thread through here the way `evaluate.py` does for PPO/SAC.

    Reuses `policy.actor_critic.GaussianActor` (PPO's actor class) purely as the network
    shape IL's weights remap onto — its `.log_std`/`.distribution()` machinery is simply
    unused. That's the same warm-start path `train_ppo.py` takes before its first rollout,
    just without any subsequent PPO training.
    """
    import torch
    device = resolve_device(device_name)
    checkpoint = ns.load_il_checkpoint(checkpoint_path, map_location="cpu")
    contract = ns.ObservationContract(checkpoint)
    actor = ns.GaussianActor(contract.scalar_feature_dim, contract.num_classes)
    ns.load_il_actor_weights(actor, checkpoint)
    actor.to(device).eval()

    def predict(seg, scalar):
        seg_t = torch.as_tensor(seg, device=device).unsqueeze(0)
        scalar_t = torch.as_tensor(scalar, device=device).unsqueeze(0)
        with torch.no_grad():
            action = actor.mean_action(seg_t, scalar_t)
        return action.squeeze(0).cpu().numpy()

    return contract, predict


def _build_agent(algorithm, contract, device):
    """Same construction `evaluate.py::build_agent` uses — duplicated rather than imported
    because that function lives in a CLI script's module scope, not drl_training's public
    package API."""
    if algorithm == "ppo":
        from policy.actor_critic import GaussianActor, ValueCritic
        from ppo.ppo_agent import PPOAgent
        actor = GaussianActor(contract.scalar_feature_dim, contract.num_classes)
        critic = ValueCritic(contract.scalar_feature_dim, contract.num_classes)
        return PPOAgent(actor, critic, {}, device)
    if algorithm == "sac":
        from sac.networks import GaussianPolicy, TwinQNetwork
        from sac.sac_agent import SACAgent
        actor = GaussianPolicy(contract.scalar_feature_dim, contract.num_classes)
        critic = TwinQNetwork(contract.scalar_feature_dim, action_dim=2, num_classes=contract.num_classes)
        return SACAgent(actor, critic, {}, device)
    raise ValueError("drl_algorithm phải là 'ppo' hoặc 'sac', nhận được '%s'" % algorithm)


def describe_checkpoint(path):
    """Vai dong tom tat metadata cua mot checkpoint DRL, de ghi vao log khi khoi dong.

    Muc dich la lam LO ra viec nap nham checkpoint. Nap nham khong gay loi nao: shape van
    khop (kien truc khong doi giua cac lan train), xe van lai duoc, chi la lai bang mot
    policy khac voi y dinh. Rieng `update` con cho biet actor da thuc su duoc train chua —
    trong `critic_warmup_updates` dau tien cua moi lan chay, actor bi dong bang nen trong so
    van y het IL.
    """
    import torch
    try:
        ck = torch.load(str(path), map_location="cpu")
    except Exception as exc:                                     # noqa: BLE001
        return ["khong doc duoc metadata: %s" % exc]
    cfg = ck.get("config") or {}
    town = cfg.get("town")
    lines = ["update=%s | algorithm=%s | reward_mode=%s" % (
        ck.get("update", "?"), ck.get("algorithm", "?"), cfg.get("reward_mode", "raw/khong ro"))]
    lines.append("train tren town=%s | collision_penalty=%s | w_lane_offset=%s" % (
        town, cfg.get("collision_penalty"), cfg.get("w_lane_offset")))
    warmup = cfg.get("critic_warmup_updates")
    update = ck.get("update")
    if isinstance(update, int) and isinstance(warmup, int) and update <= warmup:
        lines.append("[!] CANH BAO: update=%d <= critic_warmup_updates=%d — actor con DONG BANG "
                     "o checkpoint nay, trong so van la IL nguyen ban, chua hoc DRL." % (update, warmup))
    return lines


def build_drl_predictor(ns, algorithm, il_checkpoint_path, drl_checkpoint_path, device_name="cuda"):
    """Loads a PPO/SAC checkpoint (`train_ppo.py`/`train_sac.py` output) the same way
    `evaluate.py` does for offline evaluation — including its `il_checkpoint` step, since the
    observation contract (scalar feature layout, normalization stats) is only ever saved on
    the IL checkpoint, never duplicated onto the DRL one. Returns `(contract, algorithm,
    predict)`; `algorithm` may differ from the argument if the checkpoint itself declares a
    different one (same override `evaluate.py` applies) — the caller should use it as the
    cache key, not the requested value.
    """
    import torch
    device = resolve_device(device_name)
    il_checkpoint = ns.load_il_checkpoint(il_checkpoint_path, map_location="cpu")
    contract = ns.ObservationContract(il_checkpoint)

    checkpoint = torch.load(drl_checkpoint_path, map_location=device)
    checkpoint_algo = checkpoint.get("algorithm")
    if checkpoint_algo and checkpoint_algo != algorithm:
        algorithm = checkpoint_algo

    agent = _build_agent(algorithm, contract, device)
    agent.load_state_dict(checkpoint)
    agent.actor.eval()
    agent.critic.eval()

    def predict(seg, scalar):
        return agent.select_action(seg, scalar, deterministic=True)

    return contract, algorithm, predict
