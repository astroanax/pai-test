"""HRI Push-T adapter for the latent-controllability experiment.

Self-contained (does not import execution-pullback): checkpoint loading,
514-dim condition encoding, action preparation/clipping, environment
wrappers, replay with the terminal/truncated absorbing rule, plus the
``field`` closure that exposes the frozen teacher as
``field(state, times, condition)`` for ``gad_reference.teacher_endpoint``.
"""
import numpy as np
import torch

HORIZON = 16
ACTION_DIM = 2
EXPECTED_KEYS = ("vision_encoder", "noise_pred_net")
CONDITION_DIM = 514
ACTION_LOW = 0.0
ACTION_HIGH = 512.0
SUCCESS_COVERAGE = 0.95


def freeze(module):
    module.eval()
    module.requires_grad_(False)
    return module


def load_checkpoint(path, device):
    """Strict load with modules moved onto `device`."""
    import resnet
    import unet
    device = torch.device(device)
    raw = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(raw, dict):
        raise ValueError("checkpoint is not a dictionary")
    if tuple(sorted(raw.keys())) != tuple(sorted(EXPECTED_KEYS)):
        raise ValueError("checkpoint keys do not match expected teacher keys")
    vision_encoder = resnet.get_resnet("resnet18")
    vision_encoder = resnet.replace_bn_with_gn(vision_encoder)
    noise_pred_net = unet.ConditionalUnet1D(
        input_dim=ACTION_DIM, global_cond_dim=CONDITION_DIM
    )
    vision_encoder.load_state_dict(raw["vision_encoder"], strict=True)
    noise_pred_net.load_state_dict(raw["noise_pred_net"], strict=True)
    for module in (vision_encoder, noise_pred_net):
        for name, tensor in module.state_dict().items():
            if not torch.isfinite(tensor).all():
                raise ValueError("nonfinite checkpoint tensor " + name)
    vision_encoder = freeze(vision_encoder.to(device))
    noise_pred_net = freeze(noise_pred_net.to(device))
    return vision_encoder, noise_pred_net


def _device_key(device):
    device = torch.device(device)
    if device.type == "cuda":
        return ("cuda", device.index if device.index is not None else 0)
    return (device.type, device.index)


def check_devices(vision_encoder, noise_pred_net, condition, actions, device):
    expected = _device_key(device)
    actual = {_device_key(next(vision_encoder.parameters()).device),
              _device_key(next(noise_pred_net.parameters()).device),
              _device_key(condition.device), _device_key(actions.device)}
    if actual != {expected}:
        raise ValueError(f"device mismatch: {sorted(actual)} vs {expected}")
    return sorted(actual)


def encode_condition(vision_encoder, image, agent_pos):
    if not torch.isfinite(image).all():
        raise ValueError("nonfinite image tensor")
    feature_device = next(vision_encoder.parameters()).device
    if image.device != feature_device:
        raise ValueError(f"image on {image.device}, encoder on {feature_device}")
    with torch.no_grad():
        image_features = vision_encoder(image)
    features = torch.cat([image_features, agent_pos.to(image_features.device)], dim=-1)
    condition = features.flatten(start_dim=1)
    if not torch.isfinite(condition).all():
        raise ValueError("nonfinite condition vector")
    return condition


def check_condition_dim(condition):
    if condition.shape[-1] != CONDITION_DIM:
        raise ValueError("condition dimension mismatch")


def make_field(noise_pred_net):
    def field(latent, times, condition):
        return noise_pred_net(latent, times, global_cond=condition)
    return field


def validate_stats(stats):
    for key in ("action", "agent_pos"):
        if key not in stats:
            raise ValueError("normalizer lacks " + key)
        low = np.asarray(stats[key]["min"], dtype=np.float64)
        high = np.asarray(stats[key]["max"], dtype=np.float64)
        if low.shape != high.shape:
            raise ValueError(f"normalizer {key}: min/max shape mismatch")
        if not (np.isfinite(low).all() and np.isfinite(high).all()):
            raise ValueError(f"normalizer {key}: nonfinite bounds")
        if np.any(high <= low):
            raise ValueError(f"normalizer {key}: zero or negative span")
    return True


def normalize_data(data, stats, key="action"):
    ndata = (data - stats[key]["min"]) / (stats[key]["max"] - stats[key]["min"])
    return ndata * 2 - 1


def unnormalize_data(ndata, stats, key="action"):
    ndata = (ndata + 1) / 2
    return ndata * (stats[key]["max"] - stats[key]["min"]) + stats[key]["min"]


def decode_actions(normalized, stats):
    return unnormalize_data(np.asarray(normalized), stats, key="action")


def resolve_device(name):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("cuda was requested but is not available")
    if device.type not in ("cpu", "cuda"):
        raise ValueError("unsupported device " + str(device))
    return device


class HRIAdapter:
    def __init__(self, vision_encoder, noise_pred_net, stats, device,
                 legacy=False, clip_actions=False):
        validate_stats(stats)
        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.stats = stats
        self.device = torch.device(device)
        self.legacy = legacy
        self.clip_actions = clip_actions
        for name, module in (("vision_encoder", vision_encoder),
                             ("noise_pred_net", noise_pred_net)):
            if _device_key(next(module.parameters()).device) != _device_key(self.device):
                raise ValueError(f"{name} device does not match adapter device")

    def field(self, state, times, condition):
        """Frozen-teacher vector field for gad_reference.teacher_endpoint."""
        return self.noise_pred_net(state, times, global_cond=condition)

    def teacher_action(self, q, condition, steps, source="gaussian"):
        """Full teacher map q -> action chunk, detached FP32, no grad."""
        import sys
        import os
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from gad_reference import teacher_endpoint
        with torch.no_grad():
            out = teacher_endpoint(self.field, q, condition, steps, source=source)
        if not torch.isfinite(out).all():
            raise ValueError("nonfinite teacher action")
        return out.detach().to(dtype=torch.float32)

    def new_env(self, image):
        import pusht
        if image:
            return pusht.PushTImageEnv(legacy=self.legacy)
        return pusht.PushTEnv(legacy=self.legacy)

    def reset(self, env, scene):
        env.seed(scene)
        return env.reset()

    def raw_step(self, env, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (2,):
            raise ValueError("action shape must be (2,)")
        if not np.isfinite(action).all():
            raise ValueError("nonfinite action")
        out = env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, terminated, info = out
            truncated = False
        terminated = bool(terminated)
        truncated = bool(truncated)
        return obs, float(reward), terminated, truncated, bool(terminated or truncated)

    def decode_raw(self, normalized_prefix):
        physical = np.asarray(decode_actions(normalized_prefix, self.stats))
        if not np.isfinite(physical).all():
            raise ValueError("nonfinite decoded action")
        return physical

    def prepare_commands(self, normalized_prefix):
        raw = self.decode_raw(normalized_prefix)
        flat = raw.reshape(-1, 2)
        invalid = (flat < ACTION_LOW) | (flat > ACTION_HIGH)
        raw_violations = int(invalid.sum())
        coordinates = int(invalid.size)
        prepared = np.clip(raw, ACTION_LOW, ACTION_HIGH) if self.clip_actions else raw
        prepared = np.asarray(prepared, dtype=np.float64)
        executed_invalid = (prepared < ACTION_LOW) | (prepared > ACTION_HIGH)
        return dict(raw=raw, prepared=prepared,
                    raw_violations=raw_violations,
                    executed_violations=int(executed_invalid.sum()),
                    clip_fraction=(raw_violations / max(coordinates, 1)
                                   if self.clip_actions else 0.0),
                    coordinates=coordinates,
                    clipped=bool(self.clip_actions))

    def commands_for_execution(self, normalized_prefix):
        return self.prepare_commands(normalized_prefix)["prepared"]

    def features(self, env):
        angle = float(env.block.angle)
        values = np.array([env.block.position.x / 512, env.block.position.y / 512,
                           0.2 * np.sin(angle), 0.2 * np.cos(angle)], dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite physical feature")
        return values

    def signature(self, env):
        values = np.array([*env.agent.position, *env.agent.velocity,
                           *env.block.position, *env.block.velocity,
                           env.block.angle, env.block.angular_velocity],
                          dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite replay signature")
        return values

    def replay(self, scene, history):
        """Replay a history; returns (env, terminal, truncated).

        A history ending exactly at termination replays cleanly with
        terminal=True; histories continuing past termination are rejected.
        Never step an env whose replayed state is terminal OR truncated:
        use absorbing-state features instead.
        """
        env = self.new_env(image=False)
        self.reset(env, scene)
        history = list(history)
        terminal, truncated = False, False
        for position, action in enumerate(history):
            _, _, terminated, trunc, _ = self.raw_step(env, np.asarray(action, dtype=np.float64))
            terminal = bool(terminated)
            truncated = bool(trunc or truncated)
            if terminal and position < len(history) - 1:
                env.close()
                raise ValueError("history continues after episode termination")
        return env, terminal, truncated

    def execute_from_history(self, scene, history, normalized_prefix):
        commands = self.commands_for_execution(normalized_prefix)
        env, terminal, truncated = self.replay(scene, history)
        if terminal or truncated:
            frozen = self.features(env)
            env.close()
            return np.concatenate([frozen] * len(commands))
        features = []
        finished = False
        try:
            for action in commands:
                if not finished:
                    _, _, terminated, truncated, _ = self.raw_step(env, action)
                    finished = bool(terminated or truncated)
                features.append(self.features(env))
        finally:
            env.close()
        return np.concatenate(features)

    def check_replay(self, scene, history, signature, tol=1e-6):
        expected = np.asarray(signature, dtype=np.float64)
        if not np.isfinite(expected).all():
            raise ValueError("recorded replay signature is nonfinite")
        env, _, _ = self.replay(scene, history)
        try:
            replayed = self.signature(env)
        finally:
            env.close()
        gap = float(np.abs(replayed - expected).max())
        if not np.isfinite(gap) or gap > tol:
            raise ValueError(f"replay signature mismatch (scene {scene}, "
                             f"gap {gap}, tol {tol})")
        return gap

    def encode_observation(self, image, agent_pos):
        image_t = torch.from_numpy(np.asarray(image)).unsqueeze(0).to(
            self.device, dtype=torch.float32)
        pos_n = normalize_data(np.asarray(agent_pos).reshape(1, -1), self.stats,
                               key="agent_pos")
        pos_t = torch.from_numpy(pos_n).to(self.device, dtype=torch.float32)
        cond = encode_condition(self.vision_encoder, image_t, pos_t)
        check_condition_dim(cond)
        return cond.detach()
