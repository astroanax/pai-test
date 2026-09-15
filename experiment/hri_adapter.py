import numpy as np
import torch

from core import HORIZON, ACTION_DIM, freeze, integrate


EXPECTED_KEYS = ("vision_encoder", "noise_pred_net")
CONDITION_DIM = 514
ACTION_LOW = 0.0
ACTION_HIGH = 512.0
SUCCESS_COVERAGE = 0.95


def load_checkpoint(path, device):
    import resnet
    import unet
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
    return freeze(vision_encoder), freeze(noise_pred_net)


def encode_condition(vision_encoder, image, agent_pos):
    if not torch.isfinite(image).all():
        raise ValueError("nonfinite image tensor")
    with torch.no_grad():
        image_features = vision_encoder(image)
    features = torch.cat([image_features, agent_pos], dim=-1)
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


def teacher_rollout(noise_pred_net, noise, condition, steps):
    return integrate(make_field(noise_pred_net), noise, condition, 0.0, 1.0, steps)


def teacher_half_maps(noise_pred_net, noise, condition, steps=16):
    half = steps // 2
    mid = integrate(make_field(noise_pred_net), noise, condition, 0.0, 0.5, half)
    end = integrate(make_field(noise_pred_net), mid, condition, 0.5, 1.0, steps - half)
    return mid, end


def teacher_half_from(midpoint, noise_pred_net, condition, steps=16):
    second = steps - steps // 2
    return integrate(make_field(noise_pred_net), midpoint, condition, 0.5, 1.0, second)


def check_half_map_consistency(noise_pred_net, noise, condition, steps=16, tol=1e-4):
    full = teacher_rollout(noise_pred_net, noise, condition, steps)
    mid, end = teacher_half_maps(noise_pred_net, noise, condition, steps)
    gap = (full - end).abs().max().item()
    if not np.isfinite(gap) or gap > tol:
        raise ValueError("half map consistency check failed")
    return gap


def validate_stats(stats):
    """Check normalizer bounds, spans, and finiteness before any use."""
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


def check_batch_independence(noise_pred_net):
    noise_pred_net.eval()
    first = torch.randn(2, HORIZON, ACTION_DIM)
    cond = torch.randn(2, CONDITION_DIM)
    with torch.no_grad():
        out_single = noise_pred_net(first[:1], torch.zeros(1), global_cond=cond[:1])
        out_batch = noise_pred_net(first, torch.zeros(2), global_cond=cond)
    gap = (out_single - out_batch[:1]).abs().max().item()
    if not np.isfinite(gap) or gap > 1e-5:
        raise ValueError("teacher forward pass is not batch independent")
    return gap


class HRIAdapter:
    def __init__(self, vision_encoder, noise_pred_net, stats, device,
                 legacy=False, clip_actions=False):
        validate_stats(stats)
        self.vision_encoder = vision_encoder
        self.noise_pred_net = noise_pred_net
        self.stats = stats
        self.device = device
        self.legacy = legacy
        self.clip_actions = clip_actions

    def new_env(self, image):
        import pusht
        if image:
            return pusht.PushTImageEnv(legacy=self.legacy)
        return pusht.PushTEnv(legacy=self.legacy)

    def reset(self, env, scene):
        env.seed(scene)
        return env.reset()

    def raw_step(self, env, action):
        """Step with finite checking only: no clipping, no flag merging.

        Returns (obs, reward, terminated, truncated, done).
        """
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

    def step(self, env, action):
        if self.clip_actions:
            action = np.clip(action, ACTION_LOW, ACTION_HIGH)
        obs, reward, terminated, truncated, done = self.raw_step(env, action)
        return obs, reward, done, dict(terminated=terminated, truncated=truncated)

    def success(self, reward, coverage=None):
        """Success comes from the pinned environment's own definition: block
        goal coverage above SUCCESS_COVERAGE, signalled by `terminated`. A
        truncation (time limit, external cap) is NOT success."""
        return bool(reward >= SUCCESS_COVERAGE)

    def decode(self, normalized_prefix):
        physical = decode_actions(normalized_prefix, self.stats)
        if not np.isfinite(physical).all():
            raise ValueError("nonfinite decoded action")
        return [np.asarray(row, dtype=np.float64) for row in np.asarray(physical).reshape(-1, 2)]

    def decode_raw(self, normalized_prefix):
        return np.asarray(decode_actions(normalized_prefix, self.stats))

    def count_out_of_range(self, physical):
        values = np.asarray(physical, dtype=np.float64)
        return int(np.count_nonzero((values < ACTION_LOW) | (values > ACTION_HIGH))), int(values.size)

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
        env = self.new_env(image=False)
        self.reset(env, scene)
        for action in history:
            _, _, terminated, _, _ = self.raw_step(env, np.asarray(action, dtype=np.float64))
            if terminated:
                env.close()
                raise ValueError("history continues after episode termination")
        return env

    def execute_from_history(self, scene, history, normalized_prefix):
        env = self.replay(scene, history)
        features = []
        finished = False
        try:
            for action in self.decode(normalized_prefix):
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
        env = self.replay(scene, history)
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
        image_t = torch.from_numpy(np.asarray(image)).unsqueeze(0).to(self.device, dtype=torch.float32)
        pos_n = normalize_data(np.asarray(agent_pos).reshape(1, -1), self.stats, key="agent_pos")
        pos_t = torch.from_numpy(pos_n).to(self.device, dtype=torch.float32)
        cond = encode_condition(self.vision_encoder, image_t, pos_t)
        check_condition_dim(cond)
        return cond.detach()
