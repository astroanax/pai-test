import numpy as np
import torch

from core import freeze, integrate


EXPECTED_KEYS = ("vision_encoder", "noise_pred_net")
CONDITION_DIM = 514
HORIZON = 16
ACTION_DIM = 2


def load_checkpoint(path, device):
    import resnet
    import unet
    raw = torch.load(path, map_location=device, weights_only=True)
    if tuple(sorted(raw.keys())) != tuple(sorted(EXPECTED_KEYS)):
        raise ValueError("checkpoint keys do not match expected teacher keys")
    vision_encoder = resnet.get_resnet("resnet18")
    vision_encoder = resnet.replace_bn_with_gn(vision_encoder)
    noise_pred_net = unet.ConditionalUnet1D(
        input_dim=ACTION_DIM, global_cond_dim=CONDITION_DIM
    )
    vision_encoder.load_state_dict(raw["vision_encoder"], strict=True)
    noise_pred_net.load_state_dict(raw["noise_pred_net"], strict=True)
    return freeze(vision_encoder), freeze(noise_pred_net)


def encode_condition(vision_encoder, image, agent_pos):
    with torch.no_grad():
        image_features = vision_encoder(image)
    features = torch.cat([image_features, agent_pos], dim=-1)
    return features.flatten(start_dim=1)


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


def check_half_map_consistency(noise_pred_net, noise, condition, steps=16, tol=1e-4):
    full = teacher_rollout(noise_pred_net, noise, condition, steps)
    mid, end = teacher_half_maps(noise_pred_net, noise, condition, steps)
    gap = (full - end).abs().max().item()
    if gap > tol:
        raise ValueError("half map consistency check failed")
    return gap


def normalize_data(data, stats):
    ndata = (data - stats["min"]) / (stats["max"] - stats["min"])
    return ndata * 2 - 1


def unnormalize_data(ndata, stats):
    ndata = (ndata + 1) / 2
    return ndata * (stats["max"] - stats["min"]) + stats["min"]


def decode_actions(normalized, stats):
    return unnormalize_data(np.asarray(normalized), stats)


def check_batch_independence(noise_pred_net):
    noise_pred_net.eval()
    first = torch.randn(2, HORIZON, ACTION_DIM)
    cond = torch.randn(2, CONDITION_DIM)
    with torch.no_grad():
        out_single = noise_pred_net(first[:1], torch.zeros(1), global_cond=cond[:1])
        out_batch = noise_pred_net(first, torch.zeros(2), global_cond=cond)
    gap = (out_single - out_batch[:1]).abs().max().item()
    if gap > 1e-5:
        raise ValueError("teacher forward pass is not batch independent")
    return gap


class HRIAdapter:
    def __init__(self, vision_encoder, noise_pred_net, stats, device, legacy=False, clip_actions=False):
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

    def step(self, env, action):
        if self.clip_actions:
            action = np.clip(action, 0, 512)
        out = env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
            return obs, reward, done, info
        obs, reward, done, info = out
        return obs, reward, bool(done), info

    def decode(self, normalized_prefix):
        physical = decode_actions(normalized_prefix, self.stats["action"])
        return [np.asarray(row, dtype=np.float64) for row in np.asarray(physical).reshape(-1, 2)]

    def features(self, env):
        angle = float(env.block.angle)
        return np.array([env.block.position.x / 512, env.block.position.y / 512,
                         0.2 * np.sin(angle), 0.2 * np.cos(angle)], dtype=np.float64)

    def signature(self, env):
        return np.array([*env.agent.position, *env.agent.velocity,
                         *env.block.position, *env.block.velocity,
                         env.block.angle, env.block.angular_velocity], dtype=np.float64)

    def replay(self, scene, history):
        env = self.new_env(image=False)
        self.reset(env, scene)
        for action in history:
            _, _, done, _ = self.step(env, np.asarray(action, dtype=np.float64))
            if done:
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
                    _, _, finished, _ = self.step(env, action)
                features.append(self.features(env))
        finally:
            env.close()
        return np.concatenate(features)

    def check_replay(self, scene, history, signature, tol=1e-6):
        env = self.replay(scene, history)
        try:
            replayed = self.signature(env)
        finally:
            env.close()
        gap = float(np.abs(replayed - np.asarray(signature)).max())
        if gap > tol:
            raise ValueError("replay signature mismatch")
        return gap

    def encode_observation(self, image, agent_pos):
        image_t = torch.from_numpy(np.asarray(image)).unsqueeze(0).to(self.device, dtype=torch.float32)
        pos_n = normalize_data(np.asarray(agent_pos).reshape(1, -1), self.stats["agent_pos"])
        pos_t = torch.from_numpy(pos_n).to(self.device, dtype=torch.float32)
        cond = encode_condition(self.vision_encoder, image_t, pos_t)
        check_condition_dim(cond)
        return cond.detach()
