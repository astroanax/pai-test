"""JEPA models J0-J4 (frozen ResNet18 visual encoder, leakage-free targets).

Context: mean of 4 current-frame embeddings (512) + proprio/action MLP +
  force branch (per interface) -> h (256).
Predictor: MLP(h) -> future visual embedding (512).
Heads: contact onset/offset (2 logits), future canonical wrench (6).
Loss: (1 - cos) + BCE(contact) + 0.5 * Huber(wrench, standardized).

J0: no force (zeros). J1: raw wrench. J2: SE(3)-canonical wrench.
J3: canonical + contact-phase gating: g = sigmoid(MLP([|f| stats, contact
  frac])) in (0,1); force_emb = MLP(canonical) * g.
J4: canonical + supervised target adapter: linear 6x6 map fit on adapter
  episodes of the held-out group (disjoint from eval), applied to canonical
  wrench before the force MLP.
"""
import torch
import torch.nn as nn

torch.backends.cudnn.enabled = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_frozen_encoder():
    import torchvision.models as M
    net = M.resnet18(weights=M.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval()
    return net.to(DEVICE)


class ForceBranch(nn.Module):
    """Small TCN over 30x6 wrench history -> 64-d. Gating for J3."""
    def __init__(self, din=6, gated=False, width=64):
        super().__init__()
        self.gated = gated
        self.tcn = nn.Sequential(
            nn.Conv1d(din, width, 5, padding=2), nn.GELU(),
            nn.Conv1d(width, width, 5, padding=4, dilation=2), nn.GELU(),
        )
        self.out = nn.Linear(width, 64)
        if gated:
            self.gate = nn.Sequential(nn.Linear(3, 16), nn.GELU(),
                                      nn.Linear(16, 1), nn.Sigmoid())

    def forward(self, w, phase=None):
        # w: (B,30,6); phase: (B,3) [|f|mean, |f|max, contact_frac]
        h = self.tcn(w.transpose(1, 2))[:, :, -1]
        e = self.out(h)
        if self.gated:
            g = self.gate(phase)
            e = e * g
        return e


class JEPA(nn.Module):
    def __init__(self, force_kind="J2", d_prop=32, width=256):
        super().__init__()
        self.force_kind = force_kind
        self.vis = nn.Linear(512, 256)
        self.prop = nn.Sequential(nn.Linear(d_prop, 128), nn.GELU(),
                                  nn.Linear(128, 128))
        if force_kind == "J0":
            self.force = None
            self.ctx = nn.Sequential(nn.Linear(256 + 128 + 64, width), nn.GELU(),
                                     nn.Linear(width, width))
            self.zero_emb = nn.Parameter(torch.zeros(64))
        else:
            self.force = ForceBranch(gated=(force_kind == "J3"))
            self.ctx = nn.Sequential(nn.Linear(256 + 128 + 64, width), nn.GELU(),
                                     nn.Linear(width, width))
        self.pred = nn.Sequential(nn.Linear(width, width), nn.GELU(),
                                  nn.Linear(width, 512))
        self.head_contact = nn.Linear(width, 2)
        self.head_wrench = nn.Linear(width, 6)

    def forward(self, vis4, prop, w, phase=None):
        # vis4: (B,4,512) frozen embeddings; prop: (B,d); w: (B,30,6)
        v = self.vis(vis4.mean(1))
        p = self.prop(prop)
        if self.force is None:
            f = self.zero_emb.expand(v.shape[0], -1)
        else:
            f = self.force(w, phase)
        h = self.ctx(torch.cat([v, p, f], 1))
        return h, self.pred(h), self.head_contact(h), self.head_wrench(h)

    def n_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
