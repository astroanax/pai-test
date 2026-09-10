"""Train wrapper for the N-experiment: single-head regression on 16-d targets."""
import torch, numpy as np
from run_audit import TCN

torch.backends.cudnn.enabled = False
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def train_reg(Xf, Xc, Y, seed, epochs=25, bs=256, lr=3e-4, width=128):
    """TCN regression: both heads trained to predict Y (16-d) via Huber;
    we use the equivariant head (12) + invariant head (4) split: Y[:12] -> eq
    head, Y[12:16] -> inv head."""
    torch.manual_seed(seed)
    net = TCN(Xf.shape[2], Xc.shape[1], d_inv=2, d_eq=12).to(DEVICE)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    xf, xc = torch.tensor(Xf, device=DEVICE), torch.tensor(Xc, device=DEVICE)
    y = torch.tensor(Y, device=DEVICE)
    n = len(xf)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(n, device=DEVICE)
        for k in range(0, n, bs):
            s = perm[k:k+bs]
            pi, pe = net(xf[s], xc[s])
            # eq head predicts first 12 dims; inv head the last 4 (no events
            # in this experiment -> plain Huber on both)
            loss = torch.nn.functional.smooth_l1_loss(pe, y[s][:, :12]) + \
                   torch.nn.functional.smooth_l1_loss(pi, y[s][:, 12:14])
            opt.zero_grad(); loss.backward(); opt.step()
    net.eval()
    return net


@torch.no_grad()
def predict_reg(net, Xf, Xc):
    pi, pe = [], []
    for k in range(0, len(Xf), 512):
        a, b = net(torch.tensor(Xf[k:k+512], device=DEVICE),
                   torch.tensor(Xc[k:k+512], device=DEVICE))
        pi.append(a.cpu().numpy()); pe.append(b.cpu().numpy())
    P = np.concatenate(pe)
    Pi = np.concatenate(pi)
    return np.c_[P[:, :12], Pi]   # 14-d in target layout
