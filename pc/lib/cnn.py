"""Checker detection by a small convolutional network on the pixels (needs PyTorch).

Per slot a 48 x 48 top-view crop (64 mm square: the slot and its neighbours) goes in,
with the same crop of the empty-board photo when there is one, plus the ~17 numbers
learned.features computes for that slot (they join the network before its last layers,
so it starts from what is already known and can still find more in the pixels); out come
empty / robot / human. A point (column) gets one colour: the sum over its checkers of
how sure each one is of robot vs human. At most 15 per colour.

    import cnn
    det = cnn.Detector.load()                      # detector_cnn.pt next to the scripts
    ck = det.find(img, c, L, empty=None)           # [(x, y, side)]
"""
from pathlib import Path
from files import data

import cv2
import numpy as np

import learned

PATH = data("detector_cnn.pt")   # <PC code>/data/
SIZE, SPAN = 48, 64.0      # crop pixels, crop mm


def available():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        return False


def crops(img, c, L):
    """(N, SIZE, SIZE) float32 grey crops, each divided by its bright (glass) level."""
    S = learned.slots(L)
    xy = np.array([q for _, _, q in S], float)
    t = (np.arange(SIZE) + 0.5) / SIZE * SPAN - SPAN / 2
    gx, gy = np.meshgrid(t, t)
    X = xy[:, 0, None, None] + gx[None]
    Y = xy[:, 1, None, None] + gy[None]
    u, v = c.mm_to_px(X.reshape(-1), Y.reshape(-1))
    u = np.asarray(u, np.float32).reshape(len(S) * SIZE, SIZE)
    v = np.asarray(v, np.float32).reshape(len(S) * SIZE, SIZE)
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    out = cv2.remap(g, u, v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).astype(np.float32)
    out = out.reshape(len(S), SIZE, SIZE)
    level = np.percentile(out.reshape(len(S), -1), 90, axis=1)[:, None, None]
    return out / np.maximum(level, 1.0)


def inputs(img, c, L, empty=None):
    """(images (N, 2, SIZE, SIZE), extra (N, F)): crop, empty crop (or zeros); learned.features."""
    a = crops(img, c, L)
    b = crops(empty, c, L) if empty is not None else np.zeros_like(a)
    extra = learned.features(img, c, L, empty).astype(np.float32)
    return np.stack([a, b], 1).astype(np.float32), extra


def _net(n_extra):
    import torch.nn as nn

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.f = nn.Sequential(
                nn.Conv2d(2, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),     # 24
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 12
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 6
                nn.Flatten())
            self.h = nn.Sequential(nn.Linear(64 * 36 + n_extra, 64), nn.ReLU(), nn.Dropout(0.2), nn.Linear(64, 3))

        def forward(self, x, e):
            import torch
            return self.h(torch.cat([self.f(x), e], 1))
    return Net()


class Detector:
    def __init__(self, net=None, mu=None, sd=None, cols=None):
        self.net, self.mu, self.sd, self.cols = net, mu, sd, cols   # cols: which feature columns it uses

    @classmethod
    def train(cls, samples, epochs=30, seed=0, log=None, pixels_only=False, patience=3):
        """samples: [(images, extra, labels)] per photo (labels from learned.labels).
        pixels_only: a control without the computed features (only where the slot is and
        whether there is an empty photo), to see what the features add.
        Early stopping: the last 15% of the photos are kept apart; after every epoch the
        loss on them is checked, the best weights are kept, and training stops after
        `patience` epochs without a better one (or after `epochs`)."""
        import copy

        import torch
        torch.manual_seed(seed)
        n = len(samples)
        k = n - max(1, int(round(n * 0.15))) if n >= 6 else n
        samples, check = samples[:k], samples[k:]
        nf = samples[0][1].shape[1]
        cols = [nf - 4, nf - 3, nf - 2, nf - 1] if pixels_only else list(range(nf))
        X = torch.from_numpy(np.concatenate([s[0] for s in samples]))
        E = torch.from_numpy(np.concatenate([s[1] for s in samples])[:, cols])
        Y = torch.from_numpy(np.concatenate([s[2] for s in samples])).long()
        mu, sd = E.mean(0), E.std(0).clamp(min=0.05)
        mu[-2:], sd[-2:] = 0.0, 1.0     # empty-photo ratio and flag: switched by the augmentation below
        net = _net(E.shape[1])
        opt = torch.optim.Adam(net.parameters(), 1e-3, weight_decay=1e-4)
        cnt = torch.bincount(Y, minlength=3).float().clamp(min=1)
        lossf = torch.nn.CrossEntropyLoss(weight=(cnt.sum() / cnt) / 3)
        g = torch.Generator().manual_seed(seed)
        if check:
            Xc = torch.from_numpy(np.concatenate([s[0] for s in check]))
            Ec = (torch.from_numpy(np.concatenate([s[1] for s in check])[:, cols]) - mu) / sd
            Yc = torch.from_numpy(np.concatenate([s[2] for s in check])).long()
        best, keep, bad = float("inf"), None, 0
        for ep in range(epochs):
            net.train()
            perm = torch.randperm(len(Y), generator=g)
            tot = 0.0
            for k in range(0, len(Y), 256):
                i = perm[k:k + 256]
                x, e = X[i].clone(), E[i].clone()
                # small shifts (sloppy placement / calibration drift), brightness, sometimes no empty photo
                dx, dy = (int(v) for v in torch.randint(-4, 5, (2,), generator=g))
                x = torch.roll(x, (dy, dx), (2, 3))
                x[:, 0] *= 0.85 + 0.3 * torch.rand(len(i), 1, 1, generator=g)
                drop = torch.rand(len(i), generator=g) < 0.5
                x[drop, 1] = 0
                e[drop, -2] = 1.0     # learned.features without an empty photo: ratio 1, flag 1
                e[drop, -1] = 1.0
                opt.zero_grad()
                loss = lossf(net(x, (e - mu) / sd), Y[i])
                loss.backward()
                opt.step()
                tot += float(loss.detach()) * len(i)
            msg = f"      epoch {ep + 1}  loss {tot / len(Y):.3f}"
            if check:
                net.eval()
                with torch.no_grad():
                    lv = sum(float(lossf(net(Xc[j:j + 1024], Ec[j:j + 1024]), Yc[j:j + 1024])) * len(Yc[j:j + 1024])
                             for j in range(0, len(Yc), 1024)) / len(Yc)
                msg += f"  on photos it does not learn from {lv:.3f}"
                if lv < best - 1e-4:
                    best, keep, bad = lv, copy.deepcopy(net.state_dict()), 0
                else:
                    bad += 1
            if log:
                log(msg)
            if check and bad >= patience:
                break
        if keep is not None:
            net.load_state_dict(keep)
        net.eval()
        d = cls(net, mu, sd, cols)
        d.epochs = ep + 1
        return d

    def probs(self, x, e):
        import torch
        with torch.no_grad():
            e = (torch.from_numpy(e[:, self.cols]) - self.mu) / self.sd
            return torch.softmax(self.net(torch.from_numpy(x), e), 1).numpy()

    def find_x(self, x, e, L):
        P = self.probs(x, e)
        S = learned.slots(L)
        present = P[:, 0] < 0.5
        out = []
        for p in range(1, 25):
            idx = [k for k, (w, _, _) in enumerate(S) if w == p and present[k]]
            if not idx:
                continue
            vote = float(np.sum(np.log(P[idx, 1] + 1e-6) - np.log(P[idx, 2] + 1e-6)))
            side = "robot" if vote > 0 else "human"
            out += [(1 - P[k, 0], *S[k][2], side) for k in idx]
        for k, (w, _, xy) in enumerate(S):
            if not isinstance(w, int) and present[k]:
                out.append((1 - P[k, 0], *xy, w.split("_")[1]))
        for s in ("robot", "human"):   # never more than 15 of a colour: drop the least sure
            mine = sorted([o for o in out if o[3] == s])
            while len(mine) > 15:
                out.remove(mine.pop(0))
        return [(float(x), float(y), s) for _, x, y, s in out]

    def find(self, img, c, L, empty=None):
        return self.find_x(*inputs(img, c, L, empty), L)

    def board(self, x, e, L):
        import vision
        return learned.full(vision.summarize(self.find_x(x, e, L), L))

    def save(self, path=PATH):
        import torch
        torch.save({"state": self.net.state_dict(), "mu": self.mu, "sd": self.sd, "cols": self.cols}, str(path))

    @classmethod
    def load(cls, path=PATH):
        import torch
        d = torch.load(str(path), map_location="cpu")
        net = _net(len(d["mu"]))
        net.load_state_dict(d["state"])
        net.eval()
        return cls(net, d["mu"], d["sd"], d.get("cols") or list(range(len(d["mu"]))))
