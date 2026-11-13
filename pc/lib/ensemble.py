"""Both learned detectors together: the features network (learned.py) and the pixel
network (cnn.py) each give, per slot, how sure they are there is a checker, and per
point (column) how sure it is the robot's. Average the two and decide. It helps only
where the two make different mistakes (train_detector.py shows whether they do).
"""
import numpy as np

import learned


def find(det_l, det_c, F, x, e, L):
    """[(x, y, side)] from learned.Detector det_l (features F) and cnn.Detector det_c (x, e)."""
    S = learned.slots(L)
    P = det_c.probs(x, e)
    pp = (det_l.presence.prob(F) + (1 - P[:, 0])) / 2
    present = pp > 0.5
    ql = det_l.colour_prob(F, present, L)
    out = []
    for p in range(1, 25):
        idx = [k for k, (w, _, _) in enumerate(S) if w == p and present[k]]
        if not idx:
            continue
        vote = float(np.mean(np.log(P[idx, 1] + 1e-6) - np.log(P[idx, 2] + 1e-6)))
        qc = 1 / (1 + np.exp(-vote))
        q = (ql.get(p, qc) + qc) / 2
        side = "robot" if q > 0.5 else "human"
        out += [(abs(pp[k] - 0.5) * 2, *S[k][2], side) for k in idx]
    for k, (w, _, xy) in enumerate(S):
        if not isinstance(w, int) and present[k]:
            out.append((abs(pp[k] - 0.5) * 2, *xy, w.split("_")[1]))
    for s in ("robot", "human"):
        mine = sorted([o for o in out if o[3] == s])
        while len(mine) > 15:
            out.remove(mine.pop(0))
    return [(float(a), float(b), s) for _, a, b, s in out]
