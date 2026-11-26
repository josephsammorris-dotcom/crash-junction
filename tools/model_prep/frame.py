# Shared: put a Meshy vehicle (front = -x, up = +y, left = +z) into the game frame (x left, y up from ground, z forward)
import numpy as np
def fit(H, L, W, Hh, wbody=None):
    """scales so length = L, height = Hh and the body width (ignoring mirrors) = W"""
    mn, mx = H.min(0), H.max(0)
    sz = L / (mx[0] - mn[0]); sy = Hh / (mx[1] - mn[1])
    wb = wbody if wbody else 2 * np.percentile(np.abs(H[:, 2]), 99.5)
    sx = W / wb
    return dict(mn=mn, mx=mx, s=(sx, sy, sz))
def to_game(P, f):
    sx, sy, sz = f['s']; mn, mx = f['mn'], f['mx']
    return np.stack([P[:, 2] * sx, (P[:, 1] - mn[1]) * sy, -(P[:, 0] - (mn[0] + mx[0]) / 2) * sz], 1)
def nrm_game(N, f):
    sx, sy, sz = f['s']; n = np.stack([N[:, 2] / sx, N[:, 1] / sy, -N[:, 0] / sz], 1); return n / np.linalg.norm(n, axis=1, keepdims=True)
