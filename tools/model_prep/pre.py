# numpy: game-frame arrays for the estate (simplified mesh + normals from the full model), wheels cut out one by one
import sys, numpy as np
sys.path.insert(0, '.')
from glbio import read
from scipy.spatial import cKDTree
HI, LO, OUT = sys.argv[1:4]
GL, GW, GH = 4.8, 1.86, 1.45
hi = read(HI); H = hi['pos']; mn, mx = H.min(0), H.max(0)
sz, sx, sy = GL / (mx[0] - mn[0]), GW / (mx[2] - mn[2]), GH / (mx[1] - mn[1])
def to_game(P): return np.stack([P[:, 2] * sx, (P[:, 1] - mn[1]) * sy, -(P[:, 0] - (mn[0] + mx[0]) / 2) * sz], 1)
def nrm_game(N): n = np.stack([N[:, 2] / sx, N[:, 1] / sy, -N[:, 0] / sz], 1); return n / np.linalg.norm(n, axis=1, keepdims=True)
lo = read(LO); P, UV, I = lo['pos'], lo['uv'], lo['idx']
y0 = mn[1]; low = H[H[:, 1] < y0 + 0.012]; R = 0.137
cen = P[I].mean(1); wid = np.full(len(I), -1); wheels = []
for ax in (-1, 1):
    for sd in (-1, 1):
        q = low[(np.sign(low[:, 0]) == ax) & (np.sign(low[:, 2]) == sd)]
        c = np.array([q[:, 0].mean(), y0 + R, q[:, 2].mean()])
        d = np.hypot(cen[:, 0] - c[0], cen[:, 1] - c[1])
        sel = (d < R + 0.006) & (np.abs(cen[:, 2]) > 0.2) & (np.sign(cen[:, 2]) == sd) & (wid < 0)
        wid[sel] = len(wheels); wheels.append(to_game(c[None])[0])
np.savez(OUT, pos=to_game(P), uv=UV, idx=I, wid=wid, wheels=np.array(wheels), hipos=to_game(H), hinrm=nrm_game(hi['nrm']), r=R * sy)
print('wheels', np.array(wheels).round(3), 'wheel tris', [(wid == k).sum() for k in range(4)], 'body tris', (wid < 0).sum())
