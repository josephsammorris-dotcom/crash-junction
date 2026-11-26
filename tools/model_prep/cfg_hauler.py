# Red highway hauler: one model of a tractor and a trailer -> 'hauler_tractor' (6.2 x 2.6 x 3.6) and 'hauler_trailer' (9.4 x 2.6 x 4.1)
import numpy as np
TEX = 1280; TRIM_LUM = 0.12; DUAL_W = 0.085; LOWTOL = 0.012; FLATTEN = True
XT, YT, XR = -0.165, -0.135, 0.12      # trailer front face, trailer floor, end of the tractor's frame (model units)
UNITS = [dict(name='hauler_tractor', box=(6.2, 2.6, 3.6), uniform_h=True, anchor_front=True, hfit=(-0.299, 0.175), extra=dict(rigSeats=dict(z=0.1, y=1.95))),
         dict(name='hauler_trailer', box=(9.4, 2.6, 4.1))]
CENTRE = {'hauler_tractor': (0, 2.6, 0.2), 'hauler_trailer': (0, 2.0, 0)}
def keep_cluster(q): return True
def wheel_unit(cx, zc): return 1 if cx > 0.3 else 0
def is_front(cx): return cx < -0.5
def unit_of_tri(c): return 1 if (c[0] > XR or (c[0] > XT and c[1] > YT)) else 0
def is_glass(C, col, sat, lum, u): return (sat < 0.25) & (lum > 0.05) & (lum < 0.5) & (C[:, 1] > 2.0) & (C[:, 2] < C[:, 2].max() - 1.2) if u == 'hauler_tractor' else np.zeros(len(C), bool)
def fill_ok(c, u): return u == 'hauler_tractor' and c[1] > 2.0
def paint_tris(u, C, col, FN, sat, lum, G):
    if u != 'hauler_tractor': return np.zeros(len(C), bool)
    return (G == 'body') & (col[:, 0] > col[:, 1] * 1.5) & (col[:, 0] > 0.3)
def classify(u, C, col, FN, sat, lum, glass):
    x, y, z = C.T; ax = np.abs(x); G = np.full(len(C), 'body', dtype=object); zmax, zmin = z.max(), z.min()
    def setg(m, n): G[m & (G == 'body')] = n
    setg(glass, 'glass')
    if u == 'hauler_tractor':
        setg((z > zmax - 0.4) & (y < 1.0), 'bumpF')
        setg((z > zmax - 0.7) & (y > 0.9) & (y < 1.8) & (ax > 0.4) & (ax < 1.1) & ((lum > 0.7) | (col[:, 0] > col[:, 2] * 1.5) & (col[:, 1] > 0.45)), 'head')
        setg((ax > 0.8) & (y > 3.1) & (lum > 0.35) & (sat < 0.2), 'brkStack')
    else:
        setg((z < zmin + 0.4) & (y < 1.6), 'bumpR')
        setg((z < zmin + 0.5) & (y < 1.8) & (col[:, 0] > col[:, 1] * 1.6) & (col[:, 0] > 0.4), 'tail')
    return G
