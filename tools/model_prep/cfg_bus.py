# Yellow school bus (Golden Route) -> 'bus' (game box 11.6 x 2.5 x 3.2). Not resprayed.
import numpy as np
TEX = 1536; TRIM_LUM = 0.3; DUAL_W = 0.09; LOWTOL = 0.012; FLATTEN = True
UNITS = [dict(name='bus', box=(11.6, 2.5, 3.2), extra=dict(busSeats=dict(z0=-4.8, z1=3.0, wheelZ=3.8)))]
CENTRE = {'bus': (0, 2.3, 0)}
def keep_cluster(q): return True
def wheel_unit(cx, zc): return 0
def is_front(cx): return cx < 0
def unit_of_tri(c): return 0
def is_glass(C, col, sat, lum, u): return (sat < 0.25) & (lum > 0.05) & (lum < 0.5) & (C[:, 1] > 1.9)
def fill_ok(c, u): return c[1] > 1.9
def paint_tris(u, C, col, FN, sat, lum, G): return np.zeros(len(C), bool)
def classify(u, C, col, FN, sat, lum, glass):
    x, y, z = C.T; ax = np.abs(x); G = np.full(len(C), 'body', dtype=object); zmax, zmin = z.max(), z.min()
    def setg(m, n): G[m & (G == 'body')] = n
    setg(glass, 'glass')
    setg((z > zmax - 0.35) & (y < 0.95), 'bumpF'); setg((z < zmin + 0.35) & (y < 0.95), 'bumpR')
    setg((z < zmin + 0.9) & (y > 0.8) & (col[:, 0] > col[:, 1] * 1.6) & (col[:, 0] > 0.45), 'tail')
    setg((z > zmax - 0.7) & (y > 0.6) & (y < 1.5) & (ax > 0.4) & (ax < 1.2) & ((lum > 0.7) | (col[:, 0] > col[:, 2] * 1.5)), 'head')
    setg((ax > 1.0) & (y > 1.3) & (y < 2.4) & (col[:, 0] > 0.45) & (col[:, 0] > col[:, 1] * 1.8) & (z < zmax - 1.5), 'brkStop')
    mir = (ax > 1.05) & (y > 1.4) & (z > zmax - 1.4) & (lum < 0.2)
    setg(mir & (x > 0), 'mirrorL'); setg(mir & (x < 0), 'mirrorR')
    return G
