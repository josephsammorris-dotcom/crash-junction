# White box truck -> 'boxtruck' (game van 6.6 x 2.35 x 3.1). The cab takes the vehicle's colour, the box stays white.
import numpy as np
TEX = 1536; TRIM_LUM = 0.3; DUAL_W = 0.09; LOWTOL = 0.045; FLATTEN = True
UNITS = [dict(name='boxtruck', box=(6.6, 2.35, 3.1))]
CENTRE = {'boxtruck': (0, 1.7, 2.0)}
def keep_cluster(q): return abs(q[:, 2].mean()) > 0.15 and (q[:, 0].max() - q[:, 0].min()) < 0.25
def wheel_unit(cx, zc): return 0
def is_front(cx): return cx < 0
def unit_of_tri(c): return 0
def is_glass(C, col, sat, lum, u): return (sat < 0.25) & (lum > 0.05) & (lum < 0.5) & (C[:, 1] > 1.35) & (C[:, 2] > 1.5)
def fill_ok(c, u): return c[1] > 1.35 and c[2] > 1.5
def paint_tris(u, C, col, FN, sat, lum, G): return (C[:, 2] > 1.0) & (G == 'body')
def classify(u, C, col, FN, sat, lum, glass):
    x, y, z = C.T; ax = np.abs(x); G = np.full(len(C), 'body', dtype=object); zmax, zmin = z.max(), z.min()
    def setg(m, n): G[m & (G == 'body')] = n
    setg(glass, 'glass')
    setg((z > zmax - 0.3) & (y < 0.7), 'bumpF'); setg((z < zmin + 0.35) & (y < 0.75) & (y > 0.3), 'bumpR')
    setg((z < zmin + 0.4) & (y < 1.25) & (y > 0.7) & (col[:, 0] > col[:, 2] * 1.4), 'tail')
    setg((z > zmax - 0.3) & (y > 0.6) & (y < 1.4) & (ax > 0.4) & (col[:, 0] > col[:, 2] * 1.3), 'head')
    return G
