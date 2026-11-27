# Vintage green stepside pickup -> 'pickup' (game box 5.7 x 2.0 x 1.9)
import numpy as np
L, W, H = 5.7, 2.0, 1.9
TEX = 1536; WHEEL_INNER = 0.5; TRIM_LUM = 0.3; BELT_FILL = 1.18
CUTS = [((0, 0, 1.2), (0, 0, 1)), ((0, 0, -0.45), (0, 0, 1)),            # door front / rear shut lines
        ((0, 1.18, 0), (0, 1, 0)), ((0, 0.4, 0), (0, 1, 0)),             # belt line, sill
        ((0.78, 0, 0), (1, 0, 0)), ((-0.78, 0, 0), (1, 0, 0)),           # bonnet and tailgate sides
        ((0, 0, 1.3), (0, 0, 1)),                                        # bonnet rear edge (cowl)
        ((0, 0, -2.62), (0, 0, 1)), ((0, 0.68, 0), (0, 1, 0)),           # tailgate: inner face, bottom edge
        ((0, 0, 2.7), (0, 0, 1)), ((0, 0, -2.72), (0, 0, 1)), ((0, 0.62, 0), (0, 1, 0))]   # bumpers
PIVOTS = {'doorL': (0.97, 0.6, 1.2), 'doorR': (-0.97, 0.6, 1.2), 'bonnet': (0, 1.15, 1.3), 'tailgate': (0, 0.7, -2.7)}
OUTDIR = {'doorL': (1, 0, 0), 'doorR': (-1, 0, 0), 'bonnet': (0, 1, 0), 'tailgate': (0, 0.1, -1)}
PANEL_INFO = {'doorL': dict(z0=-0.45, y0=0.4, y1=1.3), 'doorR': dict(z0=-0.45, y0=0.4, y1=1.3)}
PAINT = ['body', 'doorL', 'doorR', 'bonnet']
CAB = dict(belt=1.15, roof=1.85, zf0=1.3, zf1=0.95, zr0=-0.5, zr1=-0.45, wb=0.85, wt=0.8, seats=[0.05], pickup=True)
CENTRE = (0, 1.4, 0.5)
def classify(C, col, FN, sat, lum, P, I):
    x, y, z = C.T; ax = np.abs(x); G = np.full(len(I), 'body', dtype=object)
    glass = (sat < 0.25) & (lum > 0.05) & (lum < 0.5) & (y > 1.18)
    def setg(m, n): G[m & (G == 'body')] = n
    setg((z > 2.7) & (y < 0.62), 'bumpF'); setg((z < -2.72) & (y < 0.62), 'bumpR')
    mir = (ax > 0.95) & (y > 1.2) & (z > 0.1) & (z < 1.2)
    setg(mir & (x > 0), 'mirrorL'); setg(mir & (x < 0), 'mirrorR')
    setg((z < -2.6) & (y > 0.7) & (y < 1.1) & (col[:, 0] > col[:, 1] * 1.4), 'tail')
    setg((z > 2.6) & (y > 0.6) & (y < 1.1) & (ax > 0.5) & ((col[:, 0] > col[:, 2] * 1.3) | (lum > 0.75)), 'head')
    setg((z < -2.62) & (y > 0.68) & (ax < 0.78) & (FN[:, 2] < -0.3), 'tailgate')
    setg((z > 1.3) & (z < 2.7) & (y > 0.9) & (ax < 0.78) & (FN[:, 1] > 0.45), 'bonnet')
    for s, sn in ((1, 'L'), (-1, 'R')):
        side = (np.sign(x) == s) & (ax > 0.6) & (np.abs(FN[:, 0]) > 0.3); inz = (z > -0.45) & (z < 1.2)
        setg(side & inz & (y > 0.4) & (y < 1.18), 'door' + sn); setg(side & inz & (y >= 1.18) & (y < 1.95) & glass, 'door' + sn + '_glass')
    setg(glass & (y > 1.18) & (FN[:, 1] < 0.8), 'glass')
    return G
