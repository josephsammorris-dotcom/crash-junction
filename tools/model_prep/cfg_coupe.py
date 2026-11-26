# Vintage ivory notchback coupe -> 'coupe' (game car box 4.5 x 1.74 x 1.42)
import numpy as np
L, W, H = 4.5, 1.74, 1.42
TEX = 1536; WHEEL_INNER = 0.42; TRIM_LUM = 0.35
CUTS = [((0, 0, 0.9), (0, 0, 1)), ((0, 0, -0.7), (0, 0, 1)),
        ((0, 0.94, 0), (0, 1, 0)), ((0, 0.27, 0), (0, 1, 0)),
        ((0.62, 0, 0), (1, 0, 0)), ((-0.62, 0, 0), (1, 0, 0)),
        ((0, 0, -1.68), (0, 0, 1)), ((0, 0.62, 0), (0, 1, 0)),
        ((0, 0, 2.08), (0, 0, 1)), ((0, 0, -2.1), (0, 0, 1)), ((0, 0.57, 0), (0, 1, 0))]
PIVOTS = {'doorL': (0.87, 0.6, 0.9), 'doorR': (-0.87, 0.6, 0.9), 'bonnet': (0, 0.95, 0.9), 'tailgate': (0, 0.97, -1.68)}
OUTDIR = {'doorL': (1, 0, 0), 'doorR': (-1, 0, 0), 'bonnet': (0, 1, 0), 'tailgate': (0, 1, -0.3)}
PANEL_INFO = {'doorL': dict(z0=-0.7, y0=0.27, y1=1.0), 'doorR': dict(z0=-0.7, y0=0.27, y1=1.0)}
PAINT = ['body', 'doorL', 'doorR', 'bonnet', 'tailgate']
CAB = dict(belt=0.9, roof=1.4, zf0=0.72, zf1=0.4, zr0=-1.68, zr1=-1.3, wb=0.8, wt=0.66)
def classify(C, col, FN, sat, lum, P, I):
    x, y, z = C.T; ax = np.abs(x); G = np.full(len(I), 'body', dtype=object)
    glass = (sat < 0.25) & (lum > 0.05) & (lum < 0.5)
    def setg(m, n): G[m & (G == 'body')] = n
    setg((z > 2.08) & (y < 0.57), 'bumpF'); setg((z < -2.1) & (y < 0.57), 'bumpR')
    mir = (ax > 0.8) & (y > 0.85) & (y < 1.12) & (z > 0.3) & (z < 0.8)
    setg(mir & (x > 0), 'mirrorL'); setg(mir & (x < 0), 'mirrorR')
    setg((z < -2.0) & (y > 0.55) & (y < 0.95) & (ax > 0.45) & (col[:, 0] > col[:, 1] * 1.4), 'tail')
    setg((z > 1.95) & (y > 0.5) & (y < 0.85) & (ax > 0.35) & (col[:, 0] > col[:, 2] * 1.3), 'head')
    setg((z < -1.68) & (y > 0.62) & (ax < 0.62) & ((FN[:, 1] > 0.3) | (FN[:, 2] < -0.3)) & ~glass, 'tailgate')
    setg((z > 0.9) & (z < 2.08) & (y > 0.62) & (ax < 0.62) & (FN[:, 1] > 0.45) & ~glass, 'bonnet')
    for s, sn in ((1, 'L'), (-1, 'R')):
        side = (np.sign(x) == s) & (ax > 0.55) & (np.abs(FN[:, 0]) > 0.3); inz = (z > -0.7) & (z < 0.9)
        setg(side & inz & (y > 0.27) & (y < 0.94), 'door' + sn); setg(side & inz & (y >= 0.94) & (y < 1.4) & glass, 'door' + sn + '_glass')
    setg(glass & (y > 0.88) & (FN[:, 1] < 0.8), 'glass')
    return G
BELT_FILL = 0.9
