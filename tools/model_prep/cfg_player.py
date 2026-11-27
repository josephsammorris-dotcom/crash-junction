# Cream and carbon race coupe -> 'player' (game box 4.6 x 1.9 x 1.42, wing included)
import numpy as np
L, W, H = 4.6, 1.9, 1.42
TEX = 1536; WHEEL_INNER = 0.5; TRIM_LUM = 0.3; BELT_FILL = 0.95
ZF, ZR = 0.66, -0.64                       # door front edge, rear edge
CUTS = [((0, 0, ZF), (0, 0, 1)), ((0, 0, ZR), (0, 0, 1)),               # door shut lines
        ((0, 0.93, 0), (0, 1, 0)), ((0, 0.29, 0), (0, 1, 0)),            # belt line, sill
        ((0.8, 0, 0), (1, 0, 0)), ((-0.8, 0, 0), (1, 0, 0)),             # bonnet sides
        ((0, 0, 0.74), (0, 0, 1)), ((0, 0, 1.78), (0, 0, 1)),            # bonnet rear (cowl) and front edge
        ((0, 0, -1.92), (0, 0, 1)), ((0, 0.92, 0), (0, 1, 0)),           # wing
        ((0, 0, -2.12), (0, 0, 1)), ((0, 0.52, 0), (0, 1, 0))]           # rear bumper, front/rear bumper tops
PIVOTS = {'doorL': (0.9, 0.6, ZF), 'doorR': (-0.9, 0.6, ZF), 'bonnet': (0, 0.9, 0.74)}
OUTDIR = {'doorL': (1, 0, 0), 'doorR': (-1, 0, 0), 'bonnet': (0, 1, 0)}
PANEL_INFO = {'doorL': dict(z0=ZR, y0=0.29, y1=0.93), 'doorR': dict(z0=ZR, y0=0.29, y1=0.93)}
PAINT = ['body', 'doorL', 'doorR', 'bumpF', 'bumpR']
CAB = dict(belt=0.93, roof=1.4, zf0=0.7, zf1=0.09, zr0=-1.85, zr1=-1.3, wb=0.9, wt=0.72)
CENTRE = (0, 1.1, -0.4)
def classify(C, col, FN, sat, lum, P, I):
    x, y, z = C.T; ax = np.abs(x); G = np.full(len(I), 'body', dtype=object)
    glass = (sat < 0.25) & (lum > 0.05) & (lum < 0.5)
    def setg(m, n): G[m & (G == 'body')] = n
    setg((z < -1.92) & (y > 0.92) & (ax < 0.85), 'wing')
    setg((z < -2.1) & (ax > 0.25) & (ax < 0.85) & (y > 0.5) & (y < 0.9) & (lum < 0.5), 'tail')
    setg((z > 1.95) & (ax > 0.25) & (ax < 0.85) & (y > 0.45) & (y < 0.78) & (lum < 0.45), 'head')
    setg((z > 1.78) & (y < 0.52), 'bumpF'); setg((z < -2.12) & (y < 0.52), 'bumpR')
    mir = (ax > 0.85) & (y > 0.9) & (y < 1.12) & (z > 0.2) & (z < 0.6)
    setg(mir & (x > 0), 'mirrorL'); setg(mir & (x < 0), 'mirrorR')
    setg((z > 0.74) & (z < 1.78) & (y > 0.8) & (ax < 0.8) & (FN[:, 1] > 0.45) & (y < 1.0), 'bonnet')
    for s, sn in ((1, 'L'), (-1, 'R')):
        side = (np.sign(x) == s) & (ax > 0.65) & (np.abs(FN[:, 0]) > 0.3); inz = (z > ZR) & (z < ZF)
        setg(side & inz & (y > 0.29) & (y < 0.93), 'door' + sn); setg(side & inz & (y >= 0.93) & (y < 1.4) & glass, 'door' + sn + '_glass')
    setg(glass & (y > 0.93) & (FN[:, 1] < 0.8), 'glass')
    return G
