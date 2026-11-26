# Silver retro hatchback coupe -> 'fastback' (game car box 4.4 x 1.72 x 1.36)
import numpy as np
L, W, H = 4.4, 1.72, 1.36
TEX = 1536; WHEEL_INNER = 0.42; TRIM_LUM = 0.35
CUTS = [((0, 0, 0.88), (0, 0, 1)), ((0, 0, -0.78), (0, 0, 1)),            # door shut lines (front, rear)
        ((0, 0.95, 0), (0, 1, 0)), ((0, 0.27, 0), (0, 1, 0)),             # belt line, sill
        ((0.64, 0, 0), (1, 0, 0)), ((-0.64, 0, 0), (1, 0, 0)),            # bonnet and hatch sides
        ((0, 0, -0.95), (0, 0, 1)), ((0, 0.62, 0), (0, 1, 0)),            # hatch front edge (at the roof), hatch lower edge
        ((0, 0, 2.05), (0, 0, 1)), ((0, 0, -2.05), (0, 0, 1)), ((0, 0.57, 0), (0, 1, 0))]   # bumpers
PIVOTS = {'doorL': (0.86, 0.6, 0.88), 'doorR': (-0.86, 0.6, 0.88), 'bonnet': (0, 0.93, 0.88), 'tailgate': (0, 1.3, -0.95)}
OUTDIR = {'doorL': (1, 0, 0), 'doorR': (-1, 0, 0), 'bonnet': (0, 1, 0), 'tailgate': (0, 0.5, -1)}
PANEL_INFO = {'doorL': dict(z0=-0.78, y0=0.27, y1=1.0), 'doorR': dict(z0=-0.78, y0=0.27, y1=1.0)}
PAINT = ['body', 'doorL', 'doorR', 'bonnet', 'tailgate']
CAB = dict(belt=0.9, roof=1.33, zf0=0.7, zf1=0.4, zr0=-1.85, zr1=-1.0, wb=0.8, wt=0.66)
def classify(C, col, FN, sat, lum, P, I):
    x, y, z = C.T; ax = np.abs(x); G = np.full(len(I), 'body', dtype=object)
    glass = (sat < 0.25) & (lum > 0.05) & (lum < 0.5)
    def setg(m, n): G[m & (G == 'body')] = n
    setg((z > 2.05) & (y < 0.57), 'bumpF'); setg((z < -2.05) & (y < 0.57), 'bumpR')
    mir = (ax > 0.78) & (y > 0.85) & (y < 1.12) & (z > 0.3) & (z < 0.8)
    setg(mir & (x > 0), 'mirrorL'); setg(mir & (x < 0), 'mirrorR')
    setg((z < -2.0) & (y > 0.55) & (y < 0.95) & (ax > 0.45) & (col[:, 0] > col[:, 1] * 1.4), 'tail')
    setg((z > 1.95) & (y > 0.5) & (y < 0.85) & (ax > 0.35) & (col[:, 0] > col[:, 2] * 1.3), 'head')
    hat = (z < -0.95) & (y > 0.62) & (ax < 0.64) & (FN[:, 2] < -0.2)
    setg(hat & glass, 'tailgate_glass'); setg(hat, 'tailgate')
    setg((z > 0.88) & (z < 2.05) & (y > 0.62) & (ax < 0.64) & (FN[:, 1] > 0.45) & ~glass, 'bonnet')
    for s, sn in ((1, 'L'), (-1, 'R')):
        side = (np.sign(x) == s) & (ax > 0.55) & (np.abs(FN[:, 0]) > 0.3); inz = (z > -0.78) & (z < 0.88)
        setg(side & inz & (y > 0.27) & (y < 0.95), 'door' + sn); setg(side & inz & (y >= 0.95) & (y < 1.36) & glass, 'door' + sn + '_glass')
    setg(glass & (y > 0.88) & (FN[:, 1] < 0.8), 'glass')
    return G
FLATTEN = True
BELT_FILL = 0.9
