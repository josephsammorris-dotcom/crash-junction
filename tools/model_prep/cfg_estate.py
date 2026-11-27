# Dusty red five-door estate -> 'estate' (game car box 4.95 x 1.86 x 1.52)
import numpy as np
L, W, H = 4.95, 1.86, 1.52
TEX = 1536; WHEEL_INNER = 0.45; TRIM_LUM = 0.3; BELT_FILL = 1.0
ZF, ZB, ZR = 0.82, -0.22, -1.12          # front door front edge, shut line between doors, rear door rear edge
CUTS = [((0, 0, ZF), (0, 0, 1)), ((0, 0, ZB), (0, 0, 1)), ((0, 0, ZR), (0, 0, 1)),     # door shut lines
        ((0, 0.97, 0), (0, 1, 0)), ((0, 0.27, 0), (0, 1, 0)),                          # belt line, sill
        ((0.62, 0, 0), (1, 0, 0)), ((-0.62, 0, 0), (1, 0, 0)),                         # bonnet sides
        ((0, 0, 1.12), (0, 0, 1)),                                                     # bonnet rear edge (cowl)
        ((0.76, 0, 0), (1, 0, 0)), ((-0.76, 0, 0), (1, 0, 0)),                         # tailgate sides
        ((0, 0, -1.88), (0, 0, 1)), ((0, 0.56, 0), (0, 1, 0)),                         # tailgate: front edge at the roof, bottom edge
        ((0, 0, 1.98), (0, 0, 1)), ((0, 0, -2.15), (0, 0, 1)), ((0, 0.56, 0), (0, 1, 0))]   # bumpers
PIVOTS = {'doorL': (0.9, 0.6, ZF), 'doorR': (-0.9, 0.6, ZF), 'doorL2': (0.9, 0.6, ZB), 'doorR2': (-0.9, 0.6, ZB),
          'bonnet': (0, 0.97, 1.12), 'tailgate': (0, 1.39, -1.88)}
OUTDIR = {'doorL': (1, 0, 0), 'doorR': (-1, 0, 0), 'doorL2': (1, 0, 0), 'doorR2': (-1, 0, 0), 'bonnet': (0, 1, 0), 'tailgate': (0, 0.5, -1)}
PANEL_INFO = {'doorL': dict(z0=ZB, y0=0.27, y1=0.97), 'doorR': dict(z0=ZB, y0=0.27, y1=0.97), 'doorL2': dict(z0=ZR, y0=0.27, y1=0.97), 'doorR2': dict(z0=ZR, y0=0.27, y1=0.97)}
PAINT = ['body', 'doorL', 'doorR', 'doorL2', 'doorR2', 'bonnet', 'tailgate']
CAB = dict(belt=0.97, roof=1.42, zf0=0.86, zf1=0.3, zr0=-2.15, zr1=-2.05, wb=0.86, wt=0.8)
CENTRE = (0, 1.0, -0.3)
WHEEL = {'rear': (-1.27, 0.31, 0.44), 'front': (1.47, 0.31, 0.44)}   # arch circle (z, y, radius) kept out of the doors
def RECOLOUR(C, col, FN, G):                        # the baked-in dark licence-plate recess on the tailgate
    x, y, z = C.T
    return (np.abs(x) < 0.36) & (y > 0.5) & (y < 0.9) & (z < -2.2) & (np.asarray(G) == 'tailgate') & (col.mean(1) < 0.3)
def classify(C, col, FN, sat, lum, P, I):
    x, y, z = C.T; ax = np.abs(x); G = np.full(len(I), 'body', dtype=object)
    glass = (sat < 0.25) & (lum > 0.05) & (lum < 0.5)
    def setg(m, n): G[m & (G == 'body')] = n
    setg((z > 1.98) & (y < 0.56), 'bumpF'); setg((z < -2.15) & (y < 0.56), 'bumpR')
    mir = (ax > 0.9) & (y > 0.88) & (y < 1.15) & (z > 0.5) & (z < 1.05)
    setg(mir & (x > 0), 'mirrorL'); setg(mir & (x < 0), 'mirrorR')
    setg((z < -2.25) & (ax > 0.74) & (y > 0.5) & (y < 0.88) & (FN[:, 2] < -0.2), 'tail')
    setg((z > 2.0) & (ax > 0.6) & (y > 0.58) & (y < 0.84) & (FN[:, 2] > 0.2), 'head')
    hat = (z < -1.88) & (y > 0.56) & (ax < 0.76) & (np.abs(FN[:, 0]) < 0.6) & (FN[:, 2] < 0.1)
    setg(hat & glass & (y > 0.95), 'tailgate_glass'); setg(hat, 'tailgate')
    setg((z > 1.12) & (z < 1.98) & (y > 0.6) & (ax < 0.62) & (FN[:, 1] > 0.4) & ~glass, 'bonnet')
    arch = ((z - WHEEL['rear'][0]) ** 2 + (y - WHEEL['rear'][1]) ** 2 < WHEEL['rear'][2] ** 2) | ((z - WHEEL['front'][0]) ** 2 + (y - WHEEL['front'][1]) ** 2 < WHEEL['front'][2] ** 2)
    for s, sn in ((1, 'L'), (-1, 'R')):
        side = (np.sign(x) == s) & (ax > 0.65) & (np.abs(FN[:, 0]) > 0.3) & ~arch
        for nm, z0, z1 in (('door' + sn, ZB, ZF), ('door' + sn + '2', ZR, ZB)):
            inz = (z > z0) & (z < z1)
            setg(side & inz & (y > 0.27) & (y < 0.97), nm); setg(side & inz & (y >= 0.97) & (y < 1.45) & glass, nm + '_glass')
    setg(glass & (y > 0.95) & (FN[:, 1] < 0.8), 'glass')
    return G
