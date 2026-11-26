# Split the cut estate into game parts (doors, bonnet, tailgate, glass panes, lamps, bumpers, mirrors, wheels),
# give opening panels a dark cavity behind them, copy smooth normals from the full model, and pack a CJM2 file.
import sys, io, json, struct, numpy as np
sys.path.insert(0, '.')
from glbio import read
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from PIL import Image
CUT, PRE, HI, OUT = sys.argv[1:5]
c_ = np.load(CUT); pre = np.load(PRE)
P, UV, I = c_['pos'].astype(np.float64), c_['uv'].astype(np.float64), c_['idx']
tree = cKDTree(pre['hipos']); HN = pre['hinrm']
def tnorm(X, gref=None):
    # smooth normals from the full model, ignoring neighbours on a hidden inner skin (normals facing the other way)
    d, i = tree.query(X, k=12); w = 1 / (d + 1e-5); Nn = HN[i]
    if gref is not None:
        ok = (Nn * gref[:, None, :]).sum(2) > 0.2; w = w * ok
    n = (Nn * w[..., None]).sum(1); bad = np.linalg.norm(n, axis=1) < 1e-6
    if gref is not None: n[bad] = gref[bad]
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
hi = read(HI); img = Image.open(io.BytesIO(hi['tex'])).convert('RGB'); T = np.asarray(img).astype(np.float32) / 255
def samp(uv): h, w, _ = T.shape; x = np.clip((uv[:, 0] * w).astype(int), 0, w - 1); y = np.clip((uv[:, 1] * h).astype(int), 0, h - 1); return T[y, x]
C = P[I].mean(1); UC = UV[I].mean(1); col = samp(UC)
e1 = P[I[:, 1]] - P[I[:, 0]]; e2 = P[I[:, 2]] - P[I[:, 0]]; FN = np.cross(e1, e2); FN /= np.linalg.norm(FN, axis=1, keepdims=True) + 1e-12
mxc, mnc = col.max(1), col.min(1); sat = (mxc - mnc) / (mxc + 1e-4); lum = col @ np.array([0.299, 0.587, 0.114])
glass = (sat < 0.22) & (lum > 0.07) & (lum < 0.45)
GNV = np.zeros_like(P); FA = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
for k in range(3): np.add.at(GNV, I[:, k], FA)
GNV /= np.linalg.norm(GNV, axis=1, keepdims=True) + 1e-12
x, y, z = C[:, 0], C[:, 1], C[:, 2]; ax = np.abs(x)
G = np.full(len(I), 'body', dtype=object)
def setg(mask, name): G[mask & (G == 'body')] = name
setg((z > 2.15) & (y < 0.5), 'bumpF'); setg((z < -2.2) & (y < 0.52), 'bumpR')
setg((ax > 0.86) & (y > 0.8) & (y < 1.15) & (z > 0.4) & (z < 1.6) & (x > 0), 'mirrorL'); setg((ax > 0.86) & (y > 0.8) & (y < 1.15) & (z > 0.4) & (z < 1.6) & (x < 0), 'mirrorR')
setg((z < -2.16) & (ax > 0.5) & (y > 0.5) & (y < 1.0) & (col[:, 0] > col[:, 1] * 1.6) & (sat > 0.45) & (FN[:, 2] < -0.3), 'tail')
setg((z > 2.15) & (y > 0.5) & (y < 0.82) & (ax > 0.28) & (lum > 0.5) & (sat < 0.35) & (FN[:, 2] > 0.3), 'head')
tg = (z < -1.72) & (y > 0.55) & (ax < 0.71) & (FN[:, 2] < -0.2)
setg(tg & glass, 'tailgate_glass'); setg(tg, 'tailgate')
setg((z > 1.16) & (z < 2.32) & (y > 0.7) & (ax < 0.71) & (FN[:, 1] > 0.45) & ~glass, 'bonnet')
for s, sn in ((1, 'L'), (-1, 'R')):
    sidef = (np.sign(x) == s) & (ax > 0.55) & (np.abs(FN[:, 0]) > 0.3)
    for nm, z0, z1 in (('door' + sn, -0.06, 0.84), ('door' + sn + '2', -0.94, -0.06)):
        inz = (z > z0) & (z < z1)
        setg(sidef & inz & (y > 0.29) & (y < 0.9), nm)
        setg(sidef & inz & (y >= 0.9) & (y < 1.36) & glass, nm + '_glass')
setg(glass & (y > 0.88) & (FN[:, 1] < 0.8), 'glass')
names = sorted(set(G)); print({n: int((G == n).sum()) for n in names})
# pivots (hinges): doors at their front edge, bonnet at the windscreen, tailgate at the roof
PIV = {'doorL': (0.8, 0.6, 0.84), 'doorL2': (0.8, 0.6, -0.06), 'doorR': (-0.8, 0.6, 0.84), 'doorR2': (-0.8, 0.6, -0.06),
       'bonnet': (0, float(P[(np.abs(P[:, 2] - 1.12) < 0.03) & (np.abs(P[:, 0]) < 0.3), 1].max()), 1.12), 'tailgate': (0, 1.39, -1.74)}
blobs = []
def add(a, dt): b = np.ascontiguousarray(a, dtype=dt).tobytes(); blobs.append(b); return len(blobs) - 1
def pack(tris, nrm=None, offset=(0, 0, 0), flatten=False, flip=False, inset=0.0, orient=None):
    u, inv = np.unique(tris.ravel(), return_inverse=True)
    p = P[u].copy(); uv = UV[u].copy(); ix = inv.reshape(-1, 3)
    n = tnorm(p, GNV[u]) if nrm is None else nrm
    if flatten:  # glass: flatten each connected pane onto its best-fit plane, so reflections run clean
        A = coo_matrix((np.ones(ix.size), (np.repeat(ix[:, 0], 3), ix.ravel())), shape=(len(u), len(u)))
        nc, lab = connected_components(A + A.T, directed=False)
        for k in range(nc):
            m = lab == k; q = p[m]; c0 = q.mean(0); _, _, vt = np.linalg.svd(q - c0); nn = vt[2]
            if nn @ n[m].mean(0) < 0: nn = -nn
            p[m] = q - np.outer((q - c0) @ nn, nn) - nn * inset; n[m] = nn
    # generated meshes have some triangles wound the wrong way: point every face the way its normals point
    if orient is not None:
        fn = np.cross(p[ix[:, 1]] - p[ix[:, 0]], p[ix[:, 2]] - p[ix[:, 0]]); bad = (fn @ orient) < 0; ix = ix.copy(); ix[bad] = ix[bad][:, ::-1]
    if flip: ix = ix[:, ::-1]; n = -n
    return dict(pos=add(p - np.array(offset), np.float32), nrm=add(np.clip(np.round(n * 127), -127, 127), np.int8), uv=add(uv, np.float32), idx=add(ix, np.uint16 if len(u) < 65536 else np.uint32), big=len(u) >= 65536, n=len(u))
parts = []
KIND = lambda nm: ('glass' if nm == 'glass' or nm.endswith('_glass') else 'panel' if nm in PIV else 'lamp' if nm in ('head', 'tail') else 'rigid' if nm.startswith(('bump', 'mirror')) else 'body')
for nm in names:
    tri = I[G == nm]; kind = KIND(nm)
    owner = nm[:-6] if nm.endswith('_glass') else nm
    off = PIV.get(owner, (0, 0, 0))
    parts.append(dict(name=nm, kind=kind, owner=owner if owner in PIV else None, pivot=list(off), geo=pack(tri, offset=off, flatten=(kind == 'glass'), inset=0.004), tris=len(tri)))
    if kind in ('panel',) or (kind == 'glass' and owner in PIV):  # an inner skin that moves with it, so it isn't see-through from behind
        # the model already has some inner skin; only back the faces that look outwards, and push the backing inwards
        od = np.array([np.sign(P[np.unique(tri.ravel()), 0].mean()), 0, 0]) if owner.startswith('door') else np.array([0, 1.0, 0]) if owner == 'bonnet' else np.array([0, 0.3, -1.0])
        fo = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]]) @ od > 0
        t2 = tri[fo]
        if len(t2):
            u = np.unique(t2.ravel()); Pc = P.copy(); Pc[u] = P[u] - od / np.linalg.norm(od) * 0.02; Psave = P; P = Pc
            parts.append(dict(name=nm + '_inner', kind='inner', owner=owner, pivot=list(off), geo=pack(t2, nrm=np.tile(-od / np.linalg.norm(od), (len(u), 1)), offset=off, flip=True), tris=len(t2)))
            P = Psave
    if kind == 'panel':  # the dark cavity left behind when the panel opens or is torn away
        # pushed straight in (doors towards the middle, bonnet down, tailgate forward) and shrunk a little so no edge pokes out
        u, inv = np.unique(tri.ravel(), return_inverse=True)
        out_dir = np.array([np.sign(P[u, 0].mean()), 0, 0]) if nm.startswith('door') else np.array([0, 1.0, 0]) if nm == 'bonnet' else np.array([0, 0, -1.0])
        c0 = P[u].mean(0); Pc = P.copy(); Pc[u] = c0 + (P[u] - c0) * 0.96 - out_dir * 0.08
        Psave = P; P = Pc
        nn = np.tile(-out_dir, (len(u), 1)).astype(np.float64)
        parts.append(dict(name=nm + '_cavity', kind='cavity', owner=None, pivot=[0, 0, 0], geo=pack(tri, nrm=nn, flip=False, orient=-out_dir), tris=len(tri)))
        P = Psave
wheels = []
for k in range(4):
    c = c_['wheels'][k]; wp, wu, wi = c_[f'w{k}_pos'].astype(np.float64), c_[f'w{k}_uv'], c_[f'w{k}_idx']
    gw = np.zeros_like(wp); fa = np.cross(wp[wi[:, 1]] - wp[wi[:, 0]], wp[wi[:, 2]] - wp[wi[:, 0]])
    for kk in range(3): np.add.at(gw, wi[:, kk], fa)
    gw /= np.linalg.norm(gw, axis=1, keepdims=True) + 1e-12
    n = tnorm(wp, gw)
    wheels.append(dict(x=float(c[0]), y=float(c[1]), z=float(c[2]), front=bool(c[2] > 0), side=int(np.sign(c[0])),
        geo=dict(pos=add(wp - c, np.float32), nrm=add(np.clip(np.round(n * 127), -127, 127), np.int8), uv=add(wu, np.float32), idx=add(wi, np.uint16), big=False, n=len(wp))))
# paint / gloss mask
a = T; mx_, mn_ = a.max(2), a.min(2); st = (mx_ - mn_) / (mx_ + 1e-4)
hsv = np.asarray(img.convert('HSV')).astype(np.float32); hue = hsv[..., 0] / 255 * 360
ref_hue = float(np.median(hue[(st > 0.35) & (mx_ > 0.35)])); dh = np.minimum(np.abs(hue - ref_hue), 360 - np.abs(hue - ref_hue))
mask = np.clip((1 - dh / 22) * 1.6, 0, 1) * np.clip((st - 0.18) / 0.15, 0, 1) * np.clip((mx_ - 0.12) / 0.12, 0, 1)
lm = a @ np.array([0.299, 0.587, 0.114]); gloss = np.clip((0.2 - st) / 0.12, 0, 1) * np.clip((lm - 0.12) / 0.06, 0, 1) * np.clip((0.42 - lm) / 0.08, 0, 1)
ref_lum = float(np.median(lm[mask > 0.8]))
bc = io.BytesIO(); img.resize((1536, 1536), Image.LANCZOS).save(bc, 'JPEG', quality=80); mk = io.BytesIO()
Image.fromarray(np.stack([mask * 255, gloss * 255, np.zeros_like(mask)], 2).astype(np.uint8)).resize((768, 768), Image.BILINEAR).save(mk, 'JPEG', quality=88)
H = dict(v=2, parts=[{k: v for k, v in p.items() if k != 'tris'} for p in parts], wheels=wheels, r=float(c_['r']), refLum=ref_lum,
         tex=add(np.frombuffer(bc.getvalue(), np.uint8), np.uint8), mask=add(np.frombuffer(mk.getvalue(), np.uint8), np.uint8))
offs = []; o = 0
for b in blobs: offs.append([o, len(b)]); o += (len(b) + 3) // 4 * 4
H['blobs'] = offs
js = json.dumps(H).encode(); js += b' ' * ((4 - len(js) % 4) % 4)
with open(OUT, 'wb') as fh:
    fh.write(b'CJM2'); fh.write(struct.pack('<I', len(js))); fh.write(js)
    for b in blobs: fh.write(b); fh.write(b'\0' * ((4 - len(b) % 4) % 4))
import os; print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB', [(p['name'], p['tris']) for p in parts])
