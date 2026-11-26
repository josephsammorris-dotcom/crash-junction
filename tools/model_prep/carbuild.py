# Fit a Meshy car into Crash Junction and pack it as a CJM2 file, from a small per-car config (cfg_<name>.py).
#   python3 carbuild.py <cfg module> <full-res.glb> <simplified.glb> <out.cjm>
# The model is cut along the config's planes (Blender bisect) so doors, bonnet and boot have clean edges, its triangles
# are sorted into parts, panels get a backing skin and a dark cavity, and wheels are cut out one by one.
import sys, io, json, struct, importlib, numpy as np
sys.path.insert(0, '.')
import bpy, bmesh
from glbio import read
from frame import fit, to_game, nrm_game
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from PIL import Image, ImageDraw

cfg = importlib.import_module(sys.argv[1]); HI, LO, OUT = sys.argv[2:5]
hi = read(HI); lo = read(LO)
fr = fit(hi['pos'], cfg.L, cfg.W, cfg.H)
HP, HN = to_game(hi['pos'], fr), nrm_game(hi['nrm'], fr)
tree = cKDTree(HP)
P, UV, I = to_game(lo['pos'], fr), lo['uv'], lo['idx']
sx, sy, sz = fr['s']

# ---- wheels: find the four tyres in the full model and fit each one's circle
H0 = hi['pos']; y0 = H0[:, 1].min(); low = H0[H0[:, 1] < y0 + 0.012]
wheels = []
for ax in (-1, 1):
    for sd in (-1, 1):
        q = low[(np.sign(low[:, 0]) == ax) & (np.sign(low[:, 2]) == sd)]
        cx0, zc = q[:, 0].mean(), q[:, 2].mean()
        near = H0[(np.abs(H0[:, 2] - zc) < 0.1) & (np.abs(H0[:, 0] - cx0) < 0.3)]
        best = (-1, 0, 0)
        for R in np.arange(0.12, 0.24, 0.002):
            for cx in np.arange(cx0 - 0.03, cx0 + 0.031, 0.005):
                d = np.hypot(near[:, 0] - cx, near[:, 1] - (y0 + R)); s = (np.abs(d - R) < 0.003).sum()
                if s > best[0]: best = (s, R, cx)
        wheels.append(dict(R=best[1], c=np.array([best[2], y0 + best[1], zc]), front=ax < 0, side=sd))
        print('wheel', 'F' if ax < 0 else 'R', 'L' if sd > 0 else 'R', 'R=%.3f' % best[1], 'score', best[0])
WR = float(np.median([w['R'] for w in wheels]))   # all four tyres are the same size: refit each centre with the shared radius
for w in wheels:
    zc = w['c'][2]; near = H0[(np.abs(H0[:, 2] - zc) < 0.1) & (np.abs(H0[:, 0] - w['c'][0]) < 0.3)]; best = (-1, 0)
    for cx in np.arange(w['c'][0] - 0.04, w['c'][0] + 0.041, 0.004):
        sc = (np.abs(np.hypot(near[:, 0] - cx, near[:, 1] - (y0 + WR)) - WR) < 0.003).sum()
        if sc > best[0]: best = (sc, cx)
    w['R'] = WR; w['c'] = np.array([best[1], y0 + WR, zc])
cen = P[I].mean(1); wid = np.full(len(I), -1)
for k, w in enumerate(wheels):
    c = to_game(w['c'][None], fr)[0]; w['g'] = c
    d = np.hypot(cen[:, 2] - c[2], cen[:, 1] - c[1]) / 1.0   # game frame: x = side, y up, z fore-aft
    inb = (d < WR * sy + 0.012) & (np.abs(cen[:, 0]) > cfg.WHEEL_INNER) & (np.sign(cen[:, 0]) == w['side'])
    wid[inb & (wid < 0)] = k
print('wheel tris', [(wid == k).sum() for k in range(4)])

# ---- face data on the simplified mesh
img = Image.open(io.BytesIO(hi['tex'])).convert('RGB'); T = np.asarray(img).astype(np.float32) / 255
def samp(uv): h, w, _ = T.shape; x = np.clip((uv[:, 0] * w).astype(int), 0, w - 1); y = np.clip((uv[:, 1] * h).astype(int), 0, h - 1); return T[y, x]
def facedata(P_, UV_, I_):
    C = P_[I_].mean(1); col = samp(UV_[I_].mean(1))
    FN = np.cross(P_[I_[:, 1]] - P_[I_[:, 0]], P_[I_[:, 2]] - P_[I_[:, 0]]); FN /= np.linalg.norm(FN, axis=1, keepdims=True) + 1e-12
    mx, mn = col.max(1), col.min(1); sat = (mx - mn) / (mx + 1e-4); lum = col @ np.array([0.299, 0.587, 0.114])
    return C, col, FN, sat, lum

# ---- cut the body along the config's planes so panel edges are clean
def mesh_from(tris, P_, UV_):
    bm = bmesh.new(); vs = [bm.verts.new(p) for p in P_]; bm.verts.ensure_lookup_table(); uvl = bm.loops.layers.uv.new('uv')
    for t in tris:
        try: f = bm.faces.new([vs[t[0]], vs[t[1]], vs[t[2]]])
        except ValueError: continue
        for lp, k in zip(f.loops, t): lp[uvl].uv = (UV_[k][0], UV_[k][1])
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.0)
    return bm, uvl
def export(bm, uvl):
    bmesh.ops.triangulate(bm, faces=bm.faces[:]); key = {}; pos = []; uv = []; idx = []
    for f in bm.faces:
        tri = []
        for lp in f.loops:
            k = (lp.vert.index, round(lp[uvl].uv[0], 6), round(lp[uvl].uv[1], 6))
            if k not in key: key[k] = len(pos); pos.append(tuple(lp.vert.co)); uv.append(tuple(lp[uvl].uv))
            tri.append(key[k])
        idx.append(tri)
    return np.array(pos, np.float64), np.array(uv, np.float64), np.array(idx, np.int64)
bm, uvl = mesh_from(I[wid < 0], P, UV)
for co, no in cfg.CUTS:
    bmesh.ops.bisect_plane(bm, geom=bm.verts[:] + bm.edges[:] + bm.faces[:], plane_co=co, plane_no=no, dist=0.0005)
bm.verts.index_update(); BP, BU, BI = export(bm, uvl)
wmesh = []
for k in range(4):
    wb, wl = mesh_from(I[wid == k], P, UV); wb.verts.index_update(); wmesh.append(export(wb, wl))
print('cut body tris', len(BI), 'verts', len(BP))

# ---- classify the cut body into parts
P, UV, I = BP, BU, BI
C, col, FN, sat, lum = facedata(P, UV, I)
GNV = np.zeros_like(P); FA = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
for k in range(3): np.add.at(GNV, I[:, k], FA)
GNV /= np.linalg.norm(GNV, axis=1, keepdims=True) + 1e-12
# light reflections baked into the glass texture read as paint: fill any face above the belt line that is mostly ringed by glass
base = (sat < 0.25) & (lum > 0.05) & (lum < 0.5); ek = {}
for fi, t in enumerate(I):
    for a, b in ((0, 1), (1, 2), (2, 0)): ek.setdefault((min(t[a], t[b]), max(t[a], t[b])), []).append(fi)
nbr = [[] for _ in I]
for fs in ek.values():
    for a in fs:
        for b in fs:
            if a != b: nbr[a].append(b)
for _ in range(4):
    add_ = np.array([(not base[i]) and C[i, 1] > cfg.BELT_FILL and sum(base[j] for j in nbr[i]) >= 2 for i in range(len(I))])
    if not add_.any(): break
    base |= add_
lum = np.where(base, 0.3, lum); sat = np.where(base, 0.0, sat)
G = cfg.classify(C, col, FN, sat, lum, P, I)
# glass has a hidden inner skin: keep only the faces that look outwards from the middle of the cabin
isg = np.array([n == 'glass' or n.endswith('_glass') for n in G]); outw = (FN * (C - np.array(getattr(cfg, 'CENTRE', (0, 0.9, 0)))) ).sum(1) > 0
G[isg & ~outw] = 'drop'
names = sorted(set(G) - {'drop'}); print({n: int((G == n).sum()) for n in names})

def tnorm(X, gref=None):
    d, i = tree.query(X, k=12); w = 1 / (d + 1e-5); Nn = HN[i]
    if gref is not None: w = w * ((Nn * gref[:, None, :]).sum(2) > 0.2)
    n = (Nn * w[..., None]).sum(1); bad = np.linalg.norm(n, axis=1) < 1e-6
    if gref is not None: n[bad] = gref[bad]
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
blobs = []
def add(a, dt): b = np.ascontiguousarray(a, dtype=dt).tobytes(); blobs.append(b); return len(blobs) - 1
def pack(tris, nrm=None, offset=(0, 0, 0), flatten=False, flip=False, inset=0.0, orient=None):
    u, inv = np.unique(tris.ravel(), return_inverse=True)
    p = P[u].copy(); uv = UV[u].copy(); ix = inv.reshape(-1, 3)
    n = tnorm(p, GNV[u]) if nrm is None else nrm
    if flatten:  # glass: flatten each connected pane onto its best-fit plane so reflections run clean
        A = coo_matrix((np.ones(ix.size), (np.repeat(ix[:, 0], 3), ix.ravel())), shape=(len(u), len(u)))
        nc, lab = connected_components(A + A.T, directed=False)
        for k in range(nc):
            m = lab == k; q = p[m]
            if len(q) < 4: continue
            c0 = q.mean(0); _, _, vt = np.linalg.svd(q - c0); nn = vt[2]
            if nn @ n[m].mean(0) < 0: nn = -nn
            p[m] = q - np.outer((q - c0) @ nn, nn) - nn * inset; n[m] = nn
    if orient is not None:
        fn = np.cross(p[ix[:, 1]] - p[ix[:, 0]], p[ix[:, 2]] - p[ix[:, 0]]); bad = (fn @ orient) < 0; ix = ix.copy(); ix[bad] = ix[bad][:, ::-1]
    if flip: ix = ix[:, ::-1]; n = -n
    return dict(pos=add(p - np.array(offset), np.float32), nrm=add(np.clip(np.round(n * 127), -127, 127), np.int8), uv=add(uv, np.float32), idx=add(ix, np.uint16 if len(u) < 65536 else np.uint32), big=len(u) >= 65536, n=len(u))

PIV, OUTD = cfg.PIVOTS, cfg.OUTDIR
KIND = lambda nm: ('glass' if nm == 'glass' or nm.endswith('_glass') else 'panel' if nm in PIV else 'lamp' if nm in ('head', 'tail') else 'rigid' if nm.startswith(('bump', 'mirror')) else 'body')
parts = []
for nm in names:
    tri = I[G == nm]; kind = KIND(nm); owner = nm[:-6] if nm.endswith('_glass') else nm; off = PIV.get(owner, (0, 0, 0))
    extra = {}
    if kind == 'panel': extra = cfg.PANEL_INFO.get(nm, {})
    parts.append(dict(name=nm, kind=kind, owner=owner if owner in PIV else None, pivot=list(off), geo=pack(tri, offset=off, flatten=(kind == 'glass' and getattr(cfg, 'FLATTEN', True)), inset=0.004), tris=len(tri), **extra))
    od = np.array(OUTD[owner], float) if owner in OUTD else None
    if od is not None and (kind == 'panel' or (kind == 'glass' and owner in PIV)):   # backing skin that rides with the panel
        od = od / np.linalg.norm(od)
        fo = np.cross(P[tri[:, 1]] - P[tri[:, 0]], P[tri[:, 2]] - P[tri[:, 0]]) @ od > 0; t2 = tri[fo]
        if len(t2):
            u = np.unique(t2.ravel()); Psave = P; P = P.copy(); P[u] = Psave[u] - od * 0.02
            parts.append(dict(name=nm + '_inner', kind='inner', owner=owner, pivot=list(off), geo=pack(t2, nrm=np.tile(-od, (len(u), 1)), offset=off, flip=True), tris=len(t2))); P = Psave
    if kind == 'panel':   # the dark cavity left when the panel opens or tears away
        u = np.unique(tri.ravel()); c0 = P[u].mean(0); Psave = P; P = P.copy(); P[u] = c0 + (Psave[u] - c0) * 0.96 - od * 0.08
        parts.append(dict(name=nm + '_cavity', kind='cavity', owner=None, pivot=[0, 0, 0], geo=pack(tri, nrm=np.tile(-od, (len(u), 1)).astype(float), orient=-od), tris=len(tri))); P = Psave
wl = []
for k in range(4):
    c = wheels[k]['g']; wp, wu, wi = wmesh[k]
    gw = np.zeros_like(wp); fa = np.cross(wp[wi[:, 1]] - wp[wi[:, 0]], wp[wi[:, 2]] - wp[wi[:, 0]])
    for kk in range(3): np.add.at(gw, wi[:, kk], fa)
    gw /= np.linalg.norm(gw, axis=1, keepdims=True) + 1e-12; n = tnorm(wp, gw)
    wl.append(dict(x=float(c[0]), y=float(c[1]), z=float(c[2]), front=bool(wheels[k]['front']), side=int(wheels[k]['side']),
        geo=dict(pos=add(wp - c, np.float32), nrm=add(np.clip(np.round(n * 127), -127, 127), np.int8), uv=add(wu, np.float32), idx=add(wi, np.uint16), big=False, n=len(wp))))

# ---- paint mask by UV region (works for white and silver cars too): paintable panels, minus dark trim, glass, lamps
paint_groups = set(cfg.PAINT)
mw = 1024; M = Image.new('L', (mw, mw), 0); dr = ImageDraw.Draw(M)
for ti in np.where(np.isin(G, [n for n in names if n in paint_groups or n.split('_')[0] in paint_groups and not n.endswith('_glass')]))[0]:
    pts = [(UV[k][0] * mw, UV[k][1] * mw) for k in I[ti]]; dr.polygon(pts, fill=255)
Mk = np.asarray(M).astype(np.float32) / 255
big = np.asarray(img.resize((mw, mw), Image.BILINEAR)).astype(np.float32) / 255; bl = big @ np.array([0.299, 0.587, 0.114])
mask = np.clip((bl - cfg.TRIM_LUM) / 0.1, 0, 1) * Mk
if getattr(cfg, 'GLASS_BY_UV', True):
    gm = Image.new('L', (mw, mw), 0); gd = ImageDraw.Draw(gm)
    for ti in np.where(isg & outw)[0]: gd.polygon([(UV[k][0] * mw, UV[k][1] * mw) for k in I[ti]], fill=255)
    gloss = np.asarray(gm).astype(np.float32) / 255
else: gloss = np.zeros_like(mask)
ref_lum = float(np.median(bl[mask > 0.8])) if (mask > 0.8).sum() > 100 else 0.5
print('paint cover', (mask > 0.5).mean(), 'ref lum', ref_lum)
bc = io.BytesIO(); img.resize((cfg.TEX, cfg.TEX), Image.LANCZOS).save(bc, 'JPEG', quality=80); mk = io.BytesIO()
Image.fromarray(np.stack([mask * 255, gloss * 255, np.zeros_like(mask)], 2).astype(np.uint8)).resize((768, 768), Image.BILINEAR).save(mk, 'JPEG', quality=88)
Hd = dict(v=2, parts=[{k: v for k, v in p.items() if k != 'tris'} for p in parts], wheels=wl, r=float(WR * sy), refLum=ref_lum, cab=cfg.CAB,
          tex=add(np.frombuffer(bc.getvalue(), np.uint8), np.uint8), mask=add(np.frombuffer(mk.getvalue(), np.uint8), np.uint8))
offs = []; o = 0
for b in blobs: offs.append([o, len(b)]); o += (len(b) + 3) // 4 * 4
Hd['blobs'] = offs
js = json.dumps(Hd).encode(); js += b' ' * ((4 - len(js) % 4) % 4)
with open(OUT, 'wb') as fh:
    fh.write(b'CJM2'); fh.write(struct.pack('<I', len(js))); fh.write(js)
    for b in blobs: fh.write(b); fh.write(b'\0' * ((4 - len(b) % 4) % 4))
import os; print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB', [(p['name'], p['tris']) for p in parts])
