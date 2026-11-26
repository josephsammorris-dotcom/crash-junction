# Fit a Meshy truck, bus or rig into Crash Junction and pack one CJM2 file per unit (a rig = tractor + trailer).
#   python3 truckbuild.py <cfg module> <full-res.glb> <simplified.glb> <outdir>
# Like carbuild.py, but: any number of axles (dual wheels split into single tyres), wheels cut from the model frame and
# made round again after the body's non-uniform fit, identical wheels share one mesh, and optional units.
import sys, io, json, struct, importlib, numpy as np
sys.path.insert(0, '.')
from glbio import read
from frame import fit, to_game, nrm_game
from scipy.spatial import cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.cluster.hierarchy import fcluster, linkage
from PIL import Image, ImageDraw
import os

cfg = importlib.import_module(sys.argv[1]); HI, LO, OUTD = sys.argv[2:5]
hi = read(HI); lo = read(LO)
H0 = hi['pos']; Pm, UVm, Im = lo['pos'], lo['uv'], lo['idx']
img = Image.open(io.BytesIO(hi['tex'])).convert('RGB'); T = np.asarray(img).astype(np.float32) / 255
def samp(uv): h, w, _ = T.shape; x = np.clip((uv[:, 0] * w).astype(int), 0, w - 1); y = np.clip((uv[:, 1] * h).astype(int), 0, h - 1); return T[y, x]

# ---- wheels: contact clusters in the full model, dual tyres split, circle fitted to each
y0 = H0[:, 1].min(); tol = getattr(cfg, 'LOWTOL', 0.012)
low = H0[H0[:, 1] < y0 + tol]
lab = fcluster(linkage(low[:, [0, 2]], 'single'), 0.04, 'distance')
raw = []
for k in range(1, lab.max() + 1):
    q = low[lab == k]
    if len(q) < 6 or not cfg.keep_cluster(q): continue
    zmin, zmax = q[:, 2].min(), q[:, 2].max()
    if zmax - zmin > cfg.DUAL_W:   # two tyres side by side
        mid = (zmin + zmax) / 2; c1, c2 = (zmin + mid) / 2, (mid + zmax) / 2
        raw.append(dict(x=q[:, 0].mean(), z0=zmin, z1=mid, gy=q[:, 1].min(), role='out' if abs(c1) > abs(c2) else 'in', q=q))
        raw.append(dict(x=q[:, 0].mean(), z0=mid, z1=zmax, gy=q[:, 1].min(), role='out' if abs(c2) > abs(c1) else 'in', q=q))
    else: raw.append(dict(x=q[:, 0].mean(), z0=zmin, z1=zmax, gy=q[:, 1].min(), role='single', q=q))
wheels = []
for w in raw:
    zc = (w['z0'] + w['z1']) / 2; hw = (w['z1'] - w['z0']) / 2 + 0.01
    near = H0[(np.abs(H0[:, 2] - zc) < hw) & (np.abs(H0[:, 0] - w['x']) < 0.25)]; best = (-1, 0, 0)
    for R in np.arange(0.04, 0.14, 0.002):
        for cx in np.arange(w['x'] - 0.025, w['x'] + 0.0261, 0.005):
            s = (np.abs(np.hypot(near[:, 0] - cx, near[:, 1] - (w['gy'] + R)) - R) < 0.0025).sum()
            if s > best[0]: best = (s, R, cx)
    wheels.append(dict(R=best[1], c=np.array([best[2], w['gy'] + best[1], zc]), hw=hw, role=w['role'], side=int(np.sign(zc)), unit=cfg.wheel_unit(best[2], zc), front=cfg.is_front(best[2])))
    print('wheel x=%.3f z=%.3f R=%.3f role=%s unit=%s' % (best[2], zc, best[1], w['role'], wheels[-1]['unit']))
cen = Pm[Im].mean(1); wid = np.full(len(Im), -1)
for k, w in enumerate(wheels):
    d = np.hypot(cen[:, 0] - w['c'][0], cen[:, 1] - w['c'][1])
    sel = (d < w['R'] + 0.004) & (np.abs(cen[:, 2] - w['c'][2]) <= w['hw'] + 0.004) & (wid < 0)
    wid[sel] = k
print('wheel tris', [int((wid == k).sum()) for k in range(len(wheels))])
unit_of = np.array([cfg.unit_of_tri(c) for c in cen])    # unit index per triangle, for the body

for ui, U in enumerate(cfg.UNITS):
    body = Im[(wid < 0) & (unit_of == ui)]
    uw = [k for k, w in enumerate(wheels) if w['unit'] == ui]
    allv = np.unique(np.concatenate([body.ravel()] + [Im[wid == k].ravel() for k in uw]))
    V = Pm[allv]
    L, W, Hh = U['box']; fr = fit(V, L, W, Hh)
    sx, sy, sz = fr['s']
    if U.get('hfit'): sy = Hh / (U['hfit'][1] - U['hfit'][0]); fr['s'] = (sx, sy, sz)   # height measured to the cab roof, not the stacks
    if U.get('uniform_h'): sz = sy; fr['s'] = (sx, sy, sz)
    if U.get('uniform_lh'): s_ = (sy + sz) / 2; sy = sz = s_; fr['s'] = (sx, sy, sz)
    G_all = lambda P_: to_game(P_, fr)
    dz = 0.0
    if U.get('anchor_front'): dz = L / 2 - G_all(V)[:, 2].max()
    G_ = lambda P_: G_all(P_) + np.array([0, 0, dz])
    # ---- body on the game frame
    P = G_(Pm); UV = UVm; I = body
    hn = to_game(hi['pos'], fr) + np.array([0, 0, dz]); HN = nrm_game(hi['nrm'], fr); tree = cKDTree(hn)
    C = P[I].mean(1); col = samp(UV[I].mean(1))
    FN = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]]); FN /= np.linalg.norm(FN, axis=1, keepdims=True) + 1e-12
    mx_, mn_ = col.max(1), col.min(1); sat = (mx_ - mn_) / (mx_ + 1e-4); lum = col @ np.array([0.299, 0.587, 0.114])
    print(U['name'], 'fit scale', np.round(fr['s'], 3), 'dz', round(dz, 3), 'game bbox', G_(V).min(0).round(2), G_(V).max(0).round(2))
    # glass: dark grey faces, holes filled, inner skin dropped
    ek = {}
    for fi, t in enumerate(I):
        for a, b in ((0, 1), (1, 2), (2, 0)): ek.setdefault((min(t[a], t[b]), max(t[a], t[b])), []).append(fi)
    nbr = [[] for _ in I]
    for fs in ek.values():
        for a in fs:
            for b in fs:
                if a != b: nbr[a].append(b)
    base = cfg.is_glass(C, col, sat, lum, U['name'])
    for _ in range(4):
        add_ = np.array([(not base[i]) and cfg.fill_ok(C[i], U['name']) and sum(base[j] for j in nbr[i]) >= 2 for i in range(len(I))])
        if not add_.any(): break
        base |= add_
    lum = np.where(base, 0.3, lum); sat = np.where(base, 0.0, sat)
    G = cfg.classify(U['name'], C, col, FN, sat, lum, base)
    isg = np.array([n == 'glass' for n in G]); outw = (FN * (C - np.array(cfg.CENTRE[U['name']]))).sum(1) > 0
    G[isg & ~outw] = 'drop'
    names = sorted(set(G) - {'drop'}); print({n: int((G == n).sum()) for n in names})
    GNV = np.zeros_like(P); FA = np.cross(P[I[:, 1]] - P[I[:, 0]], P[I[:, 2]] - P[I[:, 0]])
    for k in range(3): np.add.at(GNV, I[:, k], FA)
    GNV /= np.linalg.norm(GNV, axis=1, keepdims=True) + 1e-12
    def tnorm(X, gref=None):
        d, i = tree.query(X, k=12); w = 1 / (d + 1e-5); Nn = HN[i]
        if gref is not None: w = w * ((Nn * gref[:, None, :]).sum(2) > 0.2)
        n = (Nn * w[..., None]).sum(1); bad = np.linalg.norm(n, axis=1) < 1e-6
        if gref is not None: n[bad] = gref[bad]
        return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)
    blobs = []
    def add(a, dt): b = np.ascontiguousarray(a, dtype=dt).tobytes(); blobs.append(b); return len(blobs) - 1
    def pack(tris, flatten=False, inset=0.004):
        u, inv = np.unique(tris.ravel(), return_inverse=True); p = P[u].copy(); uv = UV[u].copy(); ix = inv.reshape(-1, 3); n = tnorm(p, GNV[u])
        if flatten:
            A = coo_matrix((np.ones(ix.size), (np.repeat(ix[:, 0], 3), ix.ravel())), shape=(len(u), len(u))); nc, lb = connected_components(A + A.T, directed=False)
            for k in range(nc):
                m = lb == k; q = p[m]
                if len(q) < 4: continue
                c0 = q.mean(0); _, _, vt = np.linalg.svd(q - c0); nn = vt[2]
                if nn @ n[m].mean(0) < 0: nn = -nn
                p[m] = q - np.outer((q - c0) @ nn, nn) - nn * inset; n[m] = nn
        return dict(pos=add(p, np.float32), nrm=add(np.clip(np.round(n * 127), -127, 127), np.int8), uv=add(uv, np.float32), idx=add(ix, np.uint16 if len(u) < 65536 else np.uint32), big=len(u) >= 65536, n=len(u))
    parts = []
    for nm in names:
        tri = I[G == nm]
        kind = 'glass' if nm == 'glass' else 'lamp' if nm in ('head', 'tail') else 'rigid' if nm.startswith(('bump', 'mirror', 'brk')) else 'body'
        parts.append(dict(name=nm, kind=kind, owner=None, pivot=[0, 0, 0], geo=pack(tri, flatten=(kind == 'glass' and getattr(cfg, 'FLATTEN', True))), tris=len(tri)))
    # ---- wheels: one mesh per (side, role, front) in this unit, round again, centred on the hub
    wl = []; tmpl = {}
    for k in uw:
        w = wheels[k]; key = (w['side'], w['role'], w['front'])
        hub = G_(w['c'][None])[0]; s_r = np.sqrt(sy * sz)
        if key not in tmpl:
            tri = Im[wid == k]; u, inv = np.unique(tri.ravel(), return_inverse=True); d = Pm[u] - w['c']
            wp = np.stack([d[:, 2] * sx, d[:, 1] * s_r, -d[:, 0] * s_r], 1); ix = inv.reshape(-1, 3)
            d2, i2 = tree.query(G_(Pm[u]), k=6); ww = 1 / (d2 + 1e-5); nn_ = (HN[i2] * ww[..., None]).sum(1); nn_ /= np.linalg.norm(nn_, axis=1, keepdims=True) + 1e-12
            # normals into the wheel's own frame (same axes scaling as the points)
            nn_ = nn_ / np.linalg.norm(nn_, axis=1, keepdims=True)
            tmpl[key] = dict(pos=add(wp, np.float32), nrm=add(np.clip(np.round(nn_ * 127), -127, 127), np.int8), uv=add(UVm[u], np.float32), idx=add(ix, np.uint16), big=False, n=len(u))
        wl.append(dict(x=float(hub[0]), y=float(hub[1]), z=float(hub[2]), front=bool(w['front']), side=int(w['side']), r=float(w['R'] * s_r), geo=tmpl[key]))
    # ---- paint mask by UV region
    mw = 1024; M = Image.new('L', (mw, mw), 0); dr = ImageDraw.Draw(M)
    pm = cfg.paint_tris(U['name'], C, col, FN, sat, lum, G)
    for ti in np.where(pm)[0]: dr.polygon([(UV[k][0] * mw, UV[k][1] * mw) for k in I[ti]], fill=255)
    big = np.asarray(img.resize((mw, mw), Image.BILINEAR)).astype(np.float32) / 255; bl = big @ np.array([0.299, 0.587, 0.114])
    mask = np.clip((bl - cfg.TRIM_LUM) / 0.1, 0, 1) * (np.asarray(M).astype(np.float32) / 255)
    gm = Image.new('L', (mw, mw), 0); gd = ImageDraw.Draw(gm)
    for ti in np.where(isg & outw)[0]: gd.polygon([(UV[k][0] * mw, UV[k][1] * mw) for k in I[ti]], fill=255)
    gloss = np.asarray(gm).astype(np.float32) / 255
    ref_lum = float(np.median(bl[mask > 0.8])) if (mask > 0.8).sum() > 100 else 0.5
    print(U['name'], 'paint cover', round(float((mask > 0.5).mean()), 3), 'ref lum', round(ref_lum, 3))
    bc = io.BytesIO(); img.resize((cfg.TEX, cfg.TEX), Image.LANCZOS).save(bc, 'JPEG', quality=78); mk = io.BytesIO()
    Image.fromarray(np.stack([mask * 255, gloss * 255, np.zeros_like(mask)], 2).astype(np.uint8)).resize((512, 512), Image.BILINEAR).save(mk, 'JPEG', quality=86)
    Hd = dict(v=2, parts=[{k: v for k, v in p.items() if k != 'tris'} for p in parts], wheels=wl, r=float(np.mean([w['r'] for w in wl])), refLum=ref_lum, cab=None,
              tex=add(np.frombuffer(bc.getvalue(), np.uint8), np.uint8), mask=add(np.frombuffer(mk.getvalue(), np.uint8), np.uint8))
    Hd.update(U.get('extra', {}))
    offs = []; o = 0
    for b in blobs: offs.append([o, len(b)]); o += (len(b) + 3) // 4 * 4
    Hd['blobs'] = offs
    js = json.dumps(Hd).encode(); js += b' ' * ((4 - len(js) % 4) % 4)
    out = os.path.join(OUTD, U['name'] + '.cjm')
    with open(out, 'wb') as fh:
        fh.write(b'CJM2'); fh.write(struct.pack('<I', len(js))); fh.write(js)
        for b in blobs: fh.write(b); fh.write(b'\0' * ((4 - len(b) % 4) % 4))
    print('wrote', out, os.path.getsize(out) // 1024, 'KB', [(p['name'], p['tris']) for p in parts], 'wheels', len(wl))
