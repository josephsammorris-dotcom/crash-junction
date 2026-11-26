# Turn a Meshy car (front = -x, up = +y, left = +z) into Crash Junction's vehicle frame and a compact asset file.
import sys, io, json, struct, numpy as np
sys.path.insert(0, '.')
from glbio import read
from scipy.spatial import cKDTree
from PIL import Image
HI, LODS, OUT = sys.argv[1], sys.argv[2].split(','), sys.argv[3]
GL, GW, GH = 4.8, 1.86, 1.45                         # target length / width / height (game car box is 4.9 x 1.86 x 1.45)
hi = read(HI); tree = cKDTree(hi['pos'])
def transfer_normals(P):
    d, i = tree.query(P, k=6); w = 1 / (d + 1e-5); n = (hi['nrm'][i] * w[..., None]).sum(1)
    return n / np.linalg.norm(n, axis=1, keepdims=True)
# measure the body once, from the high-res model
H = hi['pos']; mn, mx = H.min(0), H.max(0)
sz, sx, sy = GL / (mx[0] - mn[0]), GW / (mx[2] - mn[2]), GH / (mx[1] - mn[1])
def to_game(P):  # model (x fwd=-x, y up, z left) -> game (x left, y up from ground, z fwd)
    return np.stack([P[:, 2] * sx, (P[:, 1] - mn[1]) * sy, -(P[:, 0] - (mn[0] + mx[0]) / 2) * sz], 1)
def nrm_game(N):
    n = np.stack([N[:, 2] / sx, N[:, 1] / sy, -N[:, 0] / sz], 1); return n / np.linalg.norm(n, axis=1, keepdims=True)
# wheel centres from the tyre contact patches
y0 = mn[1]; low = H[H[:, 1] < y0 + 0.012]
R = 0.137
wheels = []
for ax in (-1, 1):
    for sd in (-1, 1):
        q = low[(np.sign(low[:, 0]) == ax) & (np.sign(low[:, 2]) == sd)]
        wheels.append(dict(c=np.array([q[:, 0].mean(), y0 + R, q[:, 2].mean()]), front=ax < 0, side=sd))
out = dict(scale=[sx, sy, sz], lods=[], wheels=[])
blobs = []
def add(arr, dt):
    b = np.ascontiguousarray(arr, dtype=dt).tobytes(); blobs.append(b); return len(blobs) - 1
for li, f in enumerate(LODS):
    m = read(f); P, UV, I = m['pos'], m['uv'], m['idx']
    N = transfer_normals(P)
    cen = P[I].mean(1)
    inwheel = np.zeros(len(I), bool); wid = np.full(len(I), -1)
    for k, w in enumerate(wheels):
        c = w['c']; d = np.hypot(cen[:, 0] - c[0], cen[:, 1] - c[1])
        sel = (d < R + 0.006) & (np.abs(cen[:, 2]) > 0.2) & (np.sign(cen[:, 2]) == w['side'])
        inwheel |= sel; wid[sel] = k
    G = to_game(P); NG = nrm_game(N)
    body = I[~inwheel]
    # split the body into parts by region (game frame): bumpers (front/rear, low) and the door mirrors
    gc = G[body].mean(1); zmax, zmin = G[:, 2].max(), G[:, 2].min()
    part = np.zeros(len(body), int)                                       # 0 body
    part[(gc[:, 2] > zmax - 0.3) & (gc[:, 1] < 0.52)] = 1                  # front bumper
    part[(gc[:, 2] < zmin + 0.3) & (gc[:, 1] < 0.55)] = 2                  # rear bumper
    part[(np.abs(gc[:, 0]) > GW / 2 - 0.02) & (gc[:, 1] > 0.8) & (gc[:, 1] < 1.15) & (gc[:, 2] > 0.4) & (gc[:, 2] < 1.6)] = 3  # mirrors
    L = dict(parts=[])
    for pk in range(4):
        tri = body[part == pk]
        if pk == 3:   # mirrors: split left / right
            for sd in (1, -1):
                t2 = tri[np.sign(G[tri].mean(1)[:, 0]) == sd]
                if len(t2): L['parts'].append(dict(name='mirror' + ('L' if sd > 0 else 'R'), idx=t2))
        else: L['parts'].append(dict(name=['body', 'bumpF', 'bumpR'][pk], idx=tri))
    # one wheel template (front left), centred on its hub, as its own small mesh
    w0 = [k for k, w in enumerate(wheels) if w['front'] and w['side'] > 0][0]
    wt = I[wid == w0]
    L['wheel'] = wt
    # compact each part: own vertex list
    def pack(tri, centre=None):
        u, inv = np.unique(tri.ravel(), return_inverse=True)
        p = G[u].copy()
        if centre is not None: p -= centre
        return dict(pos=add(p, np.float32), nrm=add(NG[u], np.float32), uv=add(UV[u], np.float32), idx=add(inv.reshape(-1, 3), np.uint16 if len(u) < 65536 else np.uint32), n=len(u), t=len(tri), big=len(u) >= 65536)
    wc = to_game(wheels[w0]['c'][None])[0]
    lod = dict(parts={p['name']: pack(p['idx']) for p in L['parts']}, wheel=pack(wt, wc))
    out['lods'].append(lod)
    print('lod', li, {k: v['t'] for k, v in lod['parts'].items()}, 'wheel tris', lod['wheel']['t'])
for w in wheels:
    g = to_game(w['c'][None])[0]; out['wheels'].append(dict(x=float(g[0]), y=float(g[1]), z=float(g[2]), front=bool(w['front']), side=int(np.sign(g[0]))))
out['r'] = float(R * sy); out['height'] = GH
# textures: colour map, and a paint mask (texels close to the body colour) for recolouring
img = Image.open(io.BytesIO(hi['tex'])).convert('RGB'); a = np.asarray(img).astype(np.float32) / 255
mxc, mnc = a.max(2), a.min(2); sat = (mxc - mnc) / (mxc + 1e-4)
hsv = np.asarray(img.convert('HSV')).astype(np.float32)
hue = hsv[..., 0] / 255 * 360
ref_hue = np.median(hue[(sat > 0.35) & (mxc > 0.35)])
dh = np.minimum(np.abs(hue - ref_hue), 360 - np.abs(hue - ref_hue))
mask = np.clip((1 - dh / 22) * 1.6, 0, 1) * np.clip((sat - 0.18) / 0.15, 0, 1) * np.clip((mxc - 0.12) / 0.12, 0, 1)
lum = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2])
ref_lum = float(np.median(lum[mask > 0.8]))
gloss = np.clip((0.2 - sat) / 0.12, 0, 1) * np.clip((lum - 0.12) / 0.06, 0, 1) * np.clip((0.42 - lum) / 0.08, 0, 1)  # grey glass
print('glass cover', (gloss > 0.5).mean())
print('paint hue', ref_hue, 'ref lum', ref_lum, 'mask cover', (mask > 0.5).mean())
bc = io.BytesIO(); img.save(bc, 'JPEG', quality=86); mk = io.BytesIO(); Image.fromarray(np.stack([(mask * 255), (gloss * 255), np.zeros_like(mask)], 2).astype(np.uint8)).resize((1024, 1024), Image.BILINEAR).save(mk, 'JPEG', quality=90)
out['refLum'] = ref_lum
tex = add(np.frombuffer(bc.getvalue(), np.uint8), np.uint8); msk = add(np.frombuffer(mk.getvalue(), np.uint8), np.uint8)
out['tex'], out['mask'] = tex, msk
# file: 'CJM1' | u32 json length | json | blobs (each 4-byte aligned), json carries offsets
offs = []; o = 0
for b in blobs: offs.append([o, len(b)]); o += (len(b) + 3) // 4 * 4
out['blobs'] = offs
js = json.dumps(out).encode(); js += b' ' * ((4 - len(js) % 4) % 4)
with open(OUT, 'wb') as fh:
    fh.write(b'CJM1'); fh.write(struct.pack('<I', len(js))); fh.write(js)
    for b in blobs: fh.write(b); fh.write(b'\0' * ((4 - len(b) % 4) % 4))
import os; print('wrote', OUT, os.path.getsize(OUT) // 1024, 'KB', 'wheel r', out['r'], [ (round(w['x'],2), round(w['y'],2), round(w['z'],2)) for w in out['wheels']])
