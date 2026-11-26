# Blender: slice the estate body along its shut lines so panels can be separated cleanly (UVs carried across cuts)
import bpy, bmesh, numpy as np, sys
d = np.load(sys.argv[-2]); P, UV, I, wid = d['pos'], d['uv'], d['idx'], d['wid']
CUTS = [((0, 0, 0.84), (0, 0, 1)), ((0, 0, -0.06), (0, 0, 1)), ((0, 0, -0.94), (0, 0, 1)),   # door shut lines
        ((0, 0.29, 0), (0, 1, 0)), ((0, 0.9, 0), (0, 1, 0)),                                    # sill, belt
        ((0, 0, 1.12), (0, 0, 1)), ((0.71, 0, 0), (1, 0, 0)), ((-0.71, 0, 0), (1, 0, 0)),       # bonnet: cowl, sides
        ((0, 0.55, 0), (0, 1, 0)), ((0, 0, -2.16), (0, 0, 1)),                                  # tailgate bottom, rear face
        ((0, 0, 2.15), (0, 0, 1)), ((0, 0.5, 0), (0, 1, 0)), ((0, 0, -2.2), (0, 0, 1))]         # bumpers
def mesh_from(tris):
    bm = bmesh.new(); vs = [bm.verts.new(p) for p in P]; bm.verts.ensure_lookup_table()
    uvl = bm.loops.layers.uv.new('uv')
    for t in tris:
        try: f = bm.faces.new([vs[t[0]], vs[t[1]], vs[t[2]]])
        except ValueError: continue
        for lp, k in zip(f.loops, t): lp[uvl].uv = (UV[k][0], UV[k][1])
    bmesh.ops.remove_doubles(bm, verts=bm.verts[:], dist=0.0)
    return bm, uvl
def export(bm, uvl, tag):
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    key = {}; pos = []; uv = []; idx = []
    for f in bm.faces:
        tri = []
        for lp in f.loops:
            k = (lp.vert.index, round(lp[uvl].uv[0], 6), round(lp[uvl].uv[1], 6))
            if k not in key: key[k] = len(pos); pos.append(tuple(lp.vert.co)); uv.append(tuple(lp[uvl].uv))
            tri.append(key[k])
        idx.append(tri)
    return np.array(pos, np.float32), np.array(uv, np.float32), np.array(idx, np.int32)
bm, uvl = mesh_from(I[wid < 0])
for co, no in CUTS:
    g = bm.verts[:] + bm.edges[:] + bm.faces[:]
    bmesh.ops.bisect_plane(bm, geom=g, plane_co=co, plane_no=no, dist=0.0005)
bm.verts.index_update()
bp, bu, bi = export(bm, uvl, 'body')
out = dict(pos=bp, uv=bu, idx=bi)
for k in range(4):
    wb, wl = mesh_from(I[wid == k]); wb.verts.index_update(); p_, u_, i_ = export(wb, wl, 'w'); out[f'w{k}_pos'] = p_; out[f'w{k}_uv'] = u_; out[f'w{k}_idx'] = i_
np.savez(sys.argv[-1], **out, wheels=d['wheels'], r=d['r'])
print('cut body tris', len(bi), 'verts', len(bp))
