import { NodeIO } from '@gltf-transform/core';
import { weld, simplify, prune, dedup } from '@gltf-transform/functions';
import { MeshoptSimplifier } from 'meshoptimizer';
const [,, inp, out, ratio, err] = process.argv;
await MeshoptSimplifier.ready;
const io = new NodeIO(); const doc = await io.read(inp);
await doc.transform(weld({ tolerance: 0.0001 }), simplify({ simplifier: MeshoptSimplifier, ratio: +ratio, error: +err, lockBorder: false }), dedup(), prune());
for (const m of doc.getRoot().listMeshes()) for (const p of m.listPrimitives()) console.log('tris', p.getIndices().getCount() / 3, 'verts', p.getAttribute('POSITION').getCount());
await io.write(out, doc);
