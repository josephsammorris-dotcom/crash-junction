# Vehicle models

`*.cjm` files are traffic vehicles converted from 3D models generated with Meshy (meshy.ai) from the
project's own concept art. Meshy free-plan outputs are licensed CC BY 4.0: credit "Vehicle models made with Meshy".

Format (CJM1): `CJM1`, u32 header length, JSON header, then 4-byte-aligned binary blobs (positions, normals, UVs and
indices for each part at two levels of detail, a front-left wheel centred on its hub, a JPEG colour texture and a JPEG
mask: red = paint, green = glass/gloss). Built by tools/model_prep (simplify with meshoptimizer, normals copied from
the full-resolution model, wheels/bumpers/mirrors split out, scaled to the game's vehicle box).
