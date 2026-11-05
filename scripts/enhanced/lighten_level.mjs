// Jak Living Sky: a custom level's glb with fewer triangles (Brume Village's levels, Cuttlefishthesage's, were
// too heavy for a handheld: 3 to 5 million triangles each). Each mesh is simplified on its own (meshoptimizer),
// its borders locked (no gaps between the rocks); names, extras (collision flags), materials, colours and
// UVs kept, Draco on again. Their meshes split each point some four ways (UV seams, colours, normals): the
// simplifier's permissive mode collapses across those splits, the attributes weighing in.
//
//   node lighten_level.mjs <in.glb> <out.glb> [ratio=0.35] [error=0.02]
//
// Only the decoration (drawn, not collided with: extras ignore: 1 or no set_collision): the floors and walls
// Jak stands on and bumps into (set_collision, and set_invisible ones never drawn) are left exactly as made.
// Simplified, a large floor moved by metres (2% of its size): invisible walls, gaps onto the void. They're
// few (a tenth of the triangles); the decoration is the rest.
//
// (needs, from npm: @gltf-transform/core, extensions and functions, draco3dgltf, meshoptimizer)
import { NodeIO } from '@gltf-transform/core';
import { ALL_EXTENSIONS, KHRDracoMeshCompression } from '@gltf-transform/extensions';
import { compactPrimitive } from '@gltf-transform/functions';
import { MeshoptSimplifier } from 'meshoptimizer';
import draco3d from 'draco3dgltf';

const [input, output, ratioArg, errorArg] = process.argv.slice(2);
const ratio = Number(ratioArg ?? 0.35);
const error = Number(errorArg ?? 0.02);

// the attributes the simplifier weighs (how much each matters against the shape)
const WEIGHTS = { NORMAL: 0.5, TEXCOORD_0: 1.0, COLOR_0: 0.5 };

function count(prim) {
  const idx = prim.getIndices();
  return (idx ? idx.getCount() : prim.getAttribute('POSITION').getCount()) / 3;
}

function lighten(prim) {
  const idx = prim.getIndices();
  const pos = prim.getAttribute('POSITION');
  if (!idx || prim.getMode() !== 4 || count(prim) < 64) {
    return;
  }
  // (an index list shared with another mesh: its own copy, the other left as it was)
  if (idx.listParents().filter((x) => x !== doc.getRoot()).length > 1) {
    const own = idx.clone();
    prim.setIndices(own);
    return lighten(prim);
  }
  const n = pos.getCount();
  const positions = new Float32Array(n * 3);
  const el = [];
  for (let i = 0; i < n; i++) {
    positions.set(pos.getElement(i, el), i * 3);
  }
  // the attributes, as floats side by side
  const used = Object.keys(WEIGHTS).filter((s) => prim.getAttribute(s));
  const sizes = used.map((s) => prim.getAttribute(s).getElementSize());
  const stride = sizes.reduce((a, b) => a + b, 0);
  const attributes = new Float32Array(Math.max(1, n * stride));
  const weights = [];
  used.forEach((s, k) => {
    for (let c = 0; c < sizes[k]; c++) weights.push(WEIGHTS[s]);
  });
  for (let i = 0; i < n; i++) {
    let o = i * stride;
    used.forEach((s, k) => {
      attributes.set(prim.getAttribute(s).getElement(i, new Array(sizes[k])), o);
      o += sizes[k];
    });
  }
  const indices = new Uint32Array(idx.getArray());
  const target = Math.floor((indices.length * ratio) / 3) * 3;
  const [out] = used.length
    ? MeshoptSimplifier.simplifyWithAttributes(indices, positions, 3, attributes, stride, weights, null, target,
                                               error, ['LockBorder', 'Permissive'])
    : MeshoptSimplifier.simplify(indices, positions, 3, target, error, ['LockBorder']);
  if (out.length >= 3 && out.length < indices.length) {
    idx.setArray(n > 65535 ? out : new Uint16Array(out));
    compactPrimitive(prim);
  }
}

const io = new NodeIO().registerExtensions(ALL_EXTENSIONS).registerDependencies({
  'draco3d.decoder': await draco3d.createDecoderModule(),
  'draco3d.encoder': await draco3d.createEncoderModule(),
});
const doc = await io.read(input);
await MeshoptSimplifier.ready;
// the meshes collided with, or never drawn: kept
const kept = new Set();
for (const node of doc.getRoot().listNodes()) {
  const m = node.getMesh();
  const x = node.getExtras() || {};
  if (m && ((x.set_collision && !x.ignore) || x.set_invisible)) kept.add(m);
}
const prims = doc.getRoot().listMeshes().filter((m) => !kept.has(m)).flatMap((m) => m.listPrimitives());
const before = prims.reduce((a, p) => a + count(p), 0);
prims.forEach(lighten);
const after = prims.reduce((a, p) => a + count(p), 0);
// (Draco at full precision: its default 14 bits moved a large floor by centimetres, cracks between pieces)
const draco = doc.getRoot().listExtensionsUsed().find((x) => x.extensionName === 'KHR_draco_mesh_compression');
if (draco) {
  draco.setEncoderOptions({
    method: KHRDracoMeshCompression.EncoderMethod.EDGEBREAKER,
    quantizePosition: 20,
    quantizeNormal: 12,
    quantizeTexcoord: 16,
    quantizeColor: 12,
    quantizeGeneric: 16,
  });
}
await io.write(output, doc);
console.log(`${before} -> ${after} triangles (${Math.round((100 * after) / before)}%)`);
