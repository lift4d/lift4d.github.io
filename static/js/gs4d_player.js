// gs4d_player.js — client-side deforming anisotropic 3D Gaussian-splat player.
// Loads compact verified params (canonical splats + per-frame node deltas + per-frame
// global 3x3 transforms) and reproduces the panel's exact centers/covariances per
// frame (CPU two-skinning, verified bit-exact). Renders with viser's EWA splat shader.
// A scene is one or more GSObjects (multi-object video scenes) + shared DA3 background,
// camera frustums and trajectories. three.js only for canvas/camera/orbit.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { LineSegments2 } from 'three/addons/lines/LineSegments2.js';
import { LineSegmentsGeometry } from 'three/addons/lines/LineSegmentsGeometry.js';
import { LineMaterial } from 'three/addons/lines/LineMaterial.js';

const f32 = (b) => new Float32Array(b);
const u32 = (b) => new Uint32Array(b);
const u16 = (b) => new Uint16Array(b);
const u8  = (b) => new Uint8Array(b);
// half-float (IEEE 754 binary16) -> Float32 via a 65536-entry lookup table.
let _h2f = null;
function _halfLUT() {
  if (_h2f) return _h2f;
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = (h & 0x8000) ? -1 : 1, e = (h & 0x7C00) >> 10, f = h & 0x03FF;
    t[h] = e === 0  ? s * 6.103515625e-5 * (f / 1024)
         : e === 31 ? (f ? NaN : s * Infinity)
         :            s * Math.pow(2, e - 15) * (1 + f / 1024);
  }
  _h2f = t; return t;
}
const f16 = (b) => { const u = new Uint16Array(b), L = _halfLUT(), o = new Float32Array(u.length); for (let i = 0; i < u.length; i++) o[i] = L[u[i]]; return o; };
// Smallest-three quaternion unpack: (N,) uint32 -> Float32Array (N*4) wxyz.
function _unpackQ3(u32, N) {
  const out = new Float32Array(N * 4), S = 1 / Math.sqrt(2);
  for (let i = 0; i < N; i++) {
    const u = u32[i] >>> 0;
    const imax = (u >>> 30) & 3;
    const cc = [ ((u >>> 20) & 1023) / 1023 * 2 * S - S,
                 ((u >>> 10) & 1023) / 1023 * 2 * S - S,
                 (u & 1023) / 1023 * 2 * S - S ];
    const d = Math.sqrt(Math.max(0, 1 - cc[0]*cc[0] - cc[1]*cc[1] - cc[2]*cc[2]));
    const b = i * 4; let j = 0;
    for (let k = 0; k < 4; k++) out[b + k] = (k === imax) ? d : cc[j++];
  }
  return out;
}

// KNN weights stored as the first K-1 as u8; reconstruct the full (N,K) with last = 1 - sum.
function _expandW(q, N, K) {
  const out = new Float32Array(N * K), km1 = K - 1;
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let k = 0; k < km1; k++) { const w = q[i * km1 + k] / 255; out[i * K + k] = w; s += w; }
    out[i * K + km1] = Math.max(0, 1 - s);
  }
  return out;
}

// One deforming Gaussian object: its own canonical splats, two KNN tables, per-frame
// node deltas and per-frame global transform (glob). Skinned + rendered independently.
class GSObject {
  constructor(dir) { this.dir = dir.endsWith('/') ? dir : dir + '/'; }

  async load() {
    const g = async (n, T) => T(await (await fetch(this.dir + n)).arrayBuffer());
    const m = await (await fetch(this.dir + 'meta.json')).json();
    this.N = m.N; this.M = m.M; this.F = m.F; this.Kd = m.Kd || 0; this.Kb = m.Kb || 0;
    this.optRot = !!m.opt_deform_rot;
    this.hasDelta = !!m.has_delta; this.hasBase = !!m.has_base; this.hasDeltaRot = !!m.has_delta_rot;
    const FL = m.f16 ? f16 : f32;          // per-Gaussian floats + node deltas (f16 or legacy f32)
    const IDX = m.f16 ? u16 : u32;         // KNN node indices
    const W8 = !!m.w8km1;                   // weights as K-1 u8 (reconstruct last = 1 - sum)
    this.pos    = await g('positions.bin', FL);    // (N,3)
    this.rotRaw = m.q3 ? _unpackQ3(await g('rot_raw.bin', u32), this.N) : await g('rot_raw.bin', FL);  // (N,4) wxyz
    this.scl    = await g('scale.bin', FL);        // (N,3) linear
    this.col    = await g('color.bin', u8);        // (N,3)
    this.opac   = await g('opacity.bin', FL);      // (N,1)
    if (this.hasDelta) {
      this.dIdx = await g('delta_idx.bin', IDX);
      this.dW   = W8 ? _expandW(await g('delta_w.bin', u8), this.N, this.Kd) : await g('delta_w.bin', FL);
      this.dNt  = await g('delta_node_trans.bin', FL);
      this.dNr  = this.hasDeltaRot ? await g('delta_node_rot.bin', FL) : null;
    }
    if (this.hasBase) {
      this.bIdx = await g('base_idx.bin', IDX);
      this.bW   = W8 ? _expandW(await g('base_w.bin', u8), this.N, this.Kb) : await g('base_w.bin', FL);
      this.bNt  = await g('base_node_trans.bin', FL);
      this.bNr  = await g('base_node_rot.bin', FL);
      this.bNs  = await g('base_node_scale.bin', FL);
    }
    this.gM = await g('glob_M.bin', f32);
    this.gg = await g('glob_g.bin', f32);
    this.gL = await g('glob_L.bin', f32);
    const N = this.N;
    this.center = new Float32Array(N * 3);
    this.cov    = new Float32Array(N * 6);
    this.depth  = new Float32Array(N);
    this.order  = new Uint32Array(N);
    for (let i = 0; i < N; i++) this.order[i] = i;
    // canonical centroid (cheap proxy for cross-object depth ordering via glob)
    let mx = 0, my = 0, mz = 0;
    for (let i = 0; i < N; i++) { mx += this.pos[i*3]; my += this.pos[i*3+1]; mz += this.pos[i*3+2]; }
    this.canonMean = [mx / N, my / N, mz / N];
    // node-delta keyframing: deltas are stored only at kf_idx frames; interpolate per frame.
    this.kfIdx = m.kf_idx || null;   // keyframe frame indices (== node-array frame dimension)
    this._lastK = 0;
    if (this.hasDelta) { this.dNtC = new Float32Array(this.M * 3); if (this.hasDeltaRot) this.dNrC = new Float32Array(this.M * 4); }
    if (this.hasBase)  { this.bNtC = new Float32Array(this.M * 3); this.bNrC = new Float32Array(this.M * 4); this.bNsC = new Float32Array(this.M * 3); }
    return this;
  }

  // Interpolate the node deltas at frame f from the bracketing keyframes into the
  // per-frame C buffers (linear; quats are residuals, so plain lerp is correct here).
  _interpNodes(f) {
    const M = this.M, idx = this.kfIdx;
    let k0, k1, a;
    if (idx) {
      let k = this._lastK; if (idx[k] > f) k = 0;
      while (k + 1 < idx.length && idx[k + 1] <= f) k++;
      this._lastK = k;
      k0 = k; k1 = Math.min(k + 1, idx.length - 1);
      const span = idx[k1] - idx[k0];
      a = span > 0 ? (f - idx[k0]) / span : 0;
    } else { k0 = k1 = f; a = 0; }   // no keyframing: node arrays are per-frame
    const lerp = (src, C, comp) => {
      const o0 = k0 * M * comp, o1 = k1 * M * comp, n = M * comp, b = 1 - a;
      if (a === 0) for (let i = 0; i < n; i++) C[i] = src[o0 + i];
      else for (let i = 0; i < n; i++) C[i] = b * src[o0 + i] + a * src[o1 + i];
    };
    if (this.hasDelta) { lerp(this.dNt, this.dNtC, 3); if (this.hasDeltaRot) lerp(this.dNr, this.dNrC, 4); }
    if (this.hasBase)  { lerp(this.bNt, this.bNtC, 3); lerp(this.bNr, this.bNrC, 4); lerp(this.bNs, this.bNsC, 3); }
  }

  build(scene, material) {
    const N = this.N;
    const quad = new Float32Array([-2,-2, 2,-2, 2,2, -2,2]);   // viser quad corners (±2)
    const geo = new THREE.InstancedBufferGeometry();
    geo.setAttribute('corner', new THREE.BufferAttribute(quad, 2));
    geo.setIndex([0,2,1, 0,3,2]);
    geo.instanceCount = N;
    this.aCenter = new THREE.InstancedBufferAttribute(new Float32Array(N*3), 3); this.aCenter.setUsage(THREE.DynamicDrawUsage);
    this.aCov0   = new THREE.InstancedBufferAttribute(new Float32Array(N*3), 3); this.aCov0.setUsage(THREE.DynamicDrawUsage);
    this.aCov1   = new THREE.InstancedBufferAttribute(new Float32Array(N*3), 3); this.aCov1.setUsage(THREE.DynamicDrawUsage);
    const colf = new Float32Array(N*4);   // canonical rgba (SOURCE — must not alias the sorted output)
    for (let i=0;i<N;i++){ colf[i*4]=this.col[i*3]/255; colf[i*4+1]=this.col[i*3+1]/255; colf[i*4+2]=this.col[i*3+2]/255; colf[i*4+3]=Math.min(1,Math.max(0,this.opac[i])); }
    this.aColor = new THREE.InstancedBufferAttribute(new Float32Array(N*4), 4); this.aColor.setUsage(THREE.DynamicDrawUsage);
    geo.setAttribute('iCenter', this.aCenter);
    geo.setAttribute('iCov0', this.aCov0);
    geo.setAttribute('iCov1', this.aCov1);
    geo.setAttribute('iColor', this.aColor);
    this.colf = colf;
    this.mesh = new THREE.Mesh(geo, material);
    this.mesh.frustumCulled = false;
    this.mesh.renderOrder = 10000;   // after the opaque BG so depthTest occludes correctly
    scene.add(this.mesh);
  }

  // Two-skinning reproduction of the panel's centers/covs for frame f (verified exact):
  //   d_xyz = Σ dW·dNt + Σ bW·bNt ; d_rot = Σ dW·dNr + Σ bW·bNr ; d_scale = Σ bW·bNs
  //   q = normalize(rot_raw + d_rot) ; s = max(scale + d_scale, eps)
  //   p_world = (pos + d_xyz) @ M + g ; cov_world = (L·R) diag(s²) (L·R)^T
  _skin(f) {
    const N = this.N, M = this.M, Kd = this.Kd, Kb = this.Kb;
    const pos = this.pos, rotRaw = this.rotRaw, scl = this.scl;
    const Mm = this.gM, gg = this.gg, L = this.gL;
    const hasDelta = this.hasDelta, hasBase = this.hasBase, hasDeltaRot = this.hasDeltaRot, optRot = this.optRot;
    this._interpNodes(f);                          // node deltas for frame f -> C buffers (keyframe lerp)
    const dIdx = this.dIdx, dW = this.dW, dNt = this.dNtC, dNr = this.dNrC;
    const bIdx = this.bIdx, bW = this.bW, bNt = this.bNtC, bNr = this.bNrC, bNs = this.bNsC;
    const mo = f * 9, go = f * 3;                   // glob is per-frame (not keyframed)
    const M0=Mm[mo],M1=Mm[mo+1],M2=Mm[mo+2],M3=Mm[mo+3],M4=Mm[mo+4],M5=Mm[mo+5],M6=Mm[mo+6],M7=Mm[mo+7],M8=Mm[mo+8];
    const g0=gg[go],g1=gg[go+1],g2=gg[go+2];
    const L0=L[mo],L1=L[mo+1],L2=L[mo+2],L3=L[mo+3],L4=L[mo+4],L5=L[mo+5],L6=L[mo+6],L7=L[mo+7],L8=L[mo+8];
    const center = this.center, cov = this.cov;
    for (let i = 0; i < N; i++) {
      const i3 = i*3, i4 = i*4;
      let dx=0,dy=0,dz=0, rw=0,rx=0,ry=0,rz=0, sx=0,sy=0,sz=0;
      if (hasDelta) { const ik = i*Kd;
        for (let k=0;k<Kd;k++){ const node=dIdx[ik+k], w=dW[ik+k], to=node*3;
          dx+=w*dNt[to]; dy+=w*dNt[to+1]; dz+=w*dNt[to+2];
          if (hasDeltaRot){ const ro=node*4; rw+=w*dNr[ro]; rx+=w*dNr[ro+1]; ry+=w*dNr[ro+2]; rz+=w*dNr[ro+3]; }
        }
      }
      if (hasBase) { const ik = i*Kb;
        for (let k=0;k<Kb;k++){ const node=bIdx[ik+k], w=bW[ik+k], to=node*3, ro=node*4;
          dx+=w*bNt[to]; dy+=w*bNt[to+1]; dz+=w*bNt[to+2];
          rw+=w*bNr[ro]; rx+=w*bNr[ro+1]; ry+=w*bNr[ro+2]; rz+=w*bNr[ro+3];
          sx+=w*bNs[to]; sy+=w*bNs[to+1]; sz+=w*bNs[to+2];
        }
      }
      const px=pos[i3]+dx, py=pos[i3+1]+dy, pz=pos[i3+2]+dz;
      center[i3]   = px*M0+py*M3+pz*M6 + g0;
      center[i3+1] = px*M1+py*M4+pz*M7 + g1;
      center[i3+2] = px*M2+py*M5+pz*M8 + g2;
      let qw=rotRaw[i4], qx=rotRaw[i4+1], qy=rotRaw[i4+2], qz=rotRaw[i4+3];
      if (optRot){ qw+=rw; qx+=rx; qy+=ry; qz+=rz; }
      const qn=1.0/Math.hypot(qw,qx,qy,qz); qw*=qn;qx*=qn;qy*=qn;qz*=qn;
      const r00=1-2*(qy*qy+qz*qz), r01=2*(qx*qy-qw*qz), r02=2*(qx*qz+qw*qy);
      const r10=2*(qx*qy+qw*qz), r11=1-2*(qx*qx+qz*qz), r12=2*(qy*qz-qw*qx);
      const r20=2*(qx*qz-qw*qy), r21=2*(qy*qz+qw*qx), r22=1-2*(qx*qx+qy*qy);
      let ax=scl[i3]+sx, ay=scl[i3+1]+sy, az=scl[i3+2]+sz;
      if (ax<1e-9) ax=1e-9; if (ay<1e-9) ay=1e-9; if (az<1e-9) az=1e-9;
      const s0=ax*ax, s1=ay*ay, s2=az*az;
      const A00=L0*r00+L1*r10+L2*r20, A01=L0*r01+L1*r11+L2*r21, A02=L0*r02+L1*r12+L2*r22;
      const A10=L3*r00+L4*r10+L5*r20, A11=L3*r01+L4*r11+L5*r21, A12=L3*r02+L4*r12+L5*r22;
      const A20=L6*r00+L7*r10+L8*r20, A21=L6*r01+L7*r11+L8*r21, A22=L6*r02+L7*r12+L8*r22;
      const o=i*6;
      cov[o]   = A00*A00*s0 + A01*A01*s1 + A02*A02*s2;
      cov[o+1] = A00*A10*s0 + A01*A11*s1 + A02*A12*s2;
      cov[o+2] = A00*A20*s0 + A01*A21*s1 + A02*A22*s2;
      cov[o+3] = A10*A10*s0 + A11*A11*s1 + A12*A12*s2;
      cov[o+4] = A10*A20*s0 + A11*A21*s1 + A12*A22*s2;
      cov[o+5] = A20*A20*s0 + A21*A21*s1 + A22*A22*s2;
    }
  }

  // skin frame f (clamped by caller) + depth-sort by view matrix `v` + fill instance attrs
  upload(v, f) {
    this._skin(f);
    const c = this.center, depth = this.depth, order = this.order, N = this.N;
    for (let i=0;i<N;i++){ depth[i] = v[2]*c[i*3] + v[6]*c[i*3+1] + v[10]*c[i*3+2] + v[14]; }
    order.sort((a,b)=>depth[a]-depth[b]);
    const ac=this.aCenter.array, a0=this.aCov0.array, a1=this.aCov1.array, af=this.aColor.array, cov=this.cov, col=this.colf;
    for (let j=0;j<N;j++){ const i=order[j], o=i*6, j3=j*3, j4=j*4, i3=i*3;
      ac[j3]=c[i3];ac[j3+1]=c[i3+1];ac[j3+2]=c[i3+2];
      a0[j3]=cov[o];a0[j3+1]=cov[o+1];a0[j3+2]=cov[o+2];
      a1[j3]=cov[o+3];a1[j3+1]=cov[o+4];a1[j3+2]=cov[o+5];
      af[j4]=col[i*4];af[j4+1]=col[i*4+1];af[j4+2]=col[i*4+2];af[j4+3]=col[i*4+3];
    }
    this.aCenter.needsUpdate=this.aCov0.needsUpdate=this.aCov1.needsUpdate=this.aColor.needsUpdate=true;
  }
}

export class GS4DPlayer {
  constructor(canvas, dir, opts = {}) {
    this.canvas = canvas;
    this.dir = dir.endsWith('/') ? dir : dir + '/';
    this.fps = opts.fps ?? 15;
    this.playing = true; this._alive = true;
    this._frame = 0; this._acc = 0; this._last = performance.now();
  }

  // Stop this player's render loop + free ALL its GPU resources (call before switching
  // scenes). Otherwise the old loop keeps rendering AND the scene's geometries/buffers
  // (splat instance attributes, background points, line geometries) leak — slowing the
  // new scene down over repeated switches.
  dispose() {
    this._alive = false;
    if (this._raf) cancelAnimationFrame(this._raf);
    try {
      if (this._onResize) window.removeEventListener('resize', this._onResize);
      if (this.controls) this.controls.dispose();
      const seen = new Set();
      const wreck = (o) => {
        if (o.geometry) o.geometry.dispose();
        const mats = o.material ? (Array.isArray(o.material) ? o.material : [o.material]) : [];
        // dispose materials (frees shader programs); skip textures — the circle sprite is shared.
        mats.forEach(m => { if (m && !seen.has(m)) { seen.add(m); m.dispose && m.dispose(); } });
      };
      if (this.scene) this.scene.traverse(wreck);
      if (this.outlineScene) this.outlineScene.traverse(wreck);
      [this.material, this.silMat, this.outlineMat, this.camLineMat].forEach(m => { if (m && !seen.has(m)) { seen.add(m); m.dispose && m.dispose(); } });
      if (this.silRT) this.silRT.dispose();
      if (this.renderer) this.renderer.dispose();
      // drop big typed-array references so the GC can reclaim them
      this.objs = []; this.bgXyz = this.bgRgb = this.bgcXyz = this.bgcRgb = null;
      this.trajXyz = this.trajCol = null; this.scene = null;
    } catch (e) { /* ignore */ }
  }

  async load() {
    const g = async (n, T) => T(await (await fetch(this.dir + n)).arrayBuffer());
    this.meta = await (await fetch(this.dir + 'meta.json')).json();
    // objects: multi-object manifest (meta.objects) -> N bundles in obj<i>/, else the
    // scene dir itself is the single object bundle.
    this.objs = [];
    if (this.meta.objects && this.meta.objects.length) {
      for (const o of this.meta.objects) this.objs.push(await new GSObject(this.dir + o.dir + '/').load());
    } else {
      this.objs.push(await new GSObject(this.dir).load());
    }
    this.F = this.meta.F || this.objs[0].F;
    this.N = this.objs.reduce((a, o) => a + o.N, 0);
    const FL = this.meta.f16 ? f16 : f32;   // bg positions (f16 or legacy f32)
    // shared scene-level extras
    if (this.meta.has_bg) {
      this.bgXyz = await g('bg_xyz.bin', FL); this.bgRgb = await g('bg_rgb.bin', u8); this.bgOff = await g('bg_offsets.bin', u32);
    }
    if (this.meta.has_bg_cur) {
      this.bgcXyz = await g('bgc_xyz.bin', FL); this.bgcRgb = await g('bgc_rgb.bin', u8); this.bgcOff = await g('bgc_offsets.bin', u32);
    }
    if (this.meta.has_cam) {
      this.camPos = await g('cam_pos.bin', f32); this.camWxyz = await g('cam_wxyz.bin', f32);
      this.camFov = await g('cam_fov.bin', f32); this.camAspect = await g('cam_aspect.bin', f32);
      this.nBgFrames = this.meta.n_bg_frames || this.F;
    }
    if (this.meta.n_traj) {
      this.trajXyz = await g('traj_xyz.bin', f32); this.trajCol = await g('traj_colors.bin', f32); this.nTraj = this.meta.n_traj;
    }
    return this;
  }

  start() {
    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: false, alpha: true });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(45, 1, 0.01, 1000);
    this.showOurs = true; this.showBg = true; this.bgAccum = true;
    this.showCam = true; this.showTraj = true; this.trajWidth = 1.5; this.showHighlight = true;
    const up = this.meta.cam_up || [0, -1, 0];
    this.camera.up.set(up[0], up[1], up[2]);
    this.controls = new OrbitControls(this.camera, this.canvas);
    this.controls.enableDamping = true;
    this.controls.zoomSpeed = 4.0;
    // shared splat material (viser EWA shader); per-object meshes reference it
    this.material = new THREE.ShaderMaterial({
      uniforms: { viewport: { value: new THREE.Vector2(1, 1) }, near: { value: 0.01 }, far: { value: 1000.0 } },
      vertexShader: VS, fragmentShader: FS,
      transparent: true, depthTest: true, depthWrite: false, blending: THREE.NormalBlending,
    });
    // framing: above the initial DA3 camera, pulled back along its view ray
    const la = this.meta.cam_lookat;
    if (this.camPos && this.camPos.length >= 3 && la) {
      const c0x=this.camPos[0], c0y=this.camPos[1], c0z=this.camPos[2];
      const dx=c0x-la[0], dy=c0y-la[1], dz=c0z-la[2];
      const dist = Math.hypot(dx,dy,dz) || 1;
      const BACK_OVR = {   // per-scene start distance (smaller = closer, larger = further)
        libby: 2.6, parkour: 5.2, bicycle: 3.6, sam2_human_dog: 2.3,
        rhino: 1.4, camel: 1.4, 'horsejump-low': 1.4, bear: 1.4, bird: 1.4,
      };
      const un = Math.hypot(up[0],up[1],up[2]) || 1, off = 0.2 * dist, BACK = BACK_OVR[this.meta.scene] || 1.8;
      this.camera.position.set(la[0]+dx*BACK + up[0]/un*off, la[1]+dy*BACK + up[1]/un*off, la[2]+dz*BACK + up[2]/un*off);
      this.controls.target.set(la[0], la[1], la[2]);
    } else if (this.meta.cam_eye && la) {
      this.camera.position.set(this.meta.cam_eye[0], this.meta.cam_eye[1], this.meta.cam_eye[2]);
      this.controls.target.set(la[0], la[1], la[2]);
    } else {
      const o0 = this.objs[0]; o0._skin(0);
      let cx=0,cy=0,cz=0; for (let i=0;i<o0.N;i++){cx+=o0.center[i*3];cy+=o0.center[i*3+1];cz+=o0.center[i*3+2];}
      cx/=o0.N;cy/=o0.N;cz/=o0.N;
      let ext=0; for (let i=0;i<o0.N;i++){const dx=o0.center[i*3]-cx,dy=o0.center[i*3+1]-cy,dz=o0.center[i*3+2]-cz; ext=Math.max(ext,Math.hypot(dx,dy,dz));}
      this.controls.target.set(cx,cy,cz);
      this.camera.position.set(cx, cy-2.5*ext, cz+ext);
    }
    for (const o of this.objs) o.build(this.scene, this.material);
    // per-object highlight colors: green for single object; red/green/blue… for multi-object
    const PAL = [[0.95,0.25,0.2],[0.2,0.85,0.3],[0.35,0.55,0.95],[0.95,0.8,0.2]];
    this.objs.forEach((o, i) => o.hlColor = this.objs.length > 1 ? PAL[i % PAL.length] : [0.15, 0.9, 0.45]);
    this._buildExtras();
    this._buildHighlight();
    this._resize();
    this._onResize = () => this._resize();
    window.addEventListener('resize', this._onResize);
    this._loop();
    return this;
  }

  _resize() {
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h; this.camera.updateProjectionMatrix();
    const dw = this.renderer.domElement.width, dh = this.renderer.domElement.height;
    if (this.material) {
      this.material.uniforms.viewport.value.set(dw, dh);
      this.material.uniforms.near.value = this.camera.near;
      this.material.uniforms.far.value = this.camera.far;
    }
    if (this.camLineMat) this.camLineMat.resolution.set(dw, dh);
    if (this.silRT) {
      this.silRT.setSize(dw, dh);
      this.silMat.uniforms.viewport.value.set(dw, dh);
      this.silMat.uniforms.near.value = this.camera.near;
      this.silMat.uniforms.far.value = this.camera.far;
      this.outlineMat.uniforms.texel.value.set(1 / dw, 1 / dh);
    }
  }

  _loop() {
    if (!this._alive) return;
    this._raf = requestAnimationFrame(() => this._loop());
    const now = performance.now();
    if (this.playing) { this._acc += (now - this._last)/1000 * this.fps; if (this._acc >= 1){ this._frame = (this._frame + Math.floor(this._acc)) % this.F; this._acc = 0; } }
    this._last = now;
    this.controls.update();
    this.camera.updateMatrixWorld();
    const v = this.camera.matrixWorldInverse.elements;
    for (const o of this.objs) {
      o.mesh.visible = this.showOurs;
      if (this.showOurs) o.upload(v, Math.min(this._frame, o.F - 1));   // shorter objects hold last frame
    }
    if (this.objs.length > 1) this._sortObjects(v);   // cross-object draw order (far -> near)
    this._updateExtras(this._frame);
    this._render();
  }

  // Order multi-object meshes back-to-front by world-centroid view depth, so the
  // nearer object composites over the farther one (object splats don't write depth).
  _sortObjects(v) {
    const dz = this.objs.map(o => {
      const fc = Math.min(this._frame, o.F - 1), mo = fc * 9, go = fc * 3;
      const m = o.gM, g = o.gg, c = o.canonMean;
      const wx = c[0]*m[mo]   + c[1]*m[mo+3] + c[2]*m[mo+6] + g[go];
      const wy = c[0]*m[mo+1] + c[1]*m[mo+4] + c[2]*m[mo+7] + g[go+1];
      const wz = c[0]*m[mo+2] + c[1]*m[mo+5] + c[2]*m[mo+8] + g[go+2];
      return v[2]*wx + v[6]*wy + v[10]*wz + v[14];      // view-space z (more negative = farther)
    });
    this.objs.map((_, i) => i).sort((a, b) => dz[a] - dz[b])   // farthest first
        .forEach((oi, rank) => { this.objs[oi].mesh.renderOrder = 10000 + rank; });
  }

  _render() {
    const r = this.renderer;
    if (!(this.showHighlight && this.showOurs && this.silRT && this.objs.length)) {
      r.render(this.scene, this.camera); return;
    }
    // Pass 1: render each object's silhouette (its highlight color, splat alpha) into silRT.
    const vis = { bg: this.bgPoints && this.bgPoints.visible, bgc: this.bgCurPoints && this.bgCurPoints.visible,
                  cam: this.camGroup && this.camGroup.visible, traj: this.trajLines && this.trajLines.visible };
    if (this.bgPoints) this.bgPoints.visible = false; if (this.bgCurPoints) this.bgCurPoints.visible = false;
    if (this.camGroup) this.camGroup.visible = false; if (this.trajLines) this.trajLines.visible = false;
    r.setRenderTarget(this.silRT);
    r.setClearColor(0x000000, 0); r.clear(); r.autoClear = false;
    for (const o of this.objs) {
      this.objs.forEach(x => x.mesh.visible = (x === o));
      o.mesh.material = this.silMat;
      this.silMat.uniforms.uHL.value.setRGB(o.hlColor[0], o.hlColor[1], o.hlColor[2]);
      r.render(this.scene, this.camera);
      o.mesh.material = this.material;
    }
    r.autoClear = true; r.setRenderTarget(null);
    for (const o of this.objs) o.mesh.visible = this.showOurs;
    if (this.bgPoints) this.bgPoints.visible = vis.bg; if (this.bgCurPoints) this.bgCurPoints.visible = vis.bgc;
    if (this.camGroup) this.camGroup.visible = vis.cam; if (this.trajLines) this.trajLines.visible = vis.traj;
    // Pass 2: the normal scene.  Pass 3: the silhouette-border overlay on top.
    r.render(this.scene, this.camera);
    r.autoClear = false; r.render(this.outlineScene, this.outlineCam); r.autoClear = true;
  }

  _buildHighlight() {
    this.silRT = new THREE.WebGLRenderTarget(1, 1);
    this.silMat = new THREE.ShaderMaterial({
      uniforms: { viewport: { value: new THREE.Vector2(1, 1) }, near: { value: 0.01 }, far: { value: 1000.0 },
                  uHL: { value: new THREE.Color(0.15, 0.9, 0.45) } },
      vertexShader: VS, fragmentShader: SIL_FS,
      transparent: true, depthTest: true, depthWrite: false, blending: THREE.NormalBlending,
    });
    this.outlineCam = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
    this.outlineScene = new THREE.Scene();
    this.outlineMat = new THREE.ShaderMaterial({
      uniforms: { silTex: { value: this.silRT.texture }, texel: { value: new THREE.Vector2(1, 1) }, bw: { value: 2.5 } },
      vertexShader: QUAD_VS, fragmentShader: OUTLINE_FS,
      transparent: true, depthTest: false, depthWrite: false,
    });
    const q = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), this.outlineMat);
    q.frustumCulled = false; this.outlineScene.add(q);
  }

  _buildExtras() {
    const mkPts = (xyz, rgb) => {
      const P = xyz.length / 3;
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(xyz, 3));
      const col = new Float32Array(P * 3);
      for (let i = 0; i < P * 3; i++) col[i] = rgb[i] / 255;
      geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
      const mat = new THREE.PointsMaterial({
        size: (this.meta.bg_point_size || 0.03) * 2.5, sizeAttenuation: true,
        vertexColors: true, map: _circleTex(), alphaTest: 0.5, transparent: false,
      });
      const pts = new THREE.Points(geo, mat); pts.frustumCulled = false;
      this.scene.add(pts); return pts;
    };
    if (this.meta.has_bg && this.bgXyz && this.bgXyz.length) this.bgPoints = mkPts(this.bgXyz, this.bgRgb);
    if (this.meta.has_bg_cur && this.bgcXyz && this.bgcXyz.length) this.bgCurPoints = mkPts(this.bgcXyz, this.bgcRgb);
    if (this.meta.has_cam && this.camPos) {
      const c = this.meta.cam_color || [80, 200, 255];
      this.camLineMat = new LineMaterial({ color: new THREE.Color(c[0]/255, c[1]/255, c[2]/255).getHex(), linewidth: 3.0 });
      this.camLineMat.resolution.set(this.renderer.domElement.width, this.renderer.domElement.height);
      this.camLines = new LineSegments2(new LineSegmentsGeometry(), this.camLineMat);
      this.camLines.frustumCulled = false;
      this.camGroup = new THREE.Group(); this.camGroup.add(this.camLines); this.scene.add(this.camGroup);
      this._camFrameBuilt = -1;
    }
    if (this.meta.n_traj && this.trajXyz) {
      const F = this.F, nt = this.nTraj, nseg = (F - 1) * nt;
      // Light temporal smoothing of each track (3-tap [0.25,0.5,0.25], endpoints fixed).
      // More passes => wider effective neighbor window; a few scenes get extra smoothing.
      const TRAJ_PASSES = { blackswan: 6, rhino: 6, robot: 6, cloth: 6 };
      const passes = TRAJ_PASSES[this.meta.scene] || 2;
      let tx = this.trajXyz;
      for (let p = 0; p < passes; p++) {
        const sm = tx.slice();
        for (let k = 0; k < nt; k++)
          for (let f = 1; f < F - 1; f++) {
            const c = (f*nt+k)*3, a = ((f-1)*nt+k)*3, b = ((f+1)*nt+k)*3;
            sm[c]   = 0.25*tx[a]   + 0.5*tx[c]   + 0.25*tx[b];
            sm[c+1] = 0.25*tx[a+1] + 0.5*tx[c+1] + 0.25*tx[b+1];
            sm[c+2] = 0.25*tx[a+2] + 0.5*tx[c+2] + 0.25*tx[b+2];
          }
        tx = sm;
      }
      const pos = new Float32Array(nseg * 6), col = new Float32Array(nseg * 6);
      let s = 0;
      for (let f = 0; f < F - 1; f++) {
        for (let k = 0; k < nt; k++) {
          const ia = (f*nt+k)*3, ib = ((f+1)*nt+k)*3, off = s*6;
          pos[off]=tx[ia]; pos[off+1]=tx[ia+1]; pos[off+2]=tx[ia+2];
          pos[off+3]=tx[ib]; pos[off+4]=tx[ib+1]; pos[off+5]=tx[ib+2];
          const cr=this.trajCol[k*3], cg=this.trajCol[k*3+1], cb=this.trajCol[k*3+2];
          col[off]=cr;col[off+1]=cg;col[off+2]=cb; col[off+3]=cr;col[off+4]=cg;col[off+5]=cb;
          s++;
        }
      }
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
      geo.setAttribute('color', new THREE.BufferAttribute(col, 3));
      this.trajGeo = geo; this.trajSegPerFrame = nt;
      this.trajLines = new THREE.LineSegments(geo, new THREE.LineBasicMaterial({ vertexColors: true }));
      this.trajLines.frustumCulled = false;
      this.scene.add(this.trajLines);
    }
  }

  _buildFrustum(fc) {
    const px=this.camPos[fc*3], py=this.camPos[fc*3+1], pz=this.camPos[fc*3+2];
    const w=this.camWxyz[fc*4], x=this.camWxyz[fc*4+1], y=this.camWxyz[fc*4+2], z=this.camWxyz[fc*4+3];
    this.camGroup.position.set(px, py, pz);
    this.camGroup.quaternion.set(x, y, z, w);
    const fov=this.camFov[fc], aspect=this.camAspect[fc], s=(this.meta.cam_scale || 0.35) * 1.67;
    const hh=s*Math.tan(fov/2), hw=hh*aspect, d=s;
    const c0=[-hw,-hh,d], c1=[hw,-hh,d], c2=[hw,hh,d], c3=[-hw,hh,d], ap=[0,0,0];
    const segs=[ap,c0, ap,c1, ap,c2, ap,c3, c0,c1, c1,c2, c2,c3, c3,c0];
    const pos=new Float32Array(segs.length*3);
    for (let i=0;i<segs.length;i++){ pos[i*3]=segs[i][0]; pos[i*3+1]=segs[i][1]; pos[i*3+2]=segs[i][2]; }
    this.camLines.geometry.dispose();
    const g=new LineSegmentsGeometry(); g.setPositions(pos);
    this.camLines.geometry=g;
  }

  _updateExtras(f) {
    const acc = this.bgAccum, hasCur = !!this.bgCurPoints;
    if (this.bgPoints) {                                   // accumulate 0..t; also current frame when no dense set
      const on = this.showBg && (acc || !hasCur);
      this.bgPoints.visible = on;
      if (on) {
        const fb = Math.min(f, this.bgOff.length - 2);
        const start = acc ? 0 : this.bgOff[fb];            // accumulate from 0, else current frame's segment
        this.bgPoints.geometry.setDrawRange(start, Math.max(0, this.bgOff[fb + 1] - start));
      }
    }
    if (this.bgCurPoints) {                                // legacy dense set (if present) → current frame only
      const on = this.showBg && !acc;
      this.bgCurPoints.visible = on;
      if (on) { const fb = Math.min(f, this.bgcOff.length - 2);
        this.bgCurPoints.geometry.setDrawRange(this.bgcOff[fb], Math.max(0, this.bgcOff[fb + 1] - this.bgcOff[fb])); }
    }
    if (this.camGroup) {
      this.camGroup.visible = this.showCam;
      if (this.showCam) {
        const fc = Math.min(f, (this.nBgFrames || this.F) - 1);
        if (fc !== this._camFrameBuilt) { this._buildFrustum(fc); this._camFrameBuilt = fc; }
      }
    }
    if (this.trajLines) {
      const on = this.showTraj && f >= 1;
      this.trajLines.visible = on;
      if (on) this.trajGeo.setDrawRange(0, Math.min(f, this.F - 1) * this.trajSegPerFrame * 2);
    }
  }
}

// Small round sprite for the DA3 background points (point_shape="circle").
let _circleTexCache = null;
function _circleTex() {
  if (_circleTexCache) return _circleTexCache;
  const s = 64, cv = document.createElement('canvas'); cv.width = cv.height = s;
  const ctx = cv.getContext('2d');
  ctx.beginPath(); ctx.arc(s/2, s/2, s/2 - 2, 0, Math.PI*2); ctx.fillStyle = '#fff'; ctx.fill();
  _circleTexCache = new THREE.CanvasTexture(cv);
  return _circleTexCache;
}

// Vertex + fragment shaders: verbatim port of viser 1.0.26's gaussian splat shader
// (client/src/Splatting/GaussianSplatsHelpers.ts) so the look matches the panel.
const VS = /* glsl */`
precision highp float;
attribute vec2 corner;
attribute vec3 iCenter;
attribute vec3 iCov0;
attribute vec3 iCov1;
attribute vec4 iColor;
uniform vec2 viewport; uniform float near; uniform float far;
varying vec4 vRgba; varying vec2 vPosition;
void main(){
  vec4 c_cam = modelViewMatrix * vec4(iCenter, 1.0);
  gl_Position = vec4(0.0, 0.0, 2.0, 1.0);
  if (-c_cam.z < near || -c_cam.z > far) return;
  vec4 pos2d = projectionMatrix * c_cam;
  float clip = 1.1 * pos2d.w;
  if (pos2d.x < -clip || pos2d.x > clip || pos2d.y < -clip || pos2d.y > clip) return;

  float fx = projectionMatrix[0][0] * viewport.x / 2.0;
  float fy = projectionMatrix[1][1] * viewport.y / 2.0;

  mat3 cov3d = mat3(
    iCov0.x, iCov0.y, iCov0.z,
    iCov0.y, iCov1.x, iCov1.y,
    iCov0.z, iCov1.y, iCov1.z
  );
  mat3 J = mat3(
    fx / c_cam.z, 0., 0.0,
    0., fy / c_cam.z, 0.0,
    -(fx * c_cam.x) / (c_cam.z * c_cam.z), -(fy * c_cam.y) / (c_cam.z * c_cam.z), 0.
  );
  mat3 A = J * mat3(modelViewMatrix);
  mat3 cov_proj = A * cov3d * transpose(A);
  float diag1 = cov_proj[0][0] + 0.3;
  float offDiag = cov_proj[0][1];
  float diag2 = cov_proj[1][1] + 0.3;

  float mid = 0.5 * (diag1 + diag2);
  float radius = length(vec2((diag1 - diag2) / 2.0, offDiag));
  float lambda1 = mid + radius;
  float lambda2 = mid - radius;
  if (lambda2 < 0.0) return;
  vec2 diagonalVector = normalize(vec2(offDiag, lambda1 - diag1));
  vec2 v1 = min(sqrt(2.0 * lambda1), 1024.0) * diagonalVector;
  vec2 v2 = min(sqrt(2.0 * lambda2), 1024.0) * vec2(diagonalVector.y, -diagonalVector.x);

  vRgba = iColor;
  float weightedDeterminant = vRgba.a * (diag1 * diag2 - offDiag * offDiag);
  if (weightedDeterminant < 0.25) return;
  vPosition = corner;

  gl_Position = vec4(
    (vec2(pos2d) / pos2d.w
      + corner.x * v1 / viewport * 2.0
      + corner.y * v2 / viewport * 2.0) * pos2d.w, pos2d.z, pos2d.w);
}`;

const FS = /* glsl */`
precision mediump float;
varying vec4 vRgba; varying vec2 vPosition;
void main(){
  float A = -dot(vPosition, vPosition);
  if (A < -4.0) discard;
  float B = exp(A) * vRgba.a;
  if (B < 0.01) discard;
  gl_FragColor = vec4(vRgba.rgb, B);
}`;

// Highlight: silhouette pass (same VS) writes the object's flat highlight color + coverage
// alpha into an offscreen target; the outline pass paints a colored border around the edge.
const SIL_FS = /* glsl */`
precision mediump float;
varying vec4 vRgba; varying vec2 vPosition;
uniform vec3 uHL;
void main(){
  float A = -dot(vPosition, vPosition);
  if (A < -4.0) discard;
  float B = exp(A) * vRgba.a;
  if (B < 0.01) discard;
  gl_FragColor = vec4(uHL, B);
}`;

const QUAD_VS = /* glsl */`
varying vec2 vUv;
void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;

const OUTLINE_FS = /* glsl */`
precision mediump float;
uniform sampler2D silTex; uniform vec2 texel; uniform float bw;
varying vec2 vUv;
void main(){
  vec4 c = texture2D(silTex, vUv);
  float covered = c.a > 0.35 ? 1.0 : 0.0;
  float uncovered = 1.0 - covered;
  vec3 col = c.rgb;
  for (int k = 0; k < 16; k++) {            // disk of radius ~bw: edge band = covered AND uncovered nearby
    float ang = float(k) / 16.0 * 6.2831853;
    vec4 s = texture2D(silTex, vUv + vec2(cos(ang), sin(ang)) * bw * texel);
    if (s.a > 0.35) { covered = 1.0; col = s.rgb; } else { uncovered = 1.0; }
  }
  if (covered * uncovered < 0.5) discard;   // not on a silhouette boundary
  gl_FragColor = vec4(col, 1.0);
}`;
