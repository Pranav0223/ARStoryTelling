"""
retarget.py
-----------
Maps SMPL animation data (.npz) onto a RigAnything .glb character.

Root problem this script solves:
  RigAnything exports bones with translations in the 30000-120000 unit range
  while mesh vertices are in 0-1.5 unit range. The inverse bind matrices (IBMs)
  compensate in the bind pose, but any rotation breaks that compensation because
  a 1-degree rotation on a bone at 80000 units swings child bones by ~1400 units.

Fix applied here:
  1. Detect the scale mismatch (max_bone_translation / max_mesh_coord).
  2. Divide all bone node translations by S.
  3. Divide the translation column of every IBM by S.
  After this, bones are in the same unit scale as the mesh, and rotations produce
  physically-reasonable deformations.

NPZ expected keys:
  local_rot_mats : (n_frames, n_joints, 3, 3)  per-joint local rotation matrices
  root_positions : (n_frames, 3)               (read but not used for translation)

Usage:
  python retarget.py --glb char.glb --npz output.npz --out out.glb
"""

import argparse
import base64
import logging
import struct
import sys
import numpy as np

logger = logging.getLogger(__name__)

try:
    from pygltflib import (
        GLTF2, Animation, AnimationChannel, AnimationChannelTarget,
        AnimationSampler, Accessor, BufferView, Buffer,
    )
except ImportError:
    sys.exit("pygltflib not found.  pip install pygltflib")


# ---------------------------------------------------------------------------
# Quaternion helpers  [x, y, z, w] convention  (GLTF order)
# ---------------------------------------------------------------------------

def rot_mat_to_quat(R):
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], dtype=np.float32)


def batch_rot_mats_to_quats(rot_mats):
    n_f, n_j = rot_mats.shape[:2]
    out = np.zeros((n_f, n_j, 4), dtype=np.float32)
    for f in range(n_f):
        for j in range(n_j):
            out[f, j] = rot_mat_to_quat(rot_mats[f, j])
    return out


def qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
        aw*bw - ax*bx - ay*by - az*bz,
    ], dtype=np.float32)


def qinv(q):
    return np.array([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def qnorm(q):
    n = np.linalg.norm(q)
    return q / n if n > 1e-8 else np.array([0., 0., 0., 1.], dtype=np.float32)


# ---------------------------------------------------------------------------
# GLTF buffer helpers
# ---------------------------------------------------------------------------

def _align4(buf):
    while len(buf) % 4:
        buf.append(0)


def append_f32(buf, arr):
    _align4(buf)
    off = len(buf)
    buf.extend(arr.astype(np.float32).tobytes())
    return off


def add_bv(gltf, buf_idx, byte_off, byte_len):
    idx = len(gltf.bufferViews)
    gltf.bufferViews.append(BufferView(buffer=buf_idx, byteOffset=byte_off, byteLength=byte_len))
    return idx


def add_acc(gltf, bv_idx, count, type_, min_v=None, max_v=None):
    idx = len(gltf.accessors)
    acc = Accessor(bufferView=bv_idx, byteOffset=0, componentType=5126, count=count, type=type_)
    if min_v is not None:
        acc.min = min_v
    if max_v is not None:
        acc.max = max_v
    gltf.accessors.append(acc)
    return idx


# ---------------------------------------------------------------------------
# Skeleton normalization — THE KEY FIX
# ---------------------------------------------------------------------------

def normalize_skeleton(gltf, joint_node_indices):
    """
    RigAnything GLBs have bone translations 30000-120000x larger than the mesh
    vertices. This function:
      1. Computes scale S = max_bone_translation / max_mesh_coord
      2. Divides all bone node translations by S
      3. Divides the translation column of every IBM (col 3, rows 0-2 in
         column-major float32[16]) by S

    After this, a rotation delta produces a mesh deformation proportional to
    the MESH scale (0-1.5 m), not the bone scale (0-120000 units).
    Mathematically: new_IBM = old_IBM with translation / S, which satisfies
    new_bind_world * new_IBM = Identity (same as before normalization).
    """
    # Max absolute value of any bone node's local translation
    max_bone = 0.0
    for ji in joint_node_indices:
        t = gltf.nodes[ji].translation
        if t:
            max_bone = max(max_bone, max(abs(v) for v in t))

    # Max absolute mesh vertex coordinate (from POSITION accessor min/max)
    max_mesh = 0.0
    for mesh in gltf.meshes:
        for prim in mesh.primitives:
            pos_idx = prim.attributes.POSITION
            if pos_idx is not None:
                acc = gltf.accessors[pos_idx]
                for arr in [acc.min, acc.max]:
                    if arr:
                        max_mesh = max(max_mesh, max(abs(v) for v in arr))

    if max_mesh < 1e-6:
        print("    [warn] could not read mesh extent, skipping normalization")
        return 1.0

    S = max_bone / max_mesh
    if S < 2.0:
        print(f"    no normalization needed (S={S:.2f})")
        return 1.0

    print(f"    bone/mesh scale mismatch: max_bone={max_bone:.0f}, max_mesh={max_mesh:.4f}, S={S:.0f}")
    print(f"    dividing all bone translations and IBM translations by {S:.0f} ...")

    # 1. Scale bone node translations
    for ji in joint_node_indices:
        node = gltf.nodes[ji]
        if node.translation:
            node.translation = [v / S for v in node.translation]

    # 2. Scale IBM translation column (column 3 in column-major 4x4 float32)
    #    IBM = [R^T | -R^T * p_bone; 0 | 1]
    #    When p_bone /= S, the translation column also /= S.
    skin = gltf.skins[0]
    if skin.inverseBindMatrices is None:
        print("    [warn] no inverseBindMatrices accessor, IBM not patched")
        return S

    ibm_acc = gltf.accessors[skin.inverseBindMatrices]
    ibm_bv  = gltf.bufferViews[ibm_acc.bufferView]
    buf     = gltf.buffers[ibm_bv.buffer]

    blob = gltf.binary_blob()
    if blob is not None:
        data = bytearray(blob)
    elif buf.uri and buf.uri.startswith('data:'):
        data = bytearray(base64.b64decode(buf.uri.split(',', 1)[1]))
    else:
        print("    [warn] IBM binary not accessible, IBM not patched")
        return S

    base_off = (ibm_bv.byteOffset or 0) + (ibm_acc.byteOffset or 0)
    for i in range(ibm_acc.count):
        off = base_off + i * 64  # 16 floats * 4 bytes each
        mat = list(struct.unpack_from('16f', data, off))
        # Column-major layout: translation is at indices 12, 13, 14
        mat[12] /= S
        mat[13] /= S
        mat[14] /= S
        struct.pack_into('16f', data, off, *mat)

    # Write modified data back as base64 URI
    # (pygltflib consolidates all data: buffers into the binary chunk on save)
    buf.uri = ('data:application/octet-stream;base64,'
               + base64.b64encode(bytes(data)).decode())

    return S


# ---------------------------------------------------------------------------
# Core animation function
# ---------------------------------------------------------------------------

def blend_animation(glb_path, npz_path, out_path, fps=30,
                    joint_indices=None, animation_name="NPZ_Animation"):

    # --- Load NPZ ---
    print(f"[1/6] Loading NPZ: {npz_path}")
    npz = np.load(npz_path, allow_pickle=True)
    if "local_rot_mats" not in npz:
        sys.exit("NPZ missing 'local_rot_mats'. Keys: " + str(list(npz.keys())))

    local_rot_mats = npz["local_rot_mats"]          # (n_frames, n_npz, 3, 3)
    n_frames, n_npz = local_rot_mats.shape[:2]
    print(f"    frames={n_frames}, npz_joints={n_npz}, duration={n_frames/fps:.2f}s @ {fps}fps")

    # --- Load GLB ---
    print(f"[2/6] Loading GLB: {glb_path}")
    gltf = GLTF2().load(glb_path)
    if not gltf.skins:
        sys.exit("GLB has no skins.")

    skin_joints = gltf.skins[0].joints   # node indices of skin bones
    n_glb = len(skin_joints)
    print(f"    nodes={len(gltf.nodes)}, skin_joints={n_glb}, existing_anims={len(gltf.animations)}")

    # --- Resolve joint mapping ---
    if joint_indices is None:
        joint_indices = list(range(min(n_glb, n_npz)))
    n_active = min(len(joint_indices), n_glb)
    joint_indices = joint_indices[:n_active]
    skin_joints   = skin_joints[:n_active]
    print(f"    using {n_active} NPZ joints [{joint_indices[0]}..{joint_indices[-1]}] -> {n_active} GLB bones")

    # --- Normalize skeleton scale (THE KEY FIX) ---
    print("[3/6] Normalizing skeleton scale ...")
    normalize_skeleton(gltf, skin_joints)

    # --- Read rest-pose rotations AFTER normalization ---
    # (translations changed, rotations did not)
    rest_quats = []
    for ji in skin_joints:
        r = gltf.nodes[ji].rotation
        rest_quats.append(
            np.array(r, dtype=np.float32) if r is not None
            else np.array([0., 0., 0., 1.], dtype=np.float32)
        )

    # --- Convert rotations to quaternions ---
    print("[4/6] Computing animation quaternions ...")
    sel_rots   = local_rot_mats[:, joint_indices, :, :]  # (n_frames, n_active, 3, 3)
    smpl_quats = batch_rot_mats_to_quats(sel_rots)        # (n_frames, n_active, 4)

    # Normalize relative to frame 0 so frame 0 = T-pose (mesh stays at origin).
    # delta[f,j] = inv(smpl[0,j]) * smpl[f,j]
    # final[f,j] = rest[j] * delta[f,j]
    final_quats = np.zeros_like(smpl_quats)
    for j in range(n_active):
        inv_q0 = qinv(smpl_quats[0, j])
        rq     = rest_quats[j]
        for f in range(n_frames):
            delta          = qmul(inv_q0, smpl_quats[f, j])
            final_quats[f, j] = qnorm(qmul(rq, delta))

    # --- Build animation buffer ---
    print("[5/6] Building animation buffer ...")
    times      = np.arange(n_frames, dtype=np.float32) / fps
    anim_bytes = bytearray()

    time_off  = append_f32(anim_bytes, times)
    rot_offs  = [append_f32(anim_bytes, final_quats[:, j, :]) for j in range(n_active)]

    anim_b64 = base64.b64encode(bytes(anim_bytes)).decode()
    buf_idx  = len(gltf.buffers)
    gltf.buffers.append(Buffer(
        byteLength=len(anim_bytes),
        uri=f"data:application/octet-stream;base64,{anim_b64}",
    ))

    time_bv  = add_bv(gltf, buf_idx, time_off,  n_frames * 4)
    rot_bvs  = [add_bv(gltf, buf_idx, rot_offs[j], n_frames * 16) for j in range(n_active)]

    time_acc = add_acc(gltf, time_bv,  n_frames, "SCALAR",
                       min_v=[float(times[0])], max_v=[float(times[-1])])
    rot_accs = [add_acc(gltf, rot_bvs[j], n_frames, "VEC4") for j in range(n_active)]

    samplers, channels = [], []
    for j in range(n_active):
        si = len(samplers)
        samplers.append(AnimationSampler(input=time_acc, output=rot_accs[j], interpolation="LINEAR"))
        channels.append(AnimationChannel(
            sampler=si,
            target=AnimationChannelTarget(node=skin_joints[j], path="rotation"),
        ))
    gltf.animations.append(Animation(name=animation_name, samplers=samplers, channels=channels))

    # --- Save ---
    print(f"[6/6] Saving -> {out_path}")
    gltf.save(out_path)

    import os
    kb = os.path.getsize(out_path) / 1024
    print(f"\nDone!  {out_path}  ({kb:.1f} KB)  {n_frames} frames @ {fps}fps  {n_active} channels")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Blend .npz animation onto a .glb skeleton.")
    p.add_argument("--glb",    required=True)
    p.add_argument("--npz",    required=True)
    p.add_argument("--out",    required=True)
    p.add_argument("--fps",    type=int,   default=30)
    p.add_argument("--joints", type=str,   default=None,
                   help="Comma-separated NPZ joint indices (default: 0..n_glb-1)")
    p.add_argument("--name",   type=str,   default="NPZ_Animation")
    p.add_argument("--scale",  type=float, default=None,
                   help="(Ignored - scale is now auto-detected from the GLB geometry)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    joint_indices = None
    if args.joints:
        try:
            joint_indices = [int(x.strip()) for x in args.joints.split(",")]
        except ValueError:
            sys.exit("--joints must be comma-separated integers")

    blend_animation(
        glb_path=args.glb,
        npz_path=args.npz,
        out_path=args.out,
        fps=args.fps,
        joint_indices=joint_indices,
        animation_name=args.name,
    )
