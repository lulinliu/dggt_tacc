import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from gsplat.rendering import rasterization
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from dggt.utils.geometry import unproject_depth_map_to_point_map
from dggt.utils.gs import get_split_gs


SCENE_PACKAGE_FORMAT = "dggt_4d_scene_v1"


def _normalize_times(timestamps: torch.Tensor) -> torch.Tensor:
    ts = timestamps.detach().float()
    t_min = ts.min()
    t_max = ts.max()
    if (t_max - t_min).abs() < 1e-8:
        return torch.zeros_like(ts)
    return (ts - t_min) / (t_max - t_min)


def _to_numpy(x: Any, dtype: Optional[np.dtype] = None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def _empty_frame_like(dtype: np.dtype = np.float32) -> Dict[str, np.ndarray]:
    return {
        "means": np.zeros((0, 3), dtype=dtype),
        "rgb": np.zeros((0, 3), dtype=dtype),
        "opacity": np.zeros((0,), dtype=dtype),
        "scales": np.zeros((0, 3), dtype=dtype),
        "quats": np.zeros((0, 4), dtype=dtype),
    }


def _ensure_quat_normalized(quats: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.normalize(quats, dim=-1)


def alpha_t(
    t: torch.Tensor,
    t0: torch.Tensor,
    alpha_base: torch.Tensor,
    gamma0: torch.Tensor,
    gamma1: float,
) -> torch.Tensor:
    gamma1_t = torch.tensor(float(gamma1), device=alpha_base.device, dtype=alpha_base.dtype)
    sigma = torch.log(gamma1_t) / (gamma0**2 + 1e-6)
    conf = torch.exp(sigma * (t - t0) ** 2)
    return (alpha_base * conf).float()


def extract_scene_state(
    model: torch.nn.Module,
    images: torch.Tensor,
    predictions: Dict[str, torch.Tensor],
    extrinsic_w2c: torch.Tensor,
    intrinsic: torch.Tensor,
    timestamps: torch.Tensor,
    bg_mask: torch.Tensor,
    input_views: int,
) -> Dict[str, Any]:
    """
    Extract exportable static/dynamic/sky/camera scene state from one inference batch.

    Args:
        model: DGGT model with sky_model.
        images: [1, S, C, H, W]
        predictions: model output dict.
        extrinsic_w2c: [S, 4, 4]
        intrinsic: [S, 3, 3]
        timestamps: [S]
        bg_mask: [1, S, H, W]
        input_views: 1 or 3.
    """
    device = images.device
    dtype = images.dtype

    depth_map = predictions["depth"][0]  # [S, H, W, 1]
    point_map_np = unproject_depth_map_to_point_map(depth_map, extrinsic_w2c[:, :3, :], intrinsic)
    point_map = torch.from_numpy(point_map_np).to(device).float()[None, ...]  # [1, S, H, W, 3]

    gs_map = predictions["gs_map"]
    gs_conf = predictions["gs_conf"]
    dy_map = predictions["dynamic_conf"].squeeze(-1)  # [1, S, H, W]

    norm_timestamps = _normalize_times(timestamps)

    static_mask = bg_mask & (dy_map < 0.5)
    static_points = point_map[static_mask].reshape(-1, 3)
    static_rgbs, static_opacity, static_scales, static_rotations = get_split_gs(gs_map, static_mask)
    static_dynamic_conf = dy_map[static_mask].sigmoid()
    static_opacity = static_opacity * (1 - static_dynamic_conf)

    static_gamma0 = gs_conf[static_mask]
    static_frame_idx = torch.nonzero(static_mask, as_tuple=False)[:, 1]
    static_t0 = norm_timestamps[static_frame_idx]

    dynamic_frames: List[Dict[str, np.ndarray]] = []
    seq_len = dy_map.shape[1]
    for idx in range(seq_len):
        point_map_i = point_map[:, idx]
        bg_mask_i = bg_mask[:, idx]

        dynamic_point = point_map_i[bg_mask_i].reshape(-1, 3)
        dynamic_rgb, dynamic_opacity, dynamic_scale, dynamic_rotation = get_split_gs(gs_map[:, idx], bg_mask_i)
        dynamic_conf = dy_map[:, idx][bg_mask_i].sigmoid()
        dynamic_opacity = dynamic_opacity * dynamic_conf

        if dynamic_point.shape[0] == 0:
            dynamic_frames.append(_empty_frame_like(np.float32))
            continue

        dynamic_frames.append(
            {
                "means": _to_numpy(dynamic_point, np.float32),
                "rgb": _to_numpy(dynamic_rgb, np.float32),
                "opacity": _to_numpy(dynamic_opacity, np.float32),
                "scales": _to_numpy(dynamic_scale, np.float32),
                "quats": _to_numpy(dynamic_rotation, np.float32),
            }
        )

    # Export sky gaussians by materializing projected/colored sky points once.
    s = extrinsic_w2c.shape[0]
    intrinsics_4x4 = torch.eye(4, device=device, dtype=intrinsic.dtype).unsqueeze(0).repeat(s, 1, 1)
    intrinsics_4x4[:, :3, :3] = intrinsic
    sky_rgb, proj_mask, sky_scale_res = model.sky_model._get_background_color(
        source_images=images,
        source_extrinsics=extrinsic_w2c,
        intrinsics=intrinsics_4x4,
        downsample=1,
    )
    sky_means = model.sky_model.bg_pcd[proj_mask]
    sky_quats = model.sky_model.bg_quat[proj_mask]
    sky_scales = torch.exp(model.sky_model.bg_scales)[proj_mask] + sky_scale_res
    sky_opacity = model.sky_model.bg_opacity.squeeze(-1)[proj_mask]

    s_total = extrinsic_w2c.shape[0]
    if input_views <= 0:
        raise ValueError(f"input_views must be positive, got {input_views}")
    time_index = torch.arange(s_total, device=norm_timestamps.device) // input_views
    view_index = torch.arange(s_total, device=norm_timestamps.device) % input_views

    image_hw = np.array([images.shape[-2], images.shape[-1]], dtype=np.int32)

    scene_state = {
        "frame_times": _to_numpy(norm_timestamps, np.float32),
        "extrinsics_w2c": _to_numpy(extrinsic_w2c, np.float32),
        "intrinsics": _to_numpy(intrinsic, np.float32),
        "time_index": _to_numpy(time_index, np.int32),
        "view_index": _to_numpy(view_index, np.int32),
        "image_hw": image_hw,
        "static": {
            "means": _to_numpy(static_points, np.float32),
            "rgb": _to_numpy(static_rgbs, np.float32),
            "opacity_base": _to_numpy(static_opacity, np.float32),
            "scales": _to_numpy(static_scales, np.float32),
            "quats": _to_numpy(static_rotations, np.float32),
            "lifespan_t0": _to_numpy(static_t0, np.float32),
            "lifespan_gamma0": _to_numpy(static_gamma0, np.float32),
        },
        "sky": {
            "means": _to_numpy(sky_means, np.float32),
            "rgb": _to_numpy(sky_rgb, np.float32),
            "opacity": _to_numpy(sky_opacity, np.float32),
            "scales": _to_numpy(sky_scales, np.float32),
            "quats": _to_numpy(sky_quats, np.float32),
        },
        "dynamic_frames": dynamic_frames,
    }
    return scene_state


def _build_one_motion_link(
    frame_a: Dict[str, np.ndarray],
    frame_b: Dict[str, np.ndarray],
    match_radius: float,
) -> Dict[str, np.ndarray]:
    means_a = frame_a["means"]
    means_b = frame_b["means"]

    na = means_a.shape[0]
    nb = means_b.shape[0]

    if na == 0:
        return {
            "next_idx": np.zeros((0,), dtype=np.int32),
            "valid_match": np.zeros((0,), dtype=bool),
            "velocity": np.zeros((0, 3), dtype=np.float32),
            "born_next_idx": np.arange(nb, dtype=np.int32),
        }

    if nb == 0:
        return {
            "next_idx": np.full((na,), -1, dtype=np.int32),
            "valid_match": np.zeros((na,), dtype=bool),
            "velocity": np.zeros((na, 3), dtype=np.float32),
            "born_next_idx": np.zeros((0,), dtype=np.int32),
        }

    tree = cKDTree(means_b)
    dist, nn_idx = tree.query(means_a, k=1)

    nn_idx = nn_idx.astype(np.int32)
    valid = dist <= float(match_radius)

    velocity = np.zeros((na, 3), dtype=np.float32)
    if np.any(valid):
        velocity[valid] = means_b[nn_idx[valid]] - means_a[valid]

    matched_next = set(nn_idx[valid].tolist())
    born_next_idx = np.array([j for j in range(nb) if j not in matched_next], dtype=np.int32)

    return {
        "next_idx": nn_idx,
        "valid_match": valid.astype(bool),
        "velocity": velocity,
        "born_next_idx": born_next_idx,
    }


def build_dynamic_motion_knn(
    dynamic_frames: List[Dict[str, np.ndarray]],
    match_radius: float,
) -> List[Dict[str, np.ndarray]]:
    motions: List[Dict[str, np.ndarray]] = []
    for i in range(max(0, len(dynamic_frames) - 1)):
        motions.append(_build_one_motion_link(dynamic_frames[i], dynamic_frames[i + 1], match_radius))
    return motions


def _attr_dtype(dtype: str) -> np.dtype:
    if dtype == "float16":
        return np.float16
    if dtype == "float32":
        return np.float32
    raise ValueError(f"Unsupported dtype: {dtype}")


def save_scene_package(
    package_dir: str,
    scene_name: str,
    scene_state: Dict[str, Any],
    motion_links: List[Dict[str, np.ndarray]],
    dtype: str,
    mode: int,
    lifespan_gamma1: float,
    motion_match_radius: float,
    input_views: int,
) -> str:
    os.makedirs(package_dir, exist_ok=True)
    dynamic_dir = os.path.join(package_dir, "dynamic")
    os.makedirs(dynamic_dir, exist_ok=True)

    adt = _attr_dtype(dtype)

    np.savez_compressed(
        os.path.join(package_dir, "cameras.npz"),
        frame_times=_to_numpy(scene_state["frame_times"], np.float32),
        extrinsics_w2c=_to_numpy(scene_state["extrinsics_w2c"], np.float32),
        intrinsics=_to_numpy(scene_state["intrinsics"], np.float32),
        time_index=_to_numpy(scene_state["time_index"], np.int32),
        view_index=_to_numpy(scene_state["view_index"], np.int32),
        image_hw=_to_numpy(scene_state["image_hw"], np.int32),
    )

    static = scene_state["static"]
    np.savez_compressed(
        os.path.join(package_dir, "static.npz"),
        means=_to_numpy(static["means"], adt),
        rgb=_to_numpy(static["rgb"], adt),
        opacity_base=_to_numpy(static["opacity_base"], adt),
        scales=_to_numpy(static["scales"], adt),
        quats=_to_numpy(static["quats"], adt),
        lifespan_t0=_to_numpy(static["lifespan_t0"], np.float32),
        lifespan_gamma0=_to_numpy(static["lifespan_gamma0"], adt),
    )

    sky = scene_state["sky"]
    np.savez_compressed(
        os.path.join(package_dir, "sky.npz"),
        means=_to_numpy(sky["means"], adt),
        rgb=_to_numpy(sky["rgb"], adt),
        opacity=_to_numpy(sky["opacity"], adt),
        scales=_to_numpy(sky["scales"], adt),
        quats=_to_numpy(sky["quats"], adt),
    )

    dynamic_frame_files: List[str] = []
    for i, frame in enumerate(scene_state["dynamic_frames"]):
        rel_path = f"dynamic/frame_{i:06d}.npz"
        abs_path = os.path.join(package_dir, rel_path)
        np.savez_compressed(
            abs_path,
            means=_to_numpy(frame["means"], adt),
            rgb=_to_numpy(frame["rgb"], adt),
            opacity=_to_numpy(frame["opacity"], adt),
            scales=_to_numpy(frame["scales"], adt),
            quats=_to_numpy(frame["quats"], adt),
        )
        dynamic_frame_files.append(rel_path)

    motion_files: List[str] = []
    for i, motion in enumerate(motion_links):
        rel_path = f"dynamic/motion_{i:06d}.npz"
        abs_path = os.path.join(package_dir, rel_path)
        np.savez_compressed(
            abs_path,
            next_idx=_to_numpy(motion["next_idx"], np.int32),
            valid_match=_to_numpy(motion["valid_match"], bool),
            velocity=_to_numpy(motion["velocity"], adt),
            born_next_idx=_to_numpy(motion["born_next_idx"], np.int32),
        )
        motion_files.append(rel_path)

    num_frames = int(_to_numpy(scene_state["frame_times"]).shape[0])
    manifest = {
        "format": SCENE_PACKAGE_FORMAT,
        "scene_name": scene_name,
        "mode": int(mode),
        "time_domain": "normalized_0_1",
        "camera_convention": "OpenCV_world_to_camera",
        "sequence_layout": "time_major_view_minor",
        "lifespan_formula": "alpha_t = alpha_base * exp(log(gamma1)/(gamma0^2+1e-6) * (t0-t)^2)",
        "dtype": dtype,
        "files": {
            "cameras": "cameras.npz",
            "static": "static.npz",
            "sky": "sky.npz",
            "dynamic_frames": dynamic_frame_files,
            "dynamic_motions": motion_files,
            "reference_dir": "meta/reference",
        },
        "counts": {
            "num_frames": num_frames,
            "num_dynamic_motion_links": len(motion_links),
            "num_static_gaussians": int(static["means"].shape[0]),
            "num_sky_gaussians": int(sky["means"].shape[0]),
            "num_dynamic_gaussians_per_frame": [int(frame["means"].shape[0]) for frame in scene_state["dynamic_frames"]],
        },
        "time": {
            "min": float(np.min(scene_state["frame_times"])),
            "max": float(np.max(scene_state["frame_times"])),
        },
        "exporter": {
            "motion_match_radius": float(motion_match_radius),
            "lifespan_gamma1": float(lifespan_gamma1),
            "input_views": int(input_views),
        },
    }

    manifest_path = os.path.join(package_dir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    return manifest_path


def _load_npz_dict(path: str) -> Dict[str, np.ndarray]:
    with np.load(path) as data:
        return {k: data[k] for k in data.files}


def load_scene_package(package_dir: str) -> Dict[str, Any]:
    manifest_path = os.path.join(package_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    if manifest.get("format") != SCENE_PACKAGE_FORMAT:
        raise ValueError(
            f"Unsupported scene package format: {manifest.get('format')} (expected {SCENE_PACKAGE_FORMAT})"
        )

    files = manifest["files"]
    cameras = _load_npz_dict(os.path.join(package_dir, files["cameras"]))
    static = _load_npz_dict(os.path.join(package_dir, files["static"]))
    sky = _load_npz_dict(os.path.join(package_dir, files["sky"]))

    dynamic_frames = [_load_npz_dict(os.path.join(package_dir, rel)) for rel in files["dynamic_frames"]]
    dynamic_motions = [_load_npz_dict(os.path.join(package_dir, rel)) for rel in files["dynamic_motions"]]

    return {
        "package_dir": package_dir,
        "manifest": manifest,
        "cameras": cameras,
        "static": static,
        "sky": sky,
        "dynamic_frames": dynamic_frames,
        "dynamic_motions": dynamic_motions,
    }


def validate_scene_package(package: Dict[str, Any]) -> List[str]:
    issues: List[str] = []

    manifest = package["manifest"]
    cams = package["cameras"]
    dyn_frames = package["dynamic_frames"]
    dyn_motions = package["dynamic_motions"]

    num_frames = int(cams["frame_times"].shape[0])
    if cams["extrinsics_w2c"].shape[0] != num_frames:
        issues.append("cameras.extrinsics_w2c length mismatch")
    if cams["intrinsics"].shape[0] != num_frames:
        issues.append("cameras.intrinsics length mismatch")
    if len(dyn_frames) != num_frames:
        issues.append("dynamic frame count mismatch vs num_frames")
    if len(dyn_motions) != max(0, num_frames - 1):
        issues.append("dynamic motion count mismatch vs num_frames-1")

    for i, motion in enumerate(dyn_motions):
        na = int(dyn_frames[i]["means"].shape[0])
        nb = int(dyn_frames[i + 1]["means"].shape[0])
        if motion["next_idx"].shape[0] != na:
            issues.append(f"motion[{i}].next_idx length mismatch")
        if motion["valid_match"].shape[0] != na:
            issues.append(f"motion[{i}].valid_match length mismatch")
        if motion["velocity"].shape[0] != na:
            issues.append(f"motion[{i}].velocity length mismatch")

        if na > 0 and nb > 0:
            valid = motion["valid_match"].astype(bool)
            next_idx = motion["next_idx"]
            if np.any(valid):
                if np.min(next_idx[valid]) < 0 or np.max(next_idx[valid]) >= nb:
                    issues.append(f"motion[{i}] valid next_idx out of bounds")

        born = motion["born_next_idx"]
        if born.size > 0:
            if np.min(born) < 0 or np.max(born) >= nb:
                issues.append(f"motion[{i}] born_next_idx out of bounds")

    if manifest.get("mode") != 2:
        issues.append("manifest mode must be 2 for v1")

    return issues


def _to_torch(arr: np.ndarray, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.from_numpy(np.asarray(arr)).to(device=device, dtype=dtype)


def _rasterize_gaussians(
    means: torch.Tensor,
    quats: torch.Tensor,
    scales: torch.Tensor,
    opacities: torch.Tensor,
    colors: torch.Tensor,
    viewmat: torch.Tensor,
    intrinsic: torch.Tensor,
    width: int,
    height: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if means.numel() == 0:
        rgbd = torch.zeros((1, height, width, 4), device=viewmat.device, dtype=torch.float32)
        alpha = torch.zeros((1, height, width, 1), device=viewmat.device, dtype=torch.float32)
        aux = torch.zeros((1, height, width, 1), device=viewmat.device, dtype=torch.float32)
        return rgbd, alpha, aux

    rgbd, alpha, aux = rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmat[None],
        Ks=intrinsic[None],
        width=width,
        height=height,
        render_mode="RGB+ED",
    )
    return rgbd, alpha, aux


def _select_view_indices(cameras: Dict[str, np.ndarray], camera_view: int) -> np.ndarray:
    view_index = cameras["view_index"]
    indices = np.where(view_index == int(camera_view))[0]
    if indices.size == 0:
        return np.arange(view_index.shape[0], dtype=np.int64)
    return indices.astype(np.int64)


def _interp_camera_at_time(
    times: np.ndarray,
    extrinsics_w2c: np.ndarray,
    intrinsics: np.ndarray,
    t: float,
) -> Tuple[np.ndarray, np.ndarray, int, int, float]:
    n = times.shape[0]
    if n == 1:
        return extrinsics_w2c[0], intrinsics[0], 0, 0, 0.0

    t_clamped = float(np.clip(t, float(times.min()), float(times.max())))

    right = int(np.searchsorted(times, t_clamped, side="right"))
    left = max(0, right - 1)
    right = min(n - 1, right)

    if left == right or abs(float(times[right] - times[left])) < 1e-8:
        return extrinsics_w2c[left], intrinsics[left], left, right, 0.0

    alpha = float((t_clamped - times[left]) / (times[right] - times[left]))

    r0 = R.from_matrix(extrinsics_w2c[left, :3, :3])
    r1 = R.from_matrix(extrinsics_w2c[right, :3, :3])
    slerp = Slerp([0.0, 1.0], R.concatenate([r0, r1]))
    r_interp = slerp([alpha]).as_matrix()[0]

    t0 = extrinsics_w2c[left, :3, 3]
    t1 = extrinsics_w2c[right, :3, 3]
    t_interp = (1 - alpha) * t0 + alpha * t1

    ext_interp = np.eye(4, dtype=np.float32)
    ext_interp[:3, :3] = r_interp.astype(np.float32)
    ext_interp[:3, 3] = t_interp.astype(np.float32)

    intr_interp = ((1 - alpha) * intrinsics[left] + alpha * intrinsics[right]).astype(np.float32)
    return ext_interp, intr_interp, left, right, alpha


def _dynamic_interpolate(
    frame0: Dict[str, np.ndarray],
    frame1: Dict[str, np.ndarray],
    motion: Dict[str, np.ndarray],
    alpha: float,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    means0 = _to_torch(frame0["means"], device)
    rgb0 = _to_torch(frame0["rgb"], device)
    opacity0 = _to_torch(frame0["opacity"], device)
    scales0 = _to_torch(frame0["scales"], device)
    quats0 = _to_torch(frame0["quats"], device)

    if means0.shape[0] == 0 and frame1["means"].shape[0] == 0:
        return {
            "means": means0,
            "rgb": rgb0,
            "opacity": opacity0,
            "scales": scales0,
            "quats": quats0,
        }

    next_idx = torch.from_numpy(motion["next_idx"]).to(device=device, dtype=torch.long)
    valid = torch.from_numpy(motion["valid_match"].astype(np.bool_)).to(device=device)
    velocity = _to_torch(motion["velocity"], device)

    means = means0.clone()
    rgb = rgb0.clone()
    opacity = opacity0.clone()
    scales = scales0.clone()
    quats = quats0.clone()

    if means.shape[0] > 0 and valid.any():
        means[valid] = means0[valid] + float(alpha) * velocity[valid]

        means1 = _to_torch(frame1["means"], device)
        rgb1 = _to_torch(frame1["rgb"], device)
        opacity1 = _to_torch(frame1["opacity"], device)
        scales1 = _to_torch(frame1["scales"], device)
        quats1 = _to_torch(frame1["quats"], device)

        matched = next_idx[valid]
        rgb[valid] = (1 - alpha) * rgb0[valid] + alpha * rgb1[matched]
        opacity[valid] = (1 - alpha) * opacity0[valid] + alpha * opacity1[matched]
        scales[valid] = (1 - alpha) * scales0[valid] + alpha * scales1[matched]
        q = (1 - alpha) * quats0[valid] + alpha * quats1[matched]
        quats[valid] = _ensure_quat_normalized(q)

    if means.shape[0] > 0:
        opacity[~valid] = opacity[~valid] * (1 - alpha)

    born_idx = torch.from_numpy(motion["born_next_idx"]).to(device=device, dtype=torch.long)
    if born_idx.numel() > 0:
        means1 = _to_torch(frame1["means"], device)
        rgb1 = _to_torch(frame1["rgb"], device)
        opacity1 = _to_torch(frame1["opacity"], device)
        scales1 = _to_torch(frame1["scales"], device)
        quats1 = _to_torch(frame1["quats"], device)

        means = torch.cat([means, means1[born_idx]], dim=0)
        rgb = torch.cat([rgb, rgb1[born_idx]], dim=0)
        opacity = torch.cat([opacity, opacity1[born_idx] * float(alpha)], dim=0)
        scales = torch.cat([scales, scales1[born_idx]], dim=0)
        quats = torch.cat([quats, quats1[born_idx]], dim=0)

    return {
        "means": means,
        "rgb": rgb,
        "opacity": opacity,
        "scales": scales,
        "quats": _ensure_quat_normalized(quats),
    }


def render_scene_at_time(
    package: Dict[str, Any],
    query_time: Optional[float] = None,
    frame_idx: Optional[int] = None,
    camera_view: int = 0,
    extrinsic_override: Optional[np.ndarray] = None,
    intrinsic_override: Optional[np.ndarray] = None,
    gamma1_override: Optional[float] = None,
    match_radius_override: Optional[float] = None,
    device: Optional[str] = None,
    return_depth: bool = True,
) -> Dict[str, Any]:
    cameras = package["cameras"]
    manifest = package["manifest"]

    if frame_idx is None and query_time is None:
        raise ValueError("Either frame_idx or query_time must be provided")

    frame_times = cameras["frame_times"].astype(np.float32)
    if frame_idx is not None:
        if frame_idx < 0 or frame_idx >= frame_times.shape[0]:
            raise IndexError(f"frame_idx {frame_idx} out of range [0, {frame_times.shape[0]-1}]")
        t = float(frame_times[frame_idx])
    else:
        t = float(np.clip(float(query_time), float(frame_times.min()), float(frame_times.max())))

    if device is None:
        device_t = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device_t = torch.device(device)

    view_indices = _select_view_indices(cameras, camera_view)
    view_times = frame_times[view_indices]
    view_extr = cameras["extrinsics_w2c"][view_indices]
    view_intr = cameras["intrinsics"][view_indices]

    if extrinsic_override is None or intrinsic_override is None:
        ext_np, intr_np, left_local, right_local, alpha = _interp_camera_at_time(view_times, view_extr, view_intr, t)
        left_global = int(view_indices[left_local])
        right_global = int(view_indices[right_local])
    else:
        ext_np = np.asarray(extrinsic_override, dtype=np.float32)
        intr_np = np.asarray(intrinsic_override, dtype=np.float32)
        if ext_np.shape == (3, 4):
            ext44 = np.eye(4, dtype=np.float32)
            ext44[:3, :] = ext_np
            ext_np = ext44
        if ext_np.shape != (4, 4):
            raise ValueError(f"extrinsic_override must be 4x4 or 3x4, got {ext_np.shape}")
        if intr_np.shape != (3, 3):
            raise ValueError(f"intrinsic_override must be 3x3, got {intr_np.shape}")
        # Fall back to nearest frame pair for dynamic interpolation context.
        ext_tmp, intr_tmp, left_local, right_local, alpha = _interp_camera_at_time(view_times, view_extr, view_intr, t)
        left_global = int(view_indices[left_local])
        right_global = int(view_indices[right_local])

    gamma1 = float(
        gamma1_override
        if gamma1_override is not None
        else manifest.get("exporter", {}).get("lifespan_gamma1", 0.1)
    )

    motion_radius = float(
        match_radius_override
        if match_radius_override is not None
        else manifest.get("exporter", {}).get("motion_match_radius", 1.0)
    )

    static = package["static"]
    static_means = _to_torch(static["means"], device_t)
    static_rgb = _to_torch(static["rgb"], device_t)
    static_opacity_base = _to_torch(static["opacity_base"], device_t)
    static_scales = _to_torch(static["scales"], device_t)
    static_quats = _to_torch(static["quats"], device_t)
    static_t0 = _to_torch(static["lifespan_t0"], device_t)
    static_gamma0 = _to_torch(static["lifespan_gamma0"], device_t)

    t_tensor = torch.tensor(float(t), device=device_t, dtype=torch.float32)
    static_opacity = alpha_t(t_tensor, static_t0, static_opacity_base, static_gamma0, gamma1)

    dyn_frames = package["dynamic_frames"]
    if left_global == right_global:
        dyn_state = {
            "means": _to_torch(dyn_frames[left_global]["means"], device_t),
            "rgb": _to_torch(dyn_frames[left_global]["rgb"], device_t),
            "opacity": _to_torch(dyn_frames[left_global]["opacity"], device_t),
            "scales": _to_torch(dyn_frames[left_global]["scales"], device_t),
            "quats": _to_torch(dyn_frames[left_global]["quats"], device_t),
        }
    else:
        if right_global == left_global + 1 and left_global < len(package["dynamic_motions"]):
            motion = package["dynamic_motions"][left_global]
        else:
            motion = _build_one_motion_link(dyn_frames[left_global], dyn_frames[right_global], motion_radius)
        dyn_state = _dynamic_interpolate(dyn_frames[left_global], dyn_frames[right_global], motion, alpha, device_t)

    fg_means = torch.cat([static_means, dyn_state["means"]], dim=0)
    fg_rgb = torch.cat([static_rgb, dyn_state["rgb"]], dim=0)
    fg_opacity = torch.cat([static_opacity, dyn_state["opacity"]], dim=0)
    fg_scales = torch.cat([static_scales, dyn_state["scales"]], dim=0)
    fg_quats = torch.cat([static_quats, dyn_state["quats"]], dim=0)

    sky = package["sky"]
    sky_means = _to_torch(sky["means"], device_t)
    sky_rgb = _to_torch(sky["rgb"], device_t)
    sky_opacity = _to_torch(sky["opacity"], device_t)
    sky_scales = _to_torch(sky["scales"], device_t)
    sky_quats = _to_torch(sky["quats"], device_t)

    ext_t = _to_torch(ext_np, device_t)
    intr_t = _to_torch(intr_np, device_t)

    h = int(cameras["image_hw"][0])
    w = int(cameras["image_hw"][1])

    fg_rgbd, fg_alpha, _ = _rasterize_gaussians(
        means=fg_means,
        quats=_ensure_quat_normalized(fg_quats),
        scales=fg_scales,
        opacities=fg_opacity,
        colors=fg_rgb,
        viewmat=ext_t,
        intrinsic=intr_t,
        width=w,
        height=h,
    )

    sky_rgbd, _, _ = _rasterize_gaussians(
        means=sky_means,
        quats=_ensure_quat_normalized(sky_quats),
        scales=sky_scales,
        opacities=sky_opacity,
        colors=sky_rgb,
        viewmat=ext_t,
        intrinsic=intr_t,
        width=w,
        height=h,
    )

    fg_color = fg_rgbd[..., :3]
    fg_depth = fg_rgbd[..., 3]
    sky_color = sky_rgbd[..., :3]
    sky_depth = sky_rgbd[..., 3]

    composed = fg_alpha * fg_color + (1 - fg_alpha) * sky_color
    composed = composed[0].permute(2, 0, 1).clamp(0, 1)

    out: Dict[str, Any] = {
        "image": composed.detach().cpu(),
        "time": float(t),
        "extrinsic_w2c": ext_np.astype(np.float32),
        "intrinsic": intr_np.astype(np.float32),
        "alpha": fg_alpha[0, ..., 0].detach().cpu(),
    }

    if return_depth:
        depth = torch.where(fg_alpha[..., 0] > 0, fg_depth, sky_depth)
        out["depth"] = depth[0].detach().cpu()

    return out
