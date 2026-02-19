#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import numpy as np
import pandas as pd


CAMID_TO_NAME = {
    0: "camera_front_wide_120fov",
    1: "camera_cross_left_120fov",
    2: "camera_cross_right_120fov",
    3: "camera_rear_left_70fov",
    4: "camera_rear_right_70fov",
}

NAME_RE = re.compile(r"^(\d+)_([0-9]+)\.(png|jpg|jpeg)$", re.IGNORECASE)


@dataclass
class CamIntrinsics:
    width: int
    height: int
    cx: float
    cy: float
    fw: np.ndarray  # (5,)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rectify one DGGT scene using kb_poly fw coefficients.")
    parser.add_argument(
        "--src-root",
        type=Path,
        default=Path("/DATA2/lulin2/ood/train_data_dggt_v3"),
        help="Source dataset root containing <scene_id>/images.",
    )
    parser.add_argument(
        "--scene-id",
        type=str,
        required=True,
        help="UUID scene id to copy and rectify.",
    )
    parser.add_argument(
        "--calib-parquet",
        type=Path,
        default=Path("/DATA2/lulin2/ood/train_data_dggt_v3_calibration_from_chunks/matched_camera_intrinsics_dedup.parquet"),
        help="Parquet with clip_id, camera_name, cx/cy and fw_poly_* columns.",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/DATA2/lulin2/ood/train_data_dggt_v3_rectify_preview"),
        help="Output root.",
    )
    parser.add_argument(
        "--cam-ids",
        type=str,
        default="0,1,2",
        help="Camera ids to rectify, comma separated.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="If > 0, rectify only first N frames per camera.",
    )
    parser.add_argument(
        "--focal-scale",
        type=float,
        default=1.0,
        help="Scale factor for rectified focal (base is fw_poly_1).",
    )
    parser.add_argument(
        "--copy-sky-masks",
        action="store_true",
        help="Copy sky masks and also output rectified sky masks.",
    )
    parser.add_argument(
        "--copy-raw-scene",
        action="store_true",
        help="Copy full source scene images to <out>/<scene>/images_raw before rectification.",
    )
    parser.add_argument(
        "--principal-point-mode",
        type=str,
        default="source",
        choices=["source", "center"],
        help="Rectified principal point: use source cx/cy or image center.",
    )
    parser.add_argument(
        "--auto-crop",
        action="store_true",
        help="Also output cropped images without black borders.",
    )
    parser.add_argument(
        "--crop-margin",
        type=int,
        default=2,
        help="Shrink auto-crop rectangle by this many pixels per side.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Worker threads per scene (0=auto, 1=single-thread).",
    )
    return parser.parse_args()


def load_intrinsics_map(parquet_path: Path, scene_id: str) -> Dict[str, CamIntrinsics]:
    df = pd.read_parquet(parquet_path)
    need_cols = {
        "clip_id",
        "camera_name",
        "width",
        "height",
        "cx",
        "cy",
        "fw_poly_0",
        "fw_poly_1",
        "fw_poly_2",
        "fw_poly_3",
        "fw_poly_4",
    }
    miss = sorted(need_cols - set(df.columns))
    if miss:
        raise ValueError(f"Missing columns in {parquet_path}: {miss}")

    sdf = df[df["clip_id"].astype(str) == scene_id].copy()
    if sdf.empty:
        raise ValueError(f"scene_id {scene_id} not found in {parquet_path}")

    out: Dict[str, CamIntrinsics] = {}
    for _, r in sdf.iterrows():
        cam_name = str(r["camera_name"])
        out[cam_name] = CamIntrinsics(
            width=int(r["width"]),
            height=int(r["height"]),
            cx=float(r["cx"]),
            cy=float(r["cy"]),
            fw=np.array(
                [
                    float(r["fw_poly_0"]),
                    float(r["fw_poly_1"]),
                    float(r["fw_poly_2"]),
                    float(r["fw_poly_3"]),
                    float(r["fw_poly_4"]),
                ],
                dtype=np.float64,
            ),
        )
    return out


def poly_eval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    # coeffs are [c0, c1, ...], matching fw_poly_i convention.
    y = np.zeros_like(x, dtype=np.float64)
    p = np.ones_like(x, dtype=np.float64)
    for c in coeffs:
        y += c * p
        p *= x
    return y


def build_map(
    width: int,
    height: int,
    cx_src: float,
    cy_src: float,
    fw: np.ndarray,
    fx_rect: float,
    fy_rect: float,
    cx_rect: float,
    cy_rect: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    uu, vv = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    x = (uu - cx_rect) / fx_rect
    y = (vv - cy_rect) / fy_rect
    r = np.sqrt(x * x + y * y)
    theta = np.arctan(r)
    rho = poly_eval(fw, theta)

    scale = np.empty_like(r, dtype=np.float64)
    nz = r > 1e-9
    scale[nz] = rho[nz] / r[nz]
    scale[~nz] = fw[1]

    map_x = (cx_src + x * scale).astype(np.float32)
    map_y = (cy_src + y * scale).astype(np.float32)
    valid = (
        (map_x >= 0.0)
        & (map_x <= width - 1.0)
        & (map_y >= 0.0)
        & (map_y <= height - 1.0)
    )
    return map_x, map_y, valid


def parse_camid(path: Path) -> int:
    m = NAME_RE.match(path.name)
    if m is None:
        raise ValueError(f"Unexpected file name format: {path.name}")
    return int(m.group(2))


def frame_key(path: Path) -> int:
    m = NAME_RE.match(path.name)
    if m is None:
        return 10**12
    return int(m.group(1))


def select_paths(paths: List[Path], max_frames: int) -> List[Path]:
    if max_frames <= 0:
        return paths
    by_frame: Dict[int, List[Path]] = {}
    for p in paths:
        by_frame.setdefault(frame_key(p), []).append(p)
    keep_frames = sorted(by_frame.keys())[:max_frames]
    out: List[Path] = []
    for k in keep_frames:
        out.extend(sorted(by_frame[k]))
    return out


def copy_scene(src_scene: Path, dst_scene: Path, copy_sky_masks: bool) -> None:
    dst_scene.mkdir(parents=True, exist_ok=True)
    src_images = src_scene / "images"
    if not src_images.is_dir():
        raise ValueError(f"Missing images dir: {src_images}")
    dst_images_raw = dst_scene / "images_raw"
    if dst_images_raw.exists():
        shutil.rmtree(dst_images_raw)
    shutil.copytree(src_images, dst_images_raw)

    if copy_sky_masks:
        src_sky = src_scene / "sky_masks"
        if not src_sky.is_dir():
            src_sky = src_scene / "skymasks"
        if src_sky.is_dir():
            dst_sky_raw = dst_scene / "sky_masks_raw"
            if dst_sky_raw.exists():
                shutil.rmtree(dst_sky_raw)
            shutil.copytree(src_sky, dst_sky_raw)


def largest_valid_rectangle(valid: np.ndarray) -> Tuple[int, int, int, int]:
    h, w = valid.shape
    heights = np.zeros(w, dtype=np.int32)
    best = (0, 0, 0, 0)
    best_area = 0

    for y in range(h):
        row = valid[y]
        heights = np.where(row, heights + 1, 0)
        stack: List[Tuple[int, int]] = []  # (start_x, height)

        for x in range(w + 1):
            cur_h = int(heights[x]) if x < w else 0
            start = x
            while stack and stack[-1][1] > cur_h:
                left, hh = stack.pop()
                area = hh * (x - left)
                if area > best_area:
                    best_area = area
                    y1 = y
                    y0 = y - hh + 1
                    x0 = left
                    x1 = x - 1
                    best = (x0, y0, x1, y1)
                start = left
            if not stack or stack[-1][1] < cur_h:
                stack.append((start, cur_h))

    return best


def shrink_rect(
    rect: Tuple[int, int, int, int],
    width: int,
    height: int,
    margin: int,
) -> Tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    x0 = max(0, x0 + margin)
    y0 = max(0, y0 + margin)
    x1 = min(width - 1, x1 - margin)
    y1 = min(height - 1, y1 - margin)
    if x1 < x0 or y1 < y0:
        return 0, 0, width - 1, height - 1
    return x0, y0, x1, y1


def resolve_num_workers(num_workers: int) -> int:
    if num_workers > 0:
        return num_workers
    cpu = os.cpu_count() or 1
    return max(1, min(32, cpu))


def rectify_scene(
    scene_id: str,
    dst_scene: Path,
    input_images_dir: Path,
    input_sky_dir: Path | None,
    intr_map: Dict[str, CamIntrinsics],
    cam_ids: Iterable[int],
    max_frames: int,
    focal_scale: float,
    rectify_sky_masks: bool,
    auto_crop: bool,
    crop_margin: int,
    principal_point_mode: str,
    num_workers: int,
) -> Dict[str, dict]:
    images_raw = input_images_dir
    if not images_raw.is_dir():
        raise ValueError(f"Input images dir not found: {images_raw}")
    image_files = sorted(
        [p for p in images_raw.iterdir() if p.is_file() and NAME_RE.match(p.name)],
        key=lambda p: (frame_key(p), p.name),
    )
    image_files = [p for p in image_files if parse_camid(p) in cam_ids]
    image_files = select_paths(image_files, max_frames=max_frames)

    if len(image_files) == 0:
        raise ValueError(f"No image files selected in {images_raw}")

    num_workers = resolve_num_workers(num_workers)

    images_rect = dst_scene / "images_rectified"
    if images_rect.exists():
        shutil.rmtree(images_rect)
    images_rect.mkdir(parents=True, exist_ok=True)

    images_crop = dst_scene / "images_cropped"
    if auto_crop:
        if images_crop.exists():
            shutil.rmtree(images_crop)
        images_crop.mkdir(parents=True, exist_ok=True)

    preview_dir = dst_scene / "preview"
    if preview_dir.exists():
        shutil.rmtree(preview_dir)
    preview_dir.mkdir(parents=True, exist_ok=True)

    sky_raw = input_sky_dir if input_sky_dir is not None else Path("")
    sky_rect = dst_scene / "sky_masks_rectified"
    has_sky = rectify_sky_masks and sky_raw.is_dir()
    if has_sky:
        if sky_rect.exists():
            shutil.rmtree(sky_rect)
        sky_rect.mkdir(parents=True, exist_ok=True)
    sky_crop = dst_scene / "sky_masks_cropped"
    if has_sky and auto_crop:
        if sky_crop.exists():
            shutil.rmtree(sky_crop)
        sky_crop.mkdir(parents=True, exist_ok=True)

    remap_cache: Dict[
        int,
        Tuple[np.ndarray, np.ndarray, np.ndarray, CamIntrinsics, float, float, float, float, Tuple[int, int, int, int]],
    ] = {}
    stats: Dict[str, dict] = {}

    for camid in sorted(set(parse_camid(p) for p in image_files)):
        cam_name = CAMID_TO_NAME.get(camid)
        if cam_name is None or cam_name not in intr_map:
            raise ValueError(f"Missing intrinsics for camid={camid}, cam_name={cam_name}")
        intr = intr_map[cam_name]
        fx_rect = intr.fw[1] * focal_scale
        fy_rect = intr.fw[1] * focal_scale
        if principal_point_mode == "source":
            cx_rect = intr.cx
            cy_rect = intr.cy
        elif principal_point_mode == "center":
            cx_rect = (intr.width - 1) / 2.0
            cy_rect = (intr.height - 1) / 2.0
        else:
            raise ValueError(f"Unsupported principal_point_mode: {principal_point_mode}")
        map_x, map_y, valid = build_map(
            width=intr.width,
            height=intr.height,
            cx_src=intr.cx,
            cy_src=intr.cy,
            fw=intr.fw,
            fx_rect=fx_rect,
            fy_rect=fy_rect,
            cx_rect=cx_rect,
            cy_rect=cy_rect,
        )
        # For linear interpolation, require one-pixel neighbor margin to avoid border fill.
        strict_valid = (
            (map_x >= 1.0)
            & (map_x <= intr.width - 2.0)
            & (map_y >= 1.0)
            & (map_y <= intr.height - 2.0)
        )
        crop_rect = shrink_rect(
            largest_valid_rectangle(strict_valid),
            width=intr.width,
            height=intr.height,
            margin=max(0, int(crop_margin)),
        )
        remap_cache[camid] = (map_x, map_y, valid, intr, fx_rect, fy_rect, cx_rect, cy_rect, crop_rect)
        x0, y0, x1, y1 = crop_rect
        stats[str(camid)] = {
            "camera_name": cam_name,
            "fx_rect": float(fx_rect),
            "fy_rect": float(fy_rect),
            "cx_rect": float(cx_rect),
            "cy_rect": float(cy_rect),
            "valid_ratio": float(valid.mean()),
            "no_black_crop_xyxy": [x0, y0, x1, y1],
            "no_black_crop_wh": [max(0, x1 - x0 + 1), max(0, y1 - y0 + 1)],
        }

    first_img_per_cam: Dict[int, Path] = {}
    for p in image_files:
        first_img_per_cam.setdefault(parse_camid(p), p)

    def _process_one_image(p: Path) -> None:
        camid = parse_camid(p)
        map_x, map_y, _, intr, *_rest, crop_rect = remap_cache[camid]
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"Failed to read image: {p}")
        if img.shape[1] != intr.width or img.shape[0] != intr.height:
            raise ValueError(
                f"Image shape mismatch for {p.name}: got {img.shape[1]}x{img.shape[0]}, "
                f"expected {intr.width}x{intr.height}"
            )
        rect = cv2.remap(
            img,
            map_x,
            map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        cv2.imwrite(str(images_rect / p.name), rect)

        if auto_crop:
            x0, y0, x1, y1 = crop_rect
            crop = rect[y0 : y1 + 1, x0 : x1 + 1]
            cv2.imwrite(str(images_crop / p.name), crop)

    if num_workers <= 1 or len(image_files) <= 1:
        for p in image_files:
            _process_one_image(p)
    else:
        with ThreadPoolExecutor(max_workers=min(num_workers, len(image_files))) as ex:
            futures = [ex.submit(_process_one_image, p) for p in image_files]
            for fut in as_completed(futures):
                fut.result()

    for camid in sorted(first_img_per_cam.keys()):
        p = first_img_per_cam[camid]
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        rect = cv2.imread(str(images_rect / p.name), cv2.IMREAD_COLOR)
        if img is None or rect is None:
            continue
        if auto_crop:
            x0, y0, x1, y1 = remap_cache[camid][-1]
            crop = rect[y0 : y1 + 1, x0 : x1 + 1]
            crop_resized = cv2.resize(
                crop,
                (img.shape[1], img.shape[0]),
                interpolation=cv2.INTER_LINEAR,
            )
            compare = np.concatenate([img, rect, crop_resized], axis=1)
        else:
            compare = np.concatenate([img, rect], axis=1)
        cv2.putText(compare, "raw", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            compare,
            "rectified",
            (img.shape[1] + 24, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        if auto_crop:
            cv2.putText(
                compare,
                "cropped(resized)",
                (img.shape[1] * 2 + 24, 48),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
        cv2.imwrite(str(preview_dir / f"compare_cam{camid}_{p.name}"), compare)

    if has_sky:
        sky_files = sorted(
            [p for p in sky_raw.iterdir() if p.is_file() and NAME_RE.match(p.name)],
            key=lambda p: (frame_key(p), p.name),
        )
        sky_files = [p for p in sky_files if parse_camid(p) in cam_ids]
        sky_files = select_paths(sky_files, max_frames=max_frames)
        def _process_one_sky(p: Path) -> None:
            camid = parse_camid(p)
            map_x, map_y, *_rest, crop_rect = remap_cache[camid]
            sk = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if sk is None:
                return
            sk_rect = cv2.remap(
                sk,
                map_x,
                map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
            cv2.imwrite(str(sky_rect / p.name), sk_rect)
            if auto_crop:
                x0, y0, x1, y1 = crop_rect
                sk_crop = sk_rect[y0 : y1 + 1, x0 : x1 + 1]
                cv2.imwrite(str(sky_crop / p.name), sk_crop)

        if num_workers <= 1 or len(sky_files) <= 1:
            for p in sky_files:
                _process_one_sky(p)
        else:
            with ThreadPoolExecutor(max_workers=min(num_workers, len(sky_files))) as ex:
                futures = [ex.submit(_process_one_sky, p) for p in sky_files]
                for fut in as_completed(futures):
                    fut.result()

    meta = {
        "scene_id": scene_id,
        "num_images_rectified": len(image_files),
        "num_workers": int(num_workers),
        "cam_ids": sorted(int(x) for x in cam_ids),
        "focal_scale": float(focal_scale),
        "auto_crop": bool(auto_crop),
        "crop_margin": int(crop_margin),
        "principal_point_mode": principal_point_mode,
        "per_camera": stats,
    }
    with open(dst_scene / "rectify_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def main() -> None:
    args = parse_args()
    cam_ids = [int(x) for x in args.cam_ids.split(",") if x.strip()]
    if len(cam_ids) == 0:
        raise ValueError("No camera ids parsed from --cam-ids")

    src_scene = args.src_root / args.scene_id
    if not src_scene.is_dir():
        raise ValueError(f"Scene not found: {src_scene}")

    intr_map = load_intrinsics_map(args.calib_parquet, args.scene_id)
    dst_scene = args.out_root / args.scene_id

    if args.copy_raw_scene:
        copy_scene(src_scene, dst_scene, copy_sky_masks=args.copy_sky_masks)
        input_images_dir = dst_scene / "images_raw"
        input_sky_dir = dst_scene / "sky_masks_raw" if args.copy_sky_masks else None
    else:
        dst_scene.mkdir(parents=True, exist_ok=True)
        input_images_dir = src_scene / "images"
        if args.copy_sky_masks:
            src_sky = src_scene / "sky_masks"
            if not src_sky.is_dir():
                src_sky = src_scene / "skymasks"
            input_sky_dir = src_sky if src_sky.is_dir() else None
        else:
            input_sky_dir = None

    meta = rectify_scene(
        scene_id=args.scene_id,
        dst_scene=dst_scene,
        input_images_dir=input_images_dir,
        input_sky_dir=input_sky_dir,
        intr_map=intr_map,
        cam_ids=cam_ids,
        max_frames=args.max_frames,
        focal_scale=args.focal_scale,
        rectify_sky_masks=args.copy_sky_masks,
        auto_crop=args.auto_crop,
        crop_margin=args.crop_margin,
        principal_point_mode=args.principal_point_mode,
        num_workers=args.num_workers,
    )

    print(json.dumps(meta, indent=2))
    print(f"\nOutput scene: {dst_scene}")
    print(f"Preview dir : {dst_scene / 'preview'}")
    print(f"Meta file   : {dst_scene / 'rectify_meta.json'}")


if __name__ == "__main__":
    main()
