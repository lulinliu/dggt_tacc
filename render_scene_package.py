import argparse
import os
from typing import Optional

import imageio
import numpy as np
import torch
import torchvision.transforms as T

from utils.scene_package import load_scene_package, render_scene_at_time, validate_scene_package


def _load_matrix(path: str, expected_shape: tuple) -> np.ndarray:
    if path.endswith(".npy"):
        mat = np.load(path)
    else:
        mat = np.loadtxt(path)

    mat = np.asarray(mat, dtype=np.float32)

    if expected_shape == (4, 4) and mat.shape == (3, 4):
        out = np.eye(4, dtype=np.float32)
        out[:3, :] = mat
        mat = out

    if mat.shape != expected_shape:
        raise ValueError(f"Matrix at {path} has shape {mat.shape}, expected {expected_shape}")

    return mat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render from DGGT 4D scene package")
    parser.add_argument("--package_dir", type=str, required=True, help="Path to <scene>.dggt4d folder")

    parser.add_argument("--time", type=float, default=None, help="Normalized time query in [0,1]")
    parser.add_argument("--frame_idx", type=int, default=None, help="Exact exported frame index query")

    parser.add_argument("--camera_view", type=int, default=0, help="Camera view id to use from package")
    parser.add_argument(
        "--camera_extrinsic",
        type=str,
        default=None,
        help="Optional camera extrinsic override (.npy or .txt, 4x4 or 3x4)",
    )
    parser.add_argument(
        "--camera_intrinsic",
        type=str,
        default=None,
        help="Optional camera intrinsic override (.npy or .txt, 3x3)",
    )

    parser.add_argument("--output_image", type=str, default=None, help="Output image path (png/jpg)")
    parser.add_argument("--output_depth", type=str, default=None, help="Optional output depth .npy path")

    parser.add_argument("--output_video", type=str, default=None, help="Output video path (mp4)")
    parser.add_argument("--times", type=float, nargs="+", default=None, help="Explicit query times for video")
    parser.add_argument(
        "--num_video_frames",
        type=int,
        default=60,
        help="Number of evenly spaced times if --times is omitted",
    )
    parser.add_argument("--fps", type=int, default=8, help="Video fps")

    parser.add_argument("--gamma1", type=float, default=None, help="Optional lifespan gamma1 override")
    parser.add_argument(
        "--motion_match_radius",
        type=float,
        default=None,
        help="Optional dynamic matching radius override",
    )
    parser.add_argument("--device", type=str, default=None, help="Device override, e.g. cuda:0 or cpu")

    return parser.parse_args()


def _to_uint8_rgb(image_chw: torch.Tensor) -> np.ndarray:
    image = image_chw.clamp(0, 1).permute(1, 2, 0).numpy()
    return (image * 255.0).astype(np.uint8)


def _render_single(
    package: dict,
    args: argparse.Namespace,
    time_query: Optional[float],
    frame_idx: Optional[int],
) -> dict:
    extrinsic_override = None
    intrinsic_override = None

    if args.camera_extrinsic is not None:
        extrinsic_override = _load_matrix(args.camera_extrinsic, (4, 4))
    if args.camera_intrinsic is not None:
        intrinsic_override = _load_matrix(args.camera_intrinsic, (3, 3))

    return render_scene_at_time(
        package=package,
        query_time=time_query,
        frame_idx=frame_idx,
        camera_view=args.camera_view,
        extrinsic_override=extrinsic_override,
        intrinsic_override=intrinsic_override,
        gamma1_override=args.gamma1,
        match_radius_override=args.motion_match_radius,
        device=args.device,
        return_depth=True,
    )


def main() -> None:
    args = parse_args()

    package = load_scene_package(args.package_dir)
    issues = validate_scene_package(package)
    if issues:
        formatted = "\n".join([f"  - {x}" for x in issues])
        raise RuntimeError(f"Invalid package at {args.package_dir}:\n{formatted}")

    if args.output_image is None and args.output_video is None:
        raise ValueError("Specify at least one output target: --output_image or --output_video")

    os.makedirs(os.path.dirname(args.output_image), exist_ok=True) if args.output_image and os.path.dirname(args.output_image) else None
    os.makedirs(os.path.dirname(args.output_video), exist_ok=True) if args.output_video and os.path.dirname(args.output_video) else None
    os.makedirs(os.path.dirname(args.output_depth), exist_ok=True) if args.output_depth and os.path.dirname(args.output_depth) else None

    if args.output_image is not None:
        if args.frame_idx is None and args.time is None:
            raise ValueError("For --output_image, provide --frame_idx or --time")

        out = _render_single(
            package=package,
            args=args,
            time_query=args.time,
            frame_idx=args.frame_idx,
        )

        to_pil = T.ToPILImage()
        to_pil(out["image"]).save(args.output_image)
        if args.output_depth is not None:
            np.save(args.output_depth, out["depth"].numpy())

        print(
            f"[Render] image={args.output_image} time={out['time']:.6f} "
            f"view={args.camera_view}"
        )

    if args.output_video is not None:
        frame_times = package["cameras"]["frame_times"].astype(np.float32)
        t_min = float(frame_times.min())
        t_max = float(frame_times.max())

        if args.times is not None and len(args.times) > 0:
            times = [float(np.clip(t, t_min, t_max)) for t in args.times]
        else:
            n = max(2, int(args.num_video_frames))
            times = np.linspace(t_min, t_max, n).tolist()

        video_frames = []
        for t in times:
            out = _render_single(package=package, args=args, time_query=t, frame_idx=None)
            video_frames.append(_to_uint8_rgb(out["image"]))

        imageio.mimwrite(args.output_video, np.asarray(video_frames), fps=args.fps, codec="libx264")
        print(
            f"[Render] video={args.output_video} frames={len(video_frames)} "
            f"time_range=[{times[0]:.6f}, {times[-1]:.6f}] view={args.camera_view}"
        )


if __name__ == "__main__":
    main()
