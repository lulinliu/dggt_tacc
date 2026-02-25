import argparse
import os
from typing import List

import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from datasets.dataset import WaymoOpenDataset
from dggt.models.vggt import VGGT
from dggt.utils.pose_enc import pose_encoding_to_extri_intri
from utils.scene_package import (
    build_dynamic_motion_knn,
    extract_scene_state,
    load_scene_package,
    render_scene_at_time,
    save_scene_package,
    validate_scene_package,
)


def parse_scene_names(scene_names_str: str) -> List[str]:
    scene_names_str = scene_names_str.strip()
    if scene_names_str.startswith("(") and scene_names_str.endswith(")"):
        start, end = scene_names_str[1:-1].split(",")
        return [str(i).zfill(3) for i in range(int(start), int(end) + 1)]
    return [str(int(x)).zfill(3) for x in scene_names_str.split()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export DGGT simulator-ready 4D scene packages")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the input images")
    parser.add_argument(
        "--scene_names",
        type=str,
        nargs="+",
        required=True,
        help="Scene names, supports formats like: 3 5 7 or (3,7)",
    )
    parser.add_argument("--input_views", type=int, default=1, help="Number of input views")
    parser.add_argument("--sequence_length", type=int, default=4, help="Number of input frames")
    parser.add_argument("--start_idx", type=int, default=0, help="Starting frame index")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], required=True, help="Processing mode")
    parser.add_argument("--ckpt_path", type=str, required=True, help="Path to model weights")
    parser.add_argument("--output_path", type=str, required=True, help="Directory for exported packages")

    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "float32"],
        default="float16",
        help="Attribute dtype for exported gaussian arrays",
    )
    parser.add_argument(
        "--motion_match_radius",
        type=float,
        default=1.0,
        help="Radius threshold for dynamic KNN motion linking",
    )
    parser.add_argument(
        "--lifespan_gamma1",
        type=float,
        default=0.1,
        help="Global gamma1 used in lifespan function",
    )
    parser.add_argument(
        "--save_reference_renders",
        action="store_true",
        help="Save keyframe RGB/depth renders from the exported package",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g. cuda:0 or cpu (default: auto)",
    )
    return parser.parse_args()


def _save_reference_renders(package_dir: str, package: dict, device: str) -> None:
    ref_dir = os.path.join(package_dir, "meta", "reference")
    os.makedirs(ref_dir, exist_ok=True)

    frame_times = package["cameras"]["frame_times"]
    view_index = package["cameras"]["view_index"]

    to_pil = T.ToPILImage()
    for frame_idx in range(frame_times.shape[0]):
        view_id = int(view_index[frame_idx])
        render_out = render_scene_at_time(
            package,
            frame_idx=frame_idx,
            camera_view=view_id,
            device=device,
            return_depth=True,
        )
        image = render_out["image"].clamp(0, 1)
        depth = render_out["depth"].numpy()

        rgb_path = os.path.join(ref_dir, f"frame_{frame_idx:06d}.png")
        depth_path = os.path.join(ref_dir, f"depth_{frame_idx:06d}.npy")

        to_pil(image).save(rgb_path)
        np.save(depth_path, depth)


def main() -> None:
    args = parse_args()

    if args.mode != 2:
        raise ValueError("v1 supports mode 2 only")

    if not args.image_dir:
        raise ValueError("--image_dir is empty. Provide a valid dataset root path.")
    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(f"--image_dir does not exist or is not a directory: {args.image_dir}")

    os.makedirs(args.output_path, exist_ok=True)

    if args.device is not None:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    scene_names = parse_scene_names(" ".join(args.scene_names))
    print(
        f"[Export] start image_dir={args.image_dir} scenes={scene_names} "
        f"output_path={args.output_path} mode={args.mode}",
        flush=True,
    )

    dataset = WaymoOpenDataset(
        args.image_dir,
        scene_names=scene_names,
        sequence_length=args.sequence_length,
        start_idx=args.start_idx,
        mode=args.mode,
        views=args.input_views,
    )
    missing_scene_names = [s for s in scene_names if s not in dataset.scenes]
    if missing_scene_names:
        print(
            f"[Export][Warn] requested scenes not discovered: {missing_scene_names}. "
            "Expected layout: <image_dir>/<scene_id>/images and <image_dir>/<scene_id>/sky_masks",
            flush=True,
        )
    if len(dataset) == 0:
        raise RuntimeError(
            "No scenes were loaded. "
            f"Resolved image_dir={args.image_dir}, requested scenes={scene_names}. "
            "Expected folder layout: <image_dir>/<scene_id>/images and <image_dir>/<scene_id>/sky_masks"
        )
    print(f"[Export] discovered_scenes={dataset.scenes} total={len(dataset)}", flush=True)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    model = VGGT().to(device)
    checkpoint = torch.load(args.ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint, strict=True)
    model.eval()

    with torch.no_grad():
        for data_idx, batch in enumerate(dataloader):
            print(f"[Export] processing index={data_idx + 1}/{len(dataset)}", flush=True)
            images = batch["images"].to(device)
            sky_mask = batch["masks"].to(device).permute(0, 1, 3, 4, 2)
            bg_mask = (sky_mask == 0).any(dim=-1)
            timestamps = batch["timestamps"][0].to(device)

            h, w = images.shape[-2:]
            predictions = model(images)
            extrinsics, intrinsics = pose_encoding_to_extri_intri(predictions["pose_enc"], (h, w))
            extrinsic = extrinsics[0]
            bottom = (
                torch.tensor([0.0, 0.0, 0.0, 1.0], device=extrinsic.device)
                .view(1, 1, 4)
                .expand(extrinsic.shape[0], 1, 4)
            )
            extrinsic = torch.cat([extrinsic, bottom], dim=1)
            intrinsic = intrinsics[0]

            scene_state = extract_scene_state(
                model=model,
                images=images,
                predictions=predictions,
                extrinsic_w2c=extrinsic,
                intrinsic=intrinsic,
                timestamps=timestamps,
                bg_mask=bg_mask,
                input_views=args.input_views,
            )
            motion_links = build_dynamic_motion_knn(scene_state["dynamic_frames"], args.motion_match_radius)

            if data_idx < len(dataset.scenes):
                scene_name = dataset.scenes[data_idx]
            else:
                scene_name = f"{data_idx + 1:03d}"

            package_dir = os.path.join(args.output_path, f"{scene_name}.dggt4d")
            save_scene_package(
                package_dir=package_dir,
                scene_name=scene_name,
                scene_state=scene_state,
                motion_links=motion_links,
                dtype=args.dtype,
                mode=args.mode,
                lifespan_gamma1=args.lifespan_gamma1,
                motion_match_radius=args.motion_match_radius,
                input_views=args.input_views,
            )

            package = load_scene_package(package_dir)
            issues = validate_scene_package(package)
            if issues:
                formatted = "\n".join([f"  - {x}" for x in issues])
                raise RuntimeError(f"Export validation failed for scene {scene_name}:\n{formatted}")

            if args.save_reference_renders:
                _save_reference_renders(package_dir, package, device)

            print(
                f"[Export] scene={scene_name} package={package_dir} "
                f"frames={len(scene_state['dynamic_frames'])} "
                f"static={scene_state['static']['means'].shape[0]} "
                f"sky={scene_state['sky']['means'].shape[0]}",
                flush=True,
            )


if __name__ == "__main__":
    main()
