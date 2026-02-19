#!/usr/bin/env bash
set -euo pipefail

# Bulk rectify for /DATA2/lulin2/ood/train_data_dggt_v3
#
# Default behavior:
# - Rectify images for cam 0/1/2 for ALL scenes into OUT_ROOT.
# - Rectify sky masks (if present) into OUT_ROOT.
# - Write per-scene rectified calibration parquet into OUT_ROOT.
# - DO NOT overwrite source dataset (OVERWRITE=0 by default).
#
# To overwrite source dataset in-place (images + sky masks + calibration), run:
#   OVERWRITE=1 bash rectify_train_data_dggt_v3_bulk.sh
#
# Notes:
# - This script intentionally keeps a hard safety switch: OVERWRITE defaults to 0.
# - It uses rectify_one_scene_kb.py and calibration parquet/chunks.
# - Set USE_INTERMEDIATE_ROOT=0 with OVERWRITE=1 for direct in-place overwrite mode.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ===== Required paths =====
SRC_ROOT="${SRC_ROOT:-/DATA2/lulin2/ood/train_data_dggt_v3}"
CALIB_PARQUET="${CALIB_PARQUET:-$SCRIPT_DIR/train_data_dggt_v3_calibration_from_chunks/matched_camera_intrinsics_dedup.parquet}"
CALIB_DIR="${CALIB_DIR:-}"               # optional: directory containing calibration parquet chunks/files
RECTIFY_SCRIPT="${RECTIFY_SCRIPT:-$SCRIPT_DIR/rectify_one_scene_kb.py}"
OUT_ROOT="${OUT_ROOT:-/DATA2/lulin2/ood/train_data_dggt_v3_rectified_all_nocrop}"

# ===== Rectify settings =====
CAM_IDS="${CAM_IDS:-0,1,2}"               # user request: cam 0/1/2
MAX_FRAMES="${MAX_FRAMES:-0}"             # 0 means all frames
FOCAL_SCALE="${FOCAL_SCALE:-1.0}"
PRINCIPAL_POINT_MODE="${PRINCIPAL_POINT_MODE:-source}"  # source | center
NUM_WORKERS_PER_SCENE="${NUM_WORKERS_PER_SCENE:-0}"     # 0 means auto in python script
COPY_SKY_MASKS="${COPY_SKY_MASKS:-1}"     # 1 to rectify sky masks
AUTO_CROP="${AUTO_CROP:-0}"               # keep 0 per latest request (no crop)
CROP_MARGIN="${CROP_MARGIN:-2}"
COPY_RAW_SCENE="${COPY_RAW_SCENE:-0}"     # keep 0 to avoid copying full raw scene

# ===== Scene selection =====
SCENE_LIMIT="${SCENE_LIMIT:-0}"           # 0 means all scenes
SCENE_FILTER_REGEX="${SCENE_FILTER_REGEX:-}"  # optional regex filter on scene_id

# ===== Overwrite switches (safety) =====
OVERWRITE="${OVERWRITE:-0}"               # 0: no in-place overwrite, 1: overwrite source
OVERWRITE_IMAGES="${OVERWRITE_IMAGES:-1}"
OVERWRITE_SKY_MASKS="${OVERWRITE_SKY_MASKS:-1}"
OVERWRITE_CALIB="${OVERWRITE_CALIB:-1}"
USE_INTERMEDIATE_ROOT="${USE_INTERMEDIATE_ROOT:-1}"   # 1: write to OUT_ROOT first, 0: write temp artifacts inside SRC scene
KEEP_INPLACE_ARTIFACTS="${KEEP_INPLACE_ARTIFACTS:-0}" # only for USE_INTERMEDIATE_ROOT=0

if [[ ! -d "$SRC_ROOT" ]]; then
  echo "ERROR: SRC_ROOT not found: $SRC_ROOT" >&2
  exit 1
fi
if [[ ! -f "$RECTIFY_SCRIPT" ]]; then
  echo "ERROR: RECTIFY_SCRIPT not found: $RECTIFY_SCRIPT" >&2
  exit 1
fi

if [[ "$USE_INTERMEDIATE_ROOT" != "0" && "$USE_INTERMEDIATE_ROOT" != "1" ]]; then
  echo "ERROR: USE_INTERMEDIATE_ROOT must be 0 or 1, got: $USE_INTERMEDIATE_ROOT" >&2
  exit 1
fi
if [[ "$USE_INTERMEDIATE_ROOT" == "0" && "$OVERWRITE" != "1" ]]; then
  echo "ERROR: USE_INTERMEDIATE_ROOT=0 requires OVERWRITE=1" >&2
  exit 1
fi

CLEAN_META_DIR=0
if [[ "$USE_INTERMEDIATE_ROOT" == "1" ]]; then
  mkdir -p "$OUT_ROOT"
  WORK_ROOT="$OUT_ROOT"
  META_DIR="$WORK_ROOT/_meta"
  mkdir -p "$META_DIR"
else
  WORK_ROOT="$SRC_ROOT"
  META_DIR="$(mktemp -d /tmp/rectify_bulk_meta.XXXXXX)"
  CLEAN_META_DIR=1
fi

cleanup_meta_dir() {
  if [[ "${CLEAN_META_DIR:-0}" == "1" && -n "${META_DIR:-}" && -d "${META_DIR:-}" ]]; then
    rm -rf "$META_DIR"
  fi
}
trap cleanup_meta_dir EXIT

echo "[config] SRC_ROOT=$SRC_ROOT"
echo "[config] OUT_ROOT=$OUT_ROOT"
echo "[config] WORK_ROOT=$WORK_ROOT"
echo "[config] CALIB_PARQUET=$CALIB_PARQUET"
echo "[config] CALIB_DIR=${CALIB_DIR:-<unset>}"
echo "[config] CAM_IDS=$CAM_IDS MAX_FRAMES=$MAX_FRAMES PRINCIPAL_POINT_MODE=$PRINCIPAL_POINT_MODE"
echo "[config] NUM_WORKERS_PER_SCENE=$NUM_WORKERS_PER_SCENE"
echo "[config] COPY_SKY_MASKS=$COPY_SKY_MASKS AUTO_CROP=$AUTO_CROP COPY_RAW_SCENE=$COPY_RAW_SCENE"
echo "[config] OVERWRITE=$OVERWRITE (images=$OVERWRITE_IMAGES sky=$OVERWRITE_SKY_MASKS calib=$OVERWRITE_CALIB)"
echo "[config] USE_INTERMEDIATE_ROOT=$USE_INTERMEDIATE_ROOT KEEP_INPLACE_ARTIFACTS=$KEEP_INPLACE_ARTIFACTS"

mapfile -t ALL_SCENES < <(find "$SRC_ROOT" -mindepth 1 -maxdepth 1 -type d -printf '%f\n' | sort)

SCENES=()
for sid in "${ALL_SCENES[@]}"; do
  if [[ -n "$SCENE_FILTER_REGEX" ]]; then
    if [[ ! "$sid" =~ $SCENE_FILTER_REGEX ]]; then
      continue
    fi
  fi
  SCENES+=("$sid")
done

if [[ "$SCENE_LIMIT" =~ ^[0-9]+$ ]] && (( SCENE_LIMIT > 0 )) && (( ${#SCENES[@]} > SCENE_LIMIT )); then
  SCENES=("${SCENES[@]:0:SCENE_LIMIT}")
fi

echo "[plan] scenes_to_process=${#SCENES[@]}"
if (( ${#SCENES[@]} == 0 )); then
  echo "[done] no scenes matched"
  exit 0
fi

# Resolve calibration source:
# 1) If CALIB_DIR is set, build a scene-filtered dedup parquet from that directory.
# 2) Else fallback to CALIB_PARQUET.
if [[ -n "$CALIB_DIR" ]]; then
  if [[ ! -d "$CALIB_DIR" && ! -f "$CALIB_DIR" ]]; then
    echo "ERROR: CALIB_DIR path not found: $CALIB_DIR" >&2
    exit 1
  fi

  SCENE_LIST_FILE="$META_DIR/scenes_to_process.txt"
  printf "%s\n" "${SCENES[@]}" > "$SCENE_LIST_FILE"

  RESOLVED_CALIB_PARQUET="$META_DIR/calibration_from_dir_dedup.parquet"
  echo "[calib] building parquet from CALIB_DIR -> $RESOLVED_CALIB_PARQUET"
  python3 - "$CALIB_DIR" "$SCENE_LIST_FILE" "$RESOLVED_CALIB_PARQUET" <<'PY'
import sys
from pathlib import Path
import pandas as pd

calib_dir = Path(sys.argv[1])
scene_list_file = Path(sys.argv[2])
out_path = Path(sys.argv[3])

scene_set = set(x.strip() for x in scene_list_file.read_text(encoding="utf-8").splitlines() if x.strip())
if not scene_set:
    raise SystemExit("No scenes provided for calibration filtering.")

required = [
    "clip_id", "camera_name", "width", "height", "cx", "cy",
    "bw_poly_0", "bw_poly_1", "bw_poly_2", "bw_poly_3", "bw_poly_4",
    "fw_poly_0", "fw_poly_1", "fw_poly_2", "fw_poly_3", "fw_poly_4",
]

def candidate_files(root: Path):
    if root.is_file() and root.suffix == ".parquet":
        return [root]
    if not root.is_dir():
        return []

    c1 = sorted(root.glob("camera_intrinsics.chunk_*.parquet"))
    if c1:
        return c1
    c2 = sorted(root.glob("camera_intrinsics*.parquet"))
    if c2:
        return c2
    c3 = sorted(root.glob("**/camera_intrinsics.parquet"))
    if c3:
        return c3
    return sorted(root.glob("**/*.parquet"))

files = candidate_files(calib_dir)
if not files:
    raise SystemExit(f"No parquet files found in: {calib_dir}")

parts = []
scanned = 0
matched_files = 0
for f in files:
    scanned += 1
    try:
        df = pd.read_parquet(f)
    except Exception:
        continue
    if "clip_id" not in df.columns:
        df = df.reset_index()
    if "clip_id" not in df.columns or "camera_name" not in df.columns:
        continue

    df["clip_id"] = df["clip_id"].astype(str)
    sdf = df[df["clip_id"].isin(scene_set)].copy()
    if sdf.empty:
        continue
    matched_files += 1

    for col in required:
        if col not in sdf.columns:
            if col in ("clip_id", "camera_name"):
                raise SystemExit(f"{f}: missing required column {col}")
            sdf[col] = 0.0

    parts.append(sdf[required])

if not parts:
    raise SystemExit(f"No calibration rows matched requested scenes in {calib_dir}")

all_df = pd.concat(parts, ignore_index=True)
all_df = all_df.drop_duplicates(subset=["clip_id", "camera_name"], keep="first")
missing = scene_set - set(all_df["clip_id"].unique().tolist())
if missing:
    sample = sorted(list(missing))[:20]
    raise SystemExit(f"Missing calibration for {len(missing)} scenes. sample={sample}")

out_path.parent.mkdir(parents=True, exist_ok=True)
all_df.to_parquet(out_path, index=False)
print(f"[calib] scanned_files={scanned} matched_files={matched_files} rows={len(all_df)}")
print(f"[calib] wrote {out_path}")
PY
  CALIB_PARQUET="$RESOLVED_CALIB_PARQUET"
fi

if [[ ! -f "$CALIB_PARQUET" ]]; then
  echo "ERROR: resolved CALIB_PARQUET not found: $CALIB_PARQUET" >&2
  exit 1
fi
echo "[calib] using CALIB_PARQUET=$CALIB_PARQUET"

run_rectify_scene() {
  local sid="$1"
  local cmd=(
    python3 "$RECTIFY_SCRIPT"
    --src-root "$SRC_ROOT"
    --scene-id "$sid"
    --calib-parquet "$CALIB_PARQUET"
    --out-root "$WORK_ROOT"
    --cam-ids "$CAM_IDS"
    --max-frames "$MAX_FRAMES"
    --focal-scale "$FOCAL_SCALE"
    --principal-point-mode "$PRINCIPAL_POINT_MODE"
    --num-workers "$NUM_WORKERS_PER_SCENE"
  )
  if [[ "$COPY_SKY_MASKS" == "1" ]]; then
    cmd+=(--copy-sky-masks)
  fi
  if [[ "$AUTO_CROP" == "1" ]]; then
    cmd+=(--auto-crop --crop-margin "$CROP_MARGIN")
  fi
  if [[ "$COPY_RAW_SCENE" == "1" ]]; then
    cmd+=(--copy-raw-scene)
  fi
  "${cmd[@]}"
}

write_rectified_calib() {
  local sid="$1"
  local out_scene="$WORK_ROOT/$sid"
  local out_calib="$out_scene/calibration/camera_intrinsics.parquet"
  mkdir -p "$(dirname "$out_calib")"

  python3 - "$CALIB_PARQUET" "$sid" "$out_calib" "$CAM_IDS" "$FOCAL_SCALE" "$PRINCIPAL_POINT_MODE" <<'PY'
import sys
from pathlib import Path
import pandas as pd

calib_parquet = Path(sys.argv[1])
scene_id = sys.argv[2]
out_path = Path(sys.argv[3])
cam_ids = [int(x) for x in sys.argv[4].split(",") if x.strip()]
focal_scale = float(sys.argv[5])
pp_mode = sys.argv[6]

camid_to_name = {
    0: "camera_front_wide_120fov",
    1: "camera_cross_left_120fov",
    2: "camera_cross_right_120fov",
    3: "camera_rear_left_70fov",
    4: "camera_rear_right_70fov",
}
selected_names = {camid_to_name[c] for c in cam_ids if c in camid_to_name}

df = pd.read_parquet(calib_parquet)
if "clip_id" not in df.columns:
    raise RuntimeError(f"{calib_parquet} missing clip_id column")
if "camera_name" not in df.columns:
    raise RuntimeError(f"{calib_parquet} missing camera_name column")

sdf = df[df["clip_id"].astype(str) == scene_id].copy()
if sdf.empty:
    raise RuntimeError(f"scene_id={scene_id} not found in {calib_parquet}")

for i in range(5):
    bw_col = f"bw_poly_{i}"
    fw_col = f"fw_poly_{i}"
    if bw_col not in sdf.columns:
        sdf[bw_col] = 0.0
    if fw_col not in sdf.columns:
        sdf[fw_col] = 0.0

for idx, row in sdf.iterrows():
    cam_name = str(row["camera_name"])
    if cam_name not in selected_names:
        continue

    width = float(row["width"])
    height = float(row["height"])
    old_cx = float(row["cx"])
    old_cy = float(row["cy"])
    f_rect = float(row["fw_poly_1"]) * focal_scale
    if f_rect <= 0:
        raise RuntimeError(f"Invalid focal for {scene_id}/{cam_name}: {f_rect}")

    if pp_mode == "source":
        cx_rect, cy_rect = old_cx, old_cy
    elif pp_mode == "center":
        cx_rect, cy_rect = (width - 1.0) / 2.0, (height - 1.0) / 2.0
    else:
        raise RuntimeError(f"Unsupported principal point mode: {pp_mode}")

    sdf.at[idx, "cx"] = cx_rect
    sdf.at[idx, "cy"] = cy_rect

    # Rectified "no-distortion" coefficients in a simple linear kb form.
    # fw: rho ~= f * theta
    # bw: theta ~= (1/f) * rho
    sdf.at[idx, "fw_poly_0"] = 0.0
    sdf.at[idx, "fw_poly_1"] = f_rect
    sdf.at[idx, "fw_poly_2"] = 0.0
    sdf.at[idx, "fw_poly_3"] = 0.0
    sdf.at[idx, "fw_poly_4"] = 0.0

    sdf.at[idx, "bw_poly_0"] = 0.0
    sdf.at[idx, "bw_poly_1"] = 1.0 / f_rect
    sdf.at[idx, "bw_poly_2"] = 0.0
    sdf.at[idx, "bw_poly_3"] = 0.0
    sdf.at[idx, "bw_poly_4"] = 0.0

out_path.parent.mkdir(parents=True, exist_ok=True)
sdf.to_parquet(out_path, index=False)
print(f"[calib] wrote {out_path}")
PY
}

overwrite_scene() {
  local sid="$1"
  local dst_scene="$SRC_ROOT/$sid"
  local out_scene="$WORK_ROOT/$sid"

  if [[ ! -d "$dst_scene" ]]; then
    echo "[warn] skip overwrite, dst scene missing: $dst_scene" >&2
    return 0
  fi

  if [[ "$OVERWRITE_IMAGES" == "1" ]]; then
    local src_img="$out_scene/images_rectified"
    local dst_img="$dst_scene/images"
    if [[ -d "$src_img" ]]; then
      mkdir -p "$dst_img"
      shopt -s nullglob
      for f in "$src_img"/*; do
        cp -f "$f" "$dst_img/$(basename "$f")"
      done
      shopt -u nullglob
    fi
  fi

  if [[ "$OVERWRITE_SKY_MASKS" == "1" && "$COPY_SKY_MASKS" == "1" ]]; then
    local src_sky="$out_scene/sky_masks_rectified"
    if [[ -d "$src_sky" ]]; then
      local dst_sky=""
      if [[ -d "$dst_scene/sky_masks" ]]; then
        dst_sky="$dst_scene/sky_masks"
      elif [[ -d "$dst_scene/skymasks" ]]; then
        dst_sky="$dst_scene/skymasks"
      else
        dst_sky="$dst_scene/sky_masks"
      fi
      mkdir -p "$dst_sky"
      shopt -s nullglob
      for f in "$src_sky"/*; do
        cp -f "$f" "$dst_sky/$(basename "$f")"
      done
      shopt -u nullglob
    fi
  fi

  if [[ "$OVERWRITE_CALIB" == "1" ]]; then
    local src_calib="$out_scene/calibration/camera_intrinsics.parquet"
    local dst_calib="$dst_scene/calibration/camera_intrinsics.parquet"
    if [[ -f "$src_calib" ]]; then
      mkdir -p "$(dirname "$dst_calib")"
      if [[ "$src_calib" != "$dst_calib" ]]; then
        cp -f "$src_calib" "$dst_calib"
      fi
    fi
  fi
}

cleanup_inplace_scene_artifacts() {
  local sid="$1"
  if [[ "$USE_INTERMEDIATE_ROOT" != "0" ]]; then
    return 0
  fi
  if [[ "$KEEP_INPLACE_ARTIFACTS" == "1" ]]; then
    return 0
  fi
  local scene="$SRC_ROOT/$sid"
  rm -rf \
    "$scene/images_rectified" \
    "$scene/sky_masks_rectified" \
    "$scene/images_cropped" \
    "$scene/sky_masks_cropped" \
    "$scene/preview"
  rm -f "$scene/rectify_meta.json"
}

ok=0
fail=0
for sid in "${SCENES[@]}"; do
  echo "[scene] start $sid"
  if run_rectify_scene "$sid"; then
    if write_rectified_calib "$sid"; then
      if [[ "$OVERWRITE" == "1" ]]; then
        overwrite_scene "$sid"
        cleanup_inplace_scene_artifacts "$sid"
      fi
      ok=$((ok + 1))
      echo "[scene] done  $sid"
    else
      fail=$((fail + 1))
      echo "[scene] fail  $sid (calibration write failed)" >&2
    fi
  else
    fail=$((fail + 1))
    echo "[scene] fail  $sid (rectify failed)" >&2
  fi
done

echo "[summary] ok=$ok fail=$fail overwrite=$OVERWRITE work_root=$WORK_ROOT out_root=$OUT_ROOT"
if [[ "$OVERWRITE" == "0" ]]; then
  echo "[summary] source dataset unchanged (safe mode)."
fi
