#!/usr/bin/env python3
"""
Build evaluation-first spatial tasks (x3 downsampled) JSONs.

Tasks (all using ABSOLUTE image paths, no <image> tokens in prompts):

- Task1 (cross-view correspondence):
  * Input (multi): [A_marked, B_original]
  * Input (concat): concat(A_original, B_original) as a single wide image
  * Output: [(x,y)] normalized wrt B (or full concat) or NOT_VISIBLE

- Task2A (missing across view):
  * Re-use Task1 pairs where query mask is empty => GT NOT_VISIBLE.

- Task2B (difference grounding):
  * Input: [original, inpainted] (same view)
  * Output: [(x,y)] normalized wrt inpainted view.

GT point rule (for visible cases):
  - Compute bbox of mask>0. Take bbox center (cx, cy).
  - If (cx, cy) lies inside mask: use it.
  - Else: sample ONE point uniformly from mask pixels (deterministic RNG).
  - Normalize by image width/height to [0,1].
"""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image


DATA_ROOT_X3_DEFAULT = "/local_data/jz4725/scannet_inpainted_dilate002_15obj_5frames_corrected_x3"


def load_mask(uid_dir: Path, k: int, use_dilated: bool) -> Tuple[Optional[Path], Optional[np.ndarray]]:
    """Load mask for view k (01..05). Returns (mask_path, mask_array or None)."""
    k_str = f"{k:02d}"
    base = uid_dir
    if use_dilated:
        pattern = f"{k_str}_*_mask_dialated.png"
    else:
        pattern = f"{k_str}_*_mask.png"
    paths = list(base.glob(pattern))
    if not paths:
        # fallback: try the other variant
        alt_pattern = f"{k_str}_*_mask.png" if use_dilated else f"{k_str}_*_mask_dialated.png"
        paths = list(base.glob(alt_pattern))
        if not paths:
            return None, None
    mask_path = paths[0]
    arr = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if arr is None:
        return mask_path, None
    return mask_path, arr


def load_original_and_inpainted(uid_dir: Path, k: int) -> Tuple[Optional[Path], Optional[Path]]:
    """Return (original_path, inpainted_path) for view k (if exist)."""
    k_str = f"{k:02d}"
    origs = list(uid_dir.glob(f"{k_str}_*_original.png"))
    inps = list(uid_dir.glob(f"{k_str}_*_inpainted.png"))
    orig = origs[0] if origs else None
    inp = inps[0] if inps else None
    return orig, inp


def compute_gt_point_from_mask(
    mask: np.ndarray,
    rng: random.Random,
) -> Optional[Tuple[float, float]]:
    """Compute GT point from **mask bbox** using bbox-center-else-random rule.

    Here "bbox" is explicitly the tight bounding box of `mask>0` (no external
    detector). If the bbox center lies inside the mask, we use that. Otherwise we
    sample a single pixel uniformly from `mask>0`.

    Returns (x_norm, y_norm) in [0,1] or None if mask is empty.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None

    h, w = mask.shape
    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()
    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)

    # If bbox center lies within mask, use it; else sample random point on mask.
    cx_i = int(round(cx))
    cy_i = int(round(cy))
    if 0 <= cx_i < w and 0 <= cy_i < h and mask[cy_i, cx_i] > 0:
        x, y = cx, cy
    else:
        idx = rng.randrange(len(xs))
        x, y = float(xs[idx]), float(ys[idx])

    # Normalize to [0,1] (use (coord + 0.5) / size to be pixel-center aware).
    x_norm = (x + 0.5) / float(w)
    y_norm = (y + 0.5) / float(h)
    x_norm = max(0.0, min(1.0, x_norm))
    y_norm = max(0.0, min(1.0, y_norm))
    return x_norm, y_norm


def create_marked_image(original_path: Path, mask: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Create marked image by overlaying red highlight on mask>0."""
    img = Image.open(original_path).convert("RGB")
    img_arr = np.array(img)
    overlay = img_arr.copy()
    overlay[mask > 0] = [255, 0, 0]
    mask_3d = (mask > 0)[:, :, None].astype(float)
    marked = (img_arr * (1 - alpha * mask_3d) + overlay * (alpha * mask_3d)).astype(np.uint8)
    return Image.fromarray(marked)


def concat_side_by_side(left_path: Path, right_path: Path, out_path: Path) -> None:
    """Create a horizontal concatenation of two RGB images and save as JPG/PNG."""
    img_l = Image.open(left_path).convert("RGB")
    img_r = Image.open(right_path).convert("RGB")
    h = max(img_l.height, img_r.height)
    w = img_l.width + img_r.width
    canvas = Image.new("RGB", (w, h), (0, 0, 0))
    canvas.paste(img_l, (0, 0))
    canvas.paste(img_r, (img_l.width, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def build_t1_prompt_multi() -> str:
    """Prompt for Task1, multi-image format [marked A, original B]."""
    return (
        "You are given TWO separate images:\n"
        "- Image A (REFERENCE): the target object is highlighted (marked) in the image.\n"
        "- Image B (QUERY): you need to find the SAME object as in Image A.\n\n"
        "TASK:\n"
        "1. Look at Image A and understand which object is marked.\n"
        "2. Look at Image B and determine whether the SAME object is visible.\n"
        "3. If the object is visible in Image B, output ONE point coordinate on that object.\n"
        "4. If the object is NOT visible in Image B, answer NOT_VISIBLE.\n\n"
        "OUTPUT FORMAT:\n"
        "- If visible: answer with one coordinate in normalized [0,1] range relative to Image B only, in the form: [(x, y)]\n"
        "- If NOT visible: answer exactly: NOT_VISIBLE"
    )


def build_t1_prompt_concat(x_ref_full: float, y_ref_full: float) -> str:
    """Prompt for Task1 concat format (single wide image)."""
    return (
        "You are given ONE concatenated image consisting of TWO halves:\n"
        "- LEFT HALF: reference image where the target object is indicated by a point.\n"
        "- RIGHT HALF: query image where you need to find the SAME object as in the left half.\n\n"
        f"The reference point (on the left half) is located at normalized coordinate [(x, y)] "
        f"= [({x_ref_full:.3f}, {y_ref_full:.3f})] with respect to the FULL concatenated image "
        "(width and height both normalized to [0,1]).\n\n"
        "TASK:\n"
        "1. Look at the left half and understand which object the reference point indicates.\n"
        "2. Look at the right half and determine whether the SAME object is visible.\n"
        "3. If the object is visible in the RIGHT half, output ONE point coordinate on that object "
        "in normalized [0,1] range with respect to the FULL concatenated image.\n"
        "4. If the object is NOT visible in the right half, answer NOT_VISIBLE.\n\n"
        "OUTPUT FORMAT:\n"
        "- If visible: answer with one coordinate in the form: [(x, y)] (normalized over the FULL concatenated image).\n"
        "- If NOT visible: answer exactly: NOT_VISIBLE"
    )


def build_t2b_prompt() -> str:
    """Prompt for Task2B (difference grounding, [original, inpainted])."""
    return (
        "You are given TWO images of the SAME view:\n"
        "- Image A: the original image.\n"
        "- Image B: the inpainted image where some region has been changed or removed.\n\n"
        "The target region corresponds to the area that was changed/removed in Image B.\n\n"
        "TASK:\n"
        "1. Compare Image A and Image B.\n"
        "2. In Image B, locate ONE point inside the changed/removed region.\n\n"
        "OUTPUT FORMAT:\n"
        "- Always output ONE coordinate in normalized [0,1] range relative to Image B only, "
        "in the form: [(x, y)]."
    )


@dataclass
class ViewInfo:
    k: int
    frame_id: str
    original: Path
    inpainted: Optional[Path]
    mask: Optional[np.ndarray]
    gt_point_norm: Optional[Tuple[float, float]]  # wrt this view (for query/inpainted)


def extract_frame_id(name: str) -> str:
    """Extract frame id from filename like 01_004130_original.png -> 004130."""
    parts = name.split("_")
    if len(parts) >= 3:
        return parts[1]
    return "unknown"


def collect_views_for_uid(
    uid_dir: Path,
    use_dilated_mask: bool,
    rng: random.Random,
) -> List[ViewInfo]:
    """Collect per-view info for a uid.

    Important: we **require** a mask file to be present for this view. If the
    mask file is missing or fails to load, we treat it as bad data and **skip**
    the view entirely (rather than silently assuming NOT_VISIBLE).

    This way, Task2A 的“empty mask”只指 mask 存在但全 0；mask 文件缺失属于数据问题。
    """
    views: List[ViewInfo] = []
    for k in range(1, 6):
        orig, inpainted = load_original_and_inpainted(uid_dir, k)
        if orig is None:
            continue
        mask_path, mask_arr = load_mask(uid_dir, k, use_dilated_mask)
        if mask_path is None or mask_arr is None:
            # Mask file missing or unreadable: skip this view completely.
            continue
        gt_point = compute_gt_point_from_mask(mask_arr, rng)
        frame_id = extract_frame_id(orig.name)
        views.append(
            ViewInfo(
                k=k,
                frame_id=frame_id,
                original=orig.resolve(),
                inpainted=inpainted.resolve() if inpainted is not None else None,
                mask=mask_arr,
                gt_point_norm=gt_point,
            )
        )
    return views


def build_task_samples_for_uid(
    split: str,
    scene_id: str,
    uid: str,
    uid_dir: Path,
    marked_root: Path,
    concat_root: Path,
    mode: str,
    anchor_k: int,
    alpha: float,
    use_dilated_mask: bool,
    emit_concat: bool,
    rng: random.Random,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Build samples for one (scene_id, uid).

    Returns:
        t1_multi, t1_concat, t2a_multi, t2b_multi (lists of JSON dicts).
    """
    views = collect_views_for_uid(uid_dir, use_dilated_mask, rng)
    if len(views) < 1:
        return [], [], [], []

    # Precompute marked images for each view (if it has a mask).
    marked_paths: Dict[int, Optional[Path]] = {}
    for v in views:
        if v.mask is None:
            marked_paths[v.k] = None
            continue
        marked_rel = Path(split) / scene_id / f"uid_{uid}" / f"{v.k:02d}_{v.frame_id}_marked.png"
        marked_full = marked_root / marked_rel
        marked_full.parent.mkdir(parents=True, exist_ok=True)
        if not marked_full.exists():
            img_marked = create_marked_image(v.original, v.mask, alpha)
            img_marked.save(marked_full)
        marked_paths[v.k] = marked_full.resolve()

    t1_multi: List[Dict[str, Any]] = []
    t1_concat: List[Dict[str, Any]] = []
    t2a_multi: List[Dict[str, Any]] = []
    t2b_multi: List[Dict[str, Any]] = []

    # --- Task1 (multi + concat) and Task2A (subset of Task1 where query mask empty) ---
    def iter_pairs():
        if mode == "anchor":
            # A is anchor_k, B iterates others
            ref = next((v for v in views if v.k == anchor_k), None)
            if ref is None:
                return
            for q in views:
                if q.k == ref.k:
                    continue
                yield ref, q
        else:  # allpairs
            for ref in views:
                for q in views:
                    if ref.k == q.k:
                        continue
                    yield ref, q

    for ref_view, q_view in iter_pairs():
        # Multi-image Task1
        marked_a = marked_paths.get(ref_view.k)
        if marked_a is None:
            # cannot form a proper marked reference; skip this pair
            continue
        sample_id_base = f"{scene_id}_uid{uid}_A{ref_view.k:02d}{ref_view.frame_id}_B{q_view.k:02d}{q_view.frame_id}"
        if q_view.gt_point_norm is None:
            gt_text = "NOT_VISIBLE"
        else:
            x, y = q_view.gt_point_norm
            gt_text = f"[({x:.3f}, {y:.3f})]"

        s_multi = {
            "id": f"T1_MULTI_{sample_id_base}",
            "task": "t1_multi",
            "image": [str(marked_a), str(q_view.original)],
            "query_original": str(q_view.original),
            "conversations": [
                {"from": "human", "value": build_t1_prompt_multi()},
                {"from": "gpt", "value": gt_text},
            ],
        }
        t1_multi.append(s_multi)

        # Task2A: missing across view (query mask empty)
        if q_view.gt_point_norm is None:
            s_t2a = {
                "id": f"T2A_MULTI_{sample_id_base}",
                "task": "t2a_multi",
                "image": [str(marked_a), str(q_view.original)],
                "query_original": str(q_view.original),
                "conversations": [
                    {"from": "human", "value": build_t1_prompt_multi()},
                    {"from": "gpt", "value": "NOT_VISIBLE"},
                ],
            }
            t2a_multi.append(s_t2a)

        # Concat variant for Task1 (if requested)
        if emit_concat:
            # We still use original (unmarked) left as visual; the reference point is given in text.
            if ref_view.mask is None or ref_view.gt_point_norm is None:
                # Without a visible mask/point on ref, concat reference is not well-defined; skip.
                continue
            # Convert ref_view.gt_point_norm (xL,yL wrt left image) to full concat coordinates:
            xL, yL = ref_view.gt_point_norm
            x_full = xL * 0.5
            y_full = yL
            concat_rel = Path(split) / scene_id / f"uid_{uid}" / f"{ref_view.k:02d}{ref_view.frame_id}_to_{q_view.k:02d}{q_view.frame_id}_concat.jpg"
            concat_full = concat_root / concat_rel
            concat_side_by_side(ref_view.original, q_view.original, concat_full)
            s_concat = {
                "id": f"T1_CONCAT_{sample_id_base}",
                "task": "t1_concat",
                "image": str(concat_full.resolve()),
                "query_original": str(q_view.original),
                "conversations": [
                    {
                        "from": "human",
                        "value": build_t1_prompt_concat(x_ref_full=x_full, y_ref_full=y_full),
                    },
                    {"from": "gpt", "value": gt_text},
                ],
            }
            t1_concat.append(s_concat)

    # --- Task2B (difference grounding, per view) ---
    for v in views:
        if v.inpainted is None or v.mask is None:
            continue
        if v.gt_point_norm is None:
            # mask empty after preprocessing; skip
            continue
        x, y = v.gt_point_norm
        gt_text = f"[({x:.3f}, {y:.3f})]"
        sample_id = f"{scene_id}_uid{uid}_K{v.k:02d}{v.frame_id}"
        s_t2b = {
            "id": f"T2B_MULTI_{sample_id}",
            "task": "t2b_multi",
            "image": [str(v.original), str(v.inpainted)],
            "query_original": str(v.inpainted),
            "conversations": [
                {"from": "human", "value": build_t2b_prompt()},
                {"from": "gpt", "value": gt_text},
            ],
        }
        t2b_multi.append(s_t2b)

    return t1_multi, t1_concat, t2a_multi, t2b_multi


def main() -> None:
    parser = argparse.ArgumentParser(description="Build x3 spatial tasks JSONs (Task1/Task2A/Task2B).")
    parser.add_argument(
        "--data_root",
        type=str,
        default=DATA_ROOT_X3_DEFAULT,
        help="Root of x3 dataset.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory for JSONs and generated images.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for GT sampling.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="anchor",
        choices=["anchor", "allpairs"],
        help="Pairing mode for Task1.",
    )
    parser.add_argument(
        "--anchor_k",
        type=int,
        default=1,
        help="Anchor view index (1-5) when mode=anchor.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.45,
        help="Overlay opacity for marked reference images.",
    )
    parser.add_argument(
        "--use_dilated_mask",
        action="store_true",
        default=True,
        help="Prefer *_mask_dialated.png when available.",
    )
    parser.add_argument(
        "--emit_concat",
        type=int,
        default=1,
        choices=[0, 1],
        help="Whether to emit Task1 concat JSONs/images.",
    )

    args = parser.parse_args()

    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    marked_root = out_dir / "marked_abs"
    concat_root = out_dir / "concat_abs"
    marked_root.mkdir(parents=True, exist_ok=True)
    if args.emit_concat:
        concat_root.mkdir(parents=True, exist_ok=True)

    print("==============================================")
    print("Building spatial tasks (x3)")
    print("==============================================")
    print(f" data_root   : {data_root}")
    print(f" out_dir     : {out_dir}")
    print(f" mode        : {args.mode}")
    print(f" anchor_k    : {args.anchor_k}")
    print(f" alpha       : {args.alpha}")
    print(f" use_dilated : {args.use_dilated_mask}")
    print(f" emit_concat : {args.emit_concat}")
    print()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    t1_train_multi: List[Dict[str, Any]] = []
    t1_val_multi: List[Dict[str, Any]] = []
    t1_train_concat: List[Dict[str, Any]] = []
    t1_val_concat: List[Dict[str, Any]] = []
    t2a_val_multi: List[Dict[str, Any]] = []
    t2b_val_multi: List[Dict[str, Any]] = []
    t2a_train_multi: List[Dict[str, Any]] = []
    t2b_train_multi: List[Dict[str, Any]] = []

    for split in ["train", "validation"]:
        split_dir = data_root / split
        if not split_dir.exists():
            print(f"⚠️  Split dir not found: {split_dir}")
            continue

        print(f"Processing split = {split} ...")
        scene_dirs = [d for d in sorted(split_dir.iterdir()) if d.is_dir()]
        for si, scene_dir in enumerate(scene_dirs):
            scene_id = scene_dir.name
            uid_dirs = [d for d in sorted(scene_dir.iterdir()) if d.is_dir() and d.name.startswith("uid_")]
            for uid_dir in uid_dirs:
                uid = uid_dir.name.replace("uid_", "")
                (
                    t1_m,
                    t1_c,
                    t2a_m,
                    t2b_m,
                ) = build_task_samples_for_uid(
                    split=split,
                    scene_id=scene_id,
                    uid=uid,
                    uid_dir=uid_dir,
                    marked_root=marked_root,
                    concat_root=concat_root,
                    mode=args.mode,
                    anchor_k=args.anchor_k,
                    alpha=args.alpha,
                    use_dilated_mask=args.use_dilated_mask,
                    emit_concat=bool(args.emit_concat),
                    rng=rng,
                )
                if split == "train":
                    t1_train_multi.extend(t1_m)
                    t1_train_concat.extend(t1_c)
                    t2a_train_multi.extend(t2a_m)
                    t2b_train_multi.extend(t2b_m)
                else:
                    t1_val_multi.extend(t1_m)
                    t1_val_concat.extend(t1_c)
                    t2a_val_multi.extend(t2a_m)
                    t2b_val_multi.extend(t2b_m)

            if (si + 1) % 10 == 0:
                print(f"  processed {si+1}/{len(scene_dirs)} scenes in split {split}")

    def save_json(path: Path, data: List[Dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"  -> {path} ({len(data)} samples)")

    print()
    print("Saving JSONs ...")
    save_json(out_dir / "t1_train_multi.json", t1_train_multi)
    save_json(out_dir / "t1_val_multi.json", t1_val_multi)
    if args.emit_concat:
        save_json(out_dir / "t1_train_concat.json", t1_train_concat)
        save_json(out_dir / "t1_val_concat.json", t1_val_concat)

    # Task2A / Task2B: primarily eval; still save both splits if non-empty.
    if t2a_train_multi:
        save_json(out_dir / "t2a_train_multi.json", t2a_train_multi)
    save_json(out_dir / "t2a_val_multi.json", t2a_val_multi)

    if t2b_train_multi:
        save_json(out_dir / "t2b_train_multi.json", t2b_train_multi)
    save_json(out_dir / "t2b_val_multi.json", t2b_val_multi)

    print()
    print("==============================================")
    print("✅ Spatial tasks (x3) data generation done.")
    print("==============================================")
    print(f"Marked images root : {marked_root}")
    if args.emit_concat:
        print(f"Concat images root : {concat_root}")


if __name__ == "__main__":
    main()

