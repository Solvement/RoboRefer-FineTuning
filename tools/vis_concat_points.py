import argparse, json, os, random, re
from PIL import Image, ImageDraw

def parse_first_point(text: str):
    """
    Parse first (x,y) from strings like:
      "[(0.385, 0.480)]"
      "(0.385, 0.480)"
      "x=0.385 y=0.480"
    Returns (x,y) floats or None.
    """
    if text is None:
        return None
    m = re.search(r'\(\s*([0-9]*\.?[0-9]+)\s*,\s*([0-9]*\.?[0-9]+)\s*\)', text)
    if not m:
        return None
    x = float(m.group(1)); y = float(m.group(2))
    return x, y

def parse_a_point_from_prompt(prompt: str):
    # take the first "(x, y)" in prompt as A-coord (for debugging)
    return parse_first_point(prompt or "")

def to_full_xy(xy, meta, coord_frame: str):
    """
    Convert predicted coords to FULL concat-image frame.
    coord_frame:
      - "full": already full
      - "right": coords are relative to RIGHT half (View B)
    meta provides img_a_width_norm/img_b_width_norm if available, else assume 0.5/0.5.
    """
    if xy is None:
        return None
    x, y = xy
    if coord_frame == "full":
        return x, y
    if coord_frame == "right":
        wa = float(meta.get("img_a_width_norm", 0.5))
        wb = float(meta.get("img_b_width_norm", 0.5))
        return wa + wb * x, y
    raise ValueError(f"Unknown coord_frame: {coord_frame}")

def draw_cross(draw, cx, cy, r=6, width=2):
    draw.line((cx-r, cy, cx+r, cy), width=width)
    draw.line((cx, cy-r, cx, cy+r), width=width)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="e.g. Evaluation/RefSpatial-Bench/CrossView_train")
    ap.add_argument("--question_json", default=None, help="default: <data_root>/question.json")
    ap.add_argument("--result_jsonl", default=None, help="optional: Evaluation/outputs/.../CrossView_train.jsonl")
    ap.add_argument("--coord_frame", default="full", choices=["full","right"], help="how to interpret model outputs")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    qpath = args.question_json or os.path.join(args.data_root, "question.json")
    with open(qpath, "r") as f:
        questions = json.load(f)

    pred_map = {}
    if args.result_jsonl and os.path.exists(args.result_jsonl):
        with open(args.result_jsonl, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                qid = obj.get("question_id") or obj.get("id")
                pred = parse_first_point(obj.get("text",""))
                if qid and pred:
                    pred_map[qid] = pred

    random.seed(args.seed)
    os.makedirs(args.outdir, exist_ok=True)
    picks = random.sample(questions, k=min(args.n, len(questions)))

    for q in picks:
        qid = q.get("id")
        rgb_rel = q.get("rgb_path")
        mask_rel = q.get("mask_path")
        rgb_path = os.path.join(args.data_root, rgb_rel) if rgb_rel else None
        mask_path = os.path.join(args.data_root, mask_rel) if mask_rel else None

        if not (rgb_path and os.path.exists(rgb_path) and mask_path and os.path.exists(mask_path)):
            continue

        img = Image.open(rgb_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        w, h = img.size

        # overlay mask (red)
        mask_rgba = Image.new("RGBA", img.size, (255,0,0,0))
        mp = mask.point(lambda p: 120 if p > 0 else 0)  # alpha
        mask_rgba.putalpha(mp)
        vis = img.convert("RGBA")
        vis = Image.alpha_composite(vis, mask_rgba)

        draw = ImageDraw.Draw(vis)

        # draw split line if concat
        if q.get("concat_mode", False):
            wa = float(q.get("img_a_width_norm", 0.5))
            x_split = int(round(wa * w))
            draw.line((x_split, 0, x_split, h), width=3)

        # A point from prompt (blue)
        a_xy = parse_a_point_from_prompt(q.get("prompt",""))
        if a_xy:
            ax, ay = a_xy
            cx = int(round(ax * w)); cy = int(round(ay * h))
            draw_cross(draw, cx, cy, r=10, width=3)

        # GT points (green): prefer gt_point_concat if exists else gt_point_norm
        gt = q.get("gt_point_concat") or q.get("gt_point_norm")
        if isinstance(gt, (list, tuple)) and len(gt) == 2:
            gx, gy = float(gt[0]), float(gt[1])
            cx = int(round(gx * w)); cy = int(round(gy * h))
            draw.ellipse((cx-8, cy-8, cx+8, cy+8), outline="lime", width=4)

        # Pred point (yellow)
        pred = pred_map.get(qid)
        if pred:
            px_full, py_full = to_full_xy(pred, q, args.coord_frame)
            cx = int(round(px_full * w)); cy = int(round(py_full * h))
            draw.rectangle((cx-8, cy-8, cx+8, cy+8), outline="yellow", width=4)

        out_path = os.path.join(args.outdir, f"{qid}.png")
        vis.convert("RGB").save(out_path)

    print(f"[done] wrote overlays to: {args.outdir}")

if __name__ == "__main__":
    main()
