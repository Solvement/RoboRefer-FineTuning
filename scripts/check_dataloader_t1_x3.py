#!/usr/bin/env python3
"""
One-batch dataloader sanity check for Task1 x3 (multi-image).
No model load; only dataset + dataloader + one batch.
Verifies: no assertion, batch image tensor shape reasonable.
"""
import sys
from pathlib import Path

# Repo root
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

def main():
    from transformers import AutoTokenizer
    from llava.constants import MEDIA_TOKENS
    from llava.data.dataset import LazySupervisedSpatialDataset
    from llava.data.collate import DataCollator
    from llava.train.args import DataArguments, TrainingArguments
    from torch.utils.data import DataLoader

    # Use base Qwen2 tokenizer (same as LLaVA LLM); LLaVA checkpoint may not expose tokenizer directly
    tokenizer_path = "Qwen/Qwen2-1.5B-Instruct"
    data_path = REPO_ROOT / "tmp" / "spatial_tasks_x3" / "t1_train_multi.json"
    vision_tower = "Efficient-Large-Model/paligemma-siglip-so400m-patch14-448"

    if not data_path.exists():
        print(f"Missing {data_path}; run build_spatial_tasks_x3 first.")
        sys.exit(1)

    print("Loading tokenizer from", tokenizer_path)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    # Attach media tokens like builder does
    tokenizer.media_tokens = MEDIA_TOKENS
    tokenizer.media_token_ids = {}
    for name, token in MEDIA_TOKENS.items():
        tokenizer.add_tokens([token], special_tokens=True)
        tokenizer.media_token_ids[name] = tokenizer.convert_tokens_to_ids(token)

    print("Loading image processor from", vision_tower)
    from transformers import AutoImageProcessor
    image_processor = AutoImageProcessor.from_pretrained(vision_tower, trust_remote_code=True)

    data_args = DataArguments(
        data_path=str(data_path),
        image_folder="/",
        image_aspect_ratio="resize",
        is_multimodal=True,
    )
    data_args.image_processor = image_processor

    training_args = TrainingArguments(output_dir=str(REPO_ROOT / "runs" / "train" / "mv_consistency_sft_x3"))

    print("Building LazySupervisedSpatialDataset...")
    dataset = LazySupervisedSpatialDataset(
        data_path=str(data_path),
        image_folder="/",
        tokenizer=tokenizer,
        data_args=data_args,
        training_args=training_args,
    )
    collator = DataCollator(tokenizer=tokenizer)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collator, num_workers=0)

    print("Fetching one batch...")
    batch = next(iter(dataloader))
    print("Batch keys:", list(batch.keys()))
    if "media" in batch and "image" in batch["media"]:
        imgs = batch["media"]["image"]
        print("media['image']: len =", len(imgs))
        if imgs:
            t = imgs[0] if isinstance(imgs[0], __import__("torch").Tensor) else imgs
            if hasattr(t, "shape"):
                print("  first image shape:", t.shape)
            else:
                print("  first image: list of", len(imgs[0]) if isinstance(imgs[0], list) else "?", "tensors")
    print("input_ids shape:", batch["input_ids"].shape)
    print("labels shape:", batch["labels"].shape)
    print("OK: one-batch dataloader check passed (no assertion, shapes present).")
    return 0

if __name__ == "__main__":
    sys.exit(main())
