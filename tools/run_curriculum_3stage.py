#!/usr/bin/env python3
"""
Three-stage curriculum SFT runner for crossview multi-image + depth (25% subset).

Stage 1: positives only (no negatives), 40 steps.
Stage 2: positives + tierA, 60 steps (resume from Stage 1).
Stage 3: all (pos + tierA + tierB + tierC), 100 steps (resume from Stage 2).

This script reuses the official `llava/train/train_mem.py` entry point and only changes:
- which registered dataset (`data_mixture`) is used per stage
- max_steps / logging_steps / output_dir

Model / dataset / collator / depth pipeline follow the official training code.
"""

import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any

import json


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/local_data/ky2738/envs/snpp2msg-rast/bin/python"
TRAIN_MEM = "llava/train/train_mem.py"  # Use relative path for torch.distributed.run

BASE_MODEL = ROOT / "runs" / "train" / "RoboRefer-2B-Depth-Align"

STAGES = [
    {
        "name": "stage1_pos_only",
        "data_mixture": "crossview_multimg_25pct_pos_only",
        "max_steps": 40,  # Cumulative: 0 -> 40
        "output_dir": ROOT / "runs" / "train" / "Curriculum-25pct-Stage1",
        "max_tiles": 6,  # Stage 1 can use more tiles (simpler data)
        "dataloader_workers": 4,  # Stage 1 can use more workers
    },
    {
        "name": "stage2_pos_tierA",
        "data_mixture": "crossview_multimg_25pct_pos_tierA",
        "max_steps": 100,  # Cumulative: 40 -> 100 (NOT 60! HF Trainer uses cumulative max_steps)
        "output_dir": ROOT / "runs" / "train" / "Curriculum-25pct-Stage2",
        "max_tiles": 4,  # Reduced for Stage 2 to prevent OOM
        "dataloader_workers": 2,  # Reduced for Stage 2 to save RAM
    },
    {
        "name": "stage3_all",
        "data_mixture": "crossview_multimg_25pct_all",
        "max_steps": 200,  # Cumulative: 100 -> 200 (NOT 100! HF Trainer uses cumulative max_steps)
        "output_dir": ROOT / "runs" / "train" / "Curriculum-25pct-Stage3",
        "max_tiles": 4,  # Reduced for Stage 3 to prevent OOM
        "dataloader_workers": 2,  # Reduced for Stage 3 to save RAM
    },
]


def _load_trainer_state(output_dir: Path) -> Dict[str, Any]:
    state_path = output_dir / "trainer_state.json"
    if not state_path.exists():
        print(f"⚠️  trainer_state.json not found in {output_dir}")
        return {}
    try:
        with state_path.open("r") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️  Failed to load trainer_state.json from {output_dir}: {e}")
        return {}


def _check_steps(name: str, expected_cumulative: int, output_dir: Path) -> None:
    state = _load_trainer_state(output_dir)
    global_step = state.get("global_step", None)
    if global_step is None:
        print(f"⚠️  {name}: global_step missing in trainer_state.json (expected {expected_cumulative})")
        return
    if global_step != expected_cumulative:
        raise RuntimeError(
            f"{name}: global_step={global_step} but expected cumulative {expected_cumulative}. "
            "Check max_steps / resume logic."
        )
    # Try to log LR at first & last step if present
    log_history = state.get("log_history", [])
    first_lr = None
    last_lr = None
    for log in log_history:
        if "learning_rate" in log:
            if first_lr is None:
                first_lr = log["learning_rate"]
            last_lr = log["learning_rate"]
    print(
        f"[{name}] global_step={global_step}, "
        f"lr_first={first_lr if first_lr is not None else 'n/a'}, "
        f"lr_last={last_lr if last_lr is not None else 'n/a'}"
    )


def _run_dev_eval(stage_output_dir: Path, sft_json: Path, dev_log_path: Path) -> None:
    """Run dev_format_eval.py on a stage checkpoint and append metrics to a JSONL file."""
    dev_log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        PYTHON,
        str(ROOT / "tools" / "dev_format_eval.py"),
        "--model",
        str(stage_output_dir),
        "--sft-json",
        str(sft_json),
        "--jsonl-out",
        str(dev_log_path),
    ]
    print(f"[Curriculum] Running dev_format_eval for {stage_output_dir}")
    env = os.environ.copy()
    # Ensure llava package is importable
    env["PYTHONPATH"] = str(ROOT) + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    subprocess.run(cmd, env=env)


def run_stage(stage_idx: int) -> None:
    stage = STAGES[stage_idx]
    name = stage["name"]
    data_mixture = stage["data_mixture"]
    max_steps = stage["max_steps"]
    output_dir: Path = stage["output_dir"]
    max_tiles = stage.get("max_tiles", 4)  # Default to 4 if not specified
    dataloader_workers = stage.get("dataloader_workers", 2)  # Default to 2 if not specified

    if stage_idx == 0:
        model_path = str(BASE_MODEL)
        resume_from_checkpoint = None  # Stage 1 starts from base model
        # For Stage 1, use the intended output_dir
        actual_output_dir = output_dir
        output_dir.mkdir(parents=True, exist_ok=True)
    else:
        # resume from previous stage checkpoint (HF format)
        prev_output = STAGES[stage_idx - 1]["output_dir"]
        # Find the latest checkpoint in previous stage
        import glob
        checkpoint_dirs = glob.glob(str(prev_output / "checkpoint-*"))
        if checkpoint_dirs:
            # Sort by step number and get the latest
            checkpoint_dirs.sort(key=lambda x: int(x.split("-")[-1]))
            checkpoint_path = checkpoint_dirs[-1]
            # For Stage 2/3: use previous stage output_dir as model_path (for loading weights)
            # AND explicitly pass resume_from_checkpoint to restore optimizer/scheduler/global_step
            model_path = str(prev_output)
            resume_from_checkpoint = checkpoint_path
            actual_output_dir = output_dir
            print(f"[Curriculum] Resuming from checkpoint: {resume_from_checkpoint}")
            print(f"[Curriculum] Model weights from: {model_path}")
            print(f"[Curriculum] Training in output_dir: {actual_output_dir}")
        else:
            # Fallback: use output_dir as both model_path and resume point
            model_path = str(prev_output)
            resume_from_checkpoint = str(prev_output)
            actual_output_dir = output_dir
            print(f"[Curriculum] Resuming from output_dir: {model_path}")
        
        # Create the final output directory for later use
        output_dir.mkdir(parents=True, exist_ok=True)

    # Build command list
    cmd = [
        PYTHON,
        "-m",
        "torch.distributed.run",
        "--nnodes=1",
        "--nproc_per_node=1",
        "--master_port",
        "29521",
        str(TRAIN_MEM),
        "--deepspeed",
        str(ROOT / "scripts" / "zero3.json"),
        "--model_name_or_path",
        model_path,
    ]
    
    # Add resume_from_checkpoint for Stage 2/3
    if resume_from_checkpoint is not None:
        cmd.append("--resume_from_checkpoint")
        cmd.append(resume_from_checkpoint)
    
    # Add remaining arguments
    cmd.extend([
        "--chat_template",
        "qwen2",
        "--data_mixture",
        data_mixture,
        "--vision_tower",
        "Efficient-Large-Model/paligemma-siglip-so400m-patch14-448",
        "--depth_tower",
        "Efficient-Large-Model/paligemma-siglip-so400m-patch14-448",
        "--mm_vision_select_feature",
        "cls_patch",
        "--mm_projector",
        "mlp_downsample_3x3_fix",
        "--depth_projector",
        "mlp_downsample_3x3_fix",
        "--enable_depth",
        "True",
        "--use_depth_tower",
        "True",
        "--tune_vision_tower",
        "True",
        "--tune_mm_projector",
        "True",
        "--tune_language_model",
        "True",
        "--tune_depth_tower",
        "False",
        "--tune_depth_projector",
        "True",
        "--mm_vision_select_layer",
        "-2",
        "--mm_use_im_start_end",
        "False",
        "--mm_use_im_patch_token",
        "False",
        "--image_aspect_ratio",
        "dynamic",
        "--bf16",
        "True",
        "--output_dir",
        str(actual_output_dir),
        "--num_train_epochs",
        "1",
        "--max_steps",
        str(max_steps),
        "--per_device_train_batch_size",
        "1",
        "--gradient_accumulation_steps",
        "4",
        "--evaluation_strategy",
        "no",
        "--save_strategy",
        "steps",
        "--save_steps",
        "500",
        "--save_total_limit",
        "2",
        "--learning_rate",
        "1e-5",
        "--weight_decay",
        "0.",
        "--warmup_ratio",
        "0.03",
        "--lr_scheduler_type",
        "cosine",
        "--logging_steps",
        "10",
        "--model_max_length",
        "4096",  # Keep at 4096 for Stage 1, but could reduce to 2048 for Stage 2/3 if needed
        "--gradient_checkpointing",
        "True",
        "--max_tiles",
        str(max_tiles),  # Stage-specific: 6 for Stage 1, 4 for Stage 2/3
        "--min_tiles",
        "1",
        "--torch_empty_cache_steps",
        "1",  # More frequent cache clearing to prevent memory accumulation
        "--dataloader_num_workers",
        str(dataloader_workers),  # Stage-specific: 4 for Stage 1, 2 for Stage 2/3
        "--report_to",
        "none",
    ])

    print("=" * 70)
    print(f"[Curriculum] Running {name} | data_mixture={data_mixture} | max_steps={max_steps}")
    print("Command:")
    print(" ".join(cmd))
    print("=" * 70)

    env = os.environ.copy()
    # Ensure llava package is importable
    env["PYTHONPATH"] = str(ROOT) + (":" + env["PYTHONPATH"] if "PYTHONPATH" in env else "")
    # Avoid requiring mpi4py for DeepSpeed discovery
    env["DEEPSPEED_DISABLE_MPI"] = "1"
    # Enable better error reporting
    env["PYTHONFAULTHANDLER"] = "1"
    env["TORCH_SHOW_CPP_STACKTRACES"] = "1"
    
    os.chdir(ROOT)
    # Capture both stdout and stderr to see real errors
    # Use subprocess.STDOUT to merge stderr into stdout
    result = subprocess.run(
        cmd, 
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # Merge stderr into stdout
        text=True,
        bufsize=1,  # Line buffered
    )
    
    # Print the output (which now includes stderr) for debugging
    if result.stdout:
        print(result.stdout)
    
    # Also save to log file for later inspection
    log_file = output_dir / f"stage_{name}_full.log"
    with open(log_file, "w") as f:
        f.write(result.stdout)
    print(f"[Curriculum] Full log saved to: {log_file}")
    if result.returncode != 0:
        print(f"❌ Stage {name} failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    # Safeguard: check cumulative steps
    # Always check the actual_output_dir (which is output_dir for all stages now)
    check_dir = actual_output_dir
    expected_cumulative = max_steps  # Use the stage's max_steps (which is now cumulative)
    
    # Read trainer_state.json to verify completion
    state = _load_trainer_state(check_dir)
    actual_step = state.get("global_step", None)
    
    if actual_step is None:
        raise RuntimeError(
            f"{name}: trainer_state.json missing or invalid in {check_dir}. "
            "Training may have failed before saving state."
        )
    
    if actual_step < expected_cumulative:
        raise RuntimeError(
            f"{name}: Training incomplete. global_step={actual_step} < expected {expected_cumulative}. "
            f"Check log file: {check_dir / f'stage_{name}_full.log'}"
        )
    
    if actual_step > expected_cumulative:
        print(f"⚠️  {name}: global_step={actual_step} > expected {expected_cumulative}. This is unusual but not fatal.")
    
    _check_steps(name, expected_cumulative, check_dir)
    
    # For Stage 2/3: Copy/move the final checkpoint to the intended output_dir
    # Note: Now actual_output_dir == output_dir for all stages, so this section is not needed
    # But keeping it for backward compatibility
    if stage_idx > 0 and actual_output_dir != output_dir:
        import shutil
        # Find the latest checkpoint in actual_output_dir
        checkpoint_dirs = list(actual_output_dir.glob("checkpoint-*"))
        if checkpoint_dirs:
            checkpoint_dirs.sort(key=lambda x: int(x.name.split("-")[-1]))
            latest_checkpoint = checkpoint_dirs[-1]
            # Copy the latest checkpoint to the final output_dir
            dest_checkpoint = output_dir / latest_checkpoint.name
            print(f"[Curriculum] Copying final checkpoint from {latest_checkpoint} to {dest_checkpoint}")
            if dest_checkpoint.exists():
                shutil.rmtree(dest_checkpoint)
            shutil.copytree(latest_checkpoint, dest_checkpoint)
            # Also copy trainer_state.json and config.json if they exist
            for file in ["trainer_state.json", "config.json"]:
                src_file = actual_output_dir / file
                if src_file.exists():
                    shutil.copy2(src_file, output_dir / file)
            print(f"[Curriculum] Checkpoint copied to final output_dir: {output_dir}")

    # Run dev-format eval and save JSONL under each stage dir
    dev_jsonl = output_dir / "dev_eval.jsonl"
    _run_dev_eval(
        stage_output_dir=output_dir,
        sft_json=ROOT / "tmp" / "crossview_multimg_sft_25pct_with_depth.json",
        dev_log_path=dev_jsonl,
    )


def main() -> None:
    for i in range(len(STAGES)):
        run_stage(i)


if __name__ == "__main__":
    main()

