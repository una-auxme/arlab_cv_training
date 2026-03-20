"""
Train and evaluate YOLO segmentation models for augmentation experiments.

This script runs multiple experiments (baseline / moderate_geom / strong_geom and fruit variant),
performs a validation step with fixed conf/iou, exports `best.pt` into `yolo_weights/`,
and writes a summary CSV (one row per experiment) for easy comparison.
"""

import shutil
import time
import csv
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Any, List

from ultralytics import YOLO

# yolo26n.pt is needed for AMP checks and can be copied into the Ultralytics weights cache.


# ============================================================
# 1) PATHS / SETTINGS
# ============================================================

# Project root: repository root (this file lives in scripts/local/)
project_root = Path(__file__).resolve().parents[2]

# DATA_YAML is set dynamically based on the --dataset argument

# Training settings
EPOCHS = 5000
IMGSZ = 640
BATCH = 16          # RTX 3080: safe. If OOM: use 4
DEVICE = 0          # GPU (CPU would be -1)
PATIENCE = 100

# ============================================================
# 2) VAL SETTINGS (fixed for all runs)
# ============================================================
# For a fair Precision/Recall/F1 comparison, conf/iou must be fixed.
# data is set dynamically.
def get_val_args(data_yaml: Path) -> dict:
    return dict(
        data=str(data_yaml),
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        conf=0.25,      # fixed confidence threshold (for P/R/F1 comparison)
        iou=0.50,       # fixed IoU threshold
        split="val",
    )

# ============================================================
# 3) 3 EXPERIMENTS (mosaic disabled)
# ============================================================
# Idea: vary geometry only; keep realistic color/lighting augmentation.
EXPERIMENTS = [
    {
        "name": "no_augmentation",
        "params": dict(),
    },
    {
        "name": "baseline",
        "params": dict(
            # Color / lighting variation
            hsv_h=0.03, hsv_s=0.6, hsv_v=0.5,
            # Mild geometry
            fliplr=0.5,
            translate=0.1,
            scale=0.1,
            # mosaic intentionally NOT set
        ),
    },
    {
        "name": "moderate_geom",
        "params": dict(
            hsv_h=0.02, hsv_s=0.4, hsv_v=0.4,
            fliplr=0.5,
            flipud=0.1,
            translate=0.15,
            scale=0.2,
            degrees=10.0,
        ),
    },
    {
        "name": "strong_geom",
        "params": dict(
            hsv_h=0.02, hsv_s=0.4, hsv_v=0.4,
            fliplr=0.5,
            flipud=0.2,
            translate=0.2,
            scale=0.3,
            degrees=15.0,
            shear=5.0,
            perspective=0.0005,
        ),
    },
    {
        "name": "strong_geom_fruit",
        "params": dict(
            hsv_h=0.02, hsv_s=0.4, hsv_v=0.4,
            fliplr=0.5,
            flipud=0.1,      # Reduziert von 0.2
            translate=0.2,
            scale=0.25,      # Leicht reduziert von 0.3
            degrees=15.0,
            # shear=5.0 removed - not suitable for fruit
            perspective=0.0003,  # Leicht reduziert
        ),
    },
]


# ============================================================
# 4) HELPER FUNCTIONS
# ============================================================
def format_time(seconds: float) -> str:
    """Format seconds into h/m/s."""
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m {s}s"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"


def safe_float(x: Any) -> Optional[float]:
    """Robust float-Parsing."""
    try:
        return float(x)
    except Exception:
        return None


def compute_f1(p: Optional[float], r: Optional[float]) -> Optional[float]:
    """Compute F1 from precision and recall."""
    if p is None or r is None:
        return None
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def find_col(keys: List[str], candidates: List[str]) -> Optional[str]:
    """
    Find a column in `results.csv`.

    Ultralytics column names can vary slightly between versions, so we match by substring
    (case-insensitive).
    """
    for cand in candidates:
        cl = cand.lower()
        for k in keys:
            if cl in k.lower():
                return k
    return None


def read_val_metrics(results_csv: Path) -> Dict[str, Optional[float]]:
    """
    Read `results.csv` from a `val()` run and extract mask metrics:
    - precision(M), recall(M)
    - segm mAP50, segm mAP50-95

    Then compute F1.
    """
    if not results_csv.exists():
        return {}

    with results_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows or reader.fieldnames is None:
        return {}

    keys = list(reader.fieldnames)

    # Candidate columns (masks first, then fallback)
    col_p = find_col(keys, ["metrics/precision(m)", "metrics/precision"])
    col_r = find_col(keys, ["metrics/recall(m)", "metrics/recall"])
    col_m50 = find_col(keys, ["metrics/segm_map50", "metrics/segm_mAP50", "metrics/map50"])
    col_m5095 = find_col(keys, ["metrics/segm_map50-95", "metrics/segm_mAP50-95", "metrics/map50-95"])

    def best(col: Optional[str]) -> Optional[float]:
        if col is None:
            return None
        vals = []
        for r in rows:
            v = safe_float(r.get(col))
            if v is not None:
                vals.append(v)
        return max(vals) if vals else None

    p = best(col_p)
    r = best(col_r)

    return {
        "precision": p,
        "recall": r,
        "f1": compute_f1(p, r),
        "mAP50": best(col_m50),
        "mAP50-95": best(col_m5095),
        "_cols_used": {"precision": col_p, "recall": col_r, "mAP50": col_m50, "mAP50-95": col_m5095},
    }


# ============================================================
# 5) One run: train -> save best.pt -> validate -> read metrics
# ============================================================
def run_experiment(idx: int, exp: Dict, weights_path: Path, model_name: str, dataset_name: str, run_timestamp: str, data_yaml: Path, total_experiments: int) -> Dict:
    name = exp["name"]
    params = exp["params"]

    print("\n" + "=" * 90)
    print(f"▶ Training {idx + 1}/{total_experiments}: {name}")
    print("=" * 90)

    t0 = time.time()

    # Load the model fresh (no continuing from the previous run)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")
    
    print(f"📦 Loading model from: {weights_path}")
    model = YOLO(str(weights_path))

    # Timestamp with underscores for folder structure
    timestamp_underscore = run_timestamp.replace("-", "_")
    
    # Ultralytics uses `project` and `name` to build the output folder.
    # In this repo we expect segmentation runs under:
    #   runs/segment/{model_name}/{timestamp}/{experiment}/...
    # Here:
    # - `project_name` selects the model/dataset group
    # - `unique_name` encodes run timestamp (folder) and experiment name (subfolder)
    project_name = model_name  # -> runs/segment/{model_name}/
    unique_name = f"{timestamp_underscore}/{name}"

    # Train args: base settings + augmentation parameters
    train_args = dict(
        data=str(data_yaml),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=project_name,  # -> runs/segment/{model_name}/
        name=unique_name,      # -> .../{timestamp_underscore}/{experiment}/
        amp=True,  # AMP enabled (yolo26n.pt is available)
        plots=True,  # enable YOLO plots (train_batch*.jpg, labels.jpg, confusion matrix, etc.)
    )
    train_args.update(params)

    # 1) TRAIN
    results = model.train(**train_args)
    train_dir = Path(results.save_dir)

    # 2) best.pt finden
    best_pt = train_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found: {best_pt}")

    # 3) best.pt versioniert kopieren
    out_dir = project_root / "yolo_weights"
    out_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    # `model_name` already includes the dataset name (e.g. "yolo11n-seg_data_640"), so do not add it again
    exported = out_dir / f"{model_name}-demo-dataset-{name}-{ts}.pt"
    shutil.copy2(best_pt, exported)
    print(f"✅ Exported best.pt to: {exported}")

    # 4) Validation (fair: same conf/iou/imgsz/batch for all runs)
    print("🔍 Validation (fixed conf/iou) ...")
    val_model = YOLO(str(best_pt))
    # Validation name: same structure as training, but with a `VAL_` prefix
    # Ultralytics creates: runs/{project}/{task}/val/{name}/
    val_name = f"{timestamp_underscore}/VAL_{name}"
    val_args = {**get_val_args(data_yaml), "project": project_name, "name": val_name}
    val_results = val_model.val(**val_args)
    val_dir = Path(val_results.save_dir)

    # 5) Extract metrics (first from the `val_results` object, then fallback to CSV)
    metrics = {}
    
    # Try extracting metrics directly from `val_results`
    if hasattr(val_results, "seg"):
        seg = val_results.seg
        p = safe_float(getattr(seg, "p", None))
        r = safe_float(getattr(seg, "r", None))
        mAP50 = safe_float(getattr(seg, "map50", None))
        mAP50_95 = safe_float(getattr(seg, "map", None))
        
        if p is not None and r is not None:
            metrics["precision"] = p
            metrics["recall"] = r
            metrics["f1"] = compute_f1(p, r)
            metrics["mAP50"] = mAP50
            metrics["mAP50-95"] = mAP50_95
    
    # Fallback: try reading from results.csv (if present)
    if not metrics.get("precision"):
        csv_metrics = read_val_metrics(val_dir / "results.csv")
        if csv_metrics.get("precision") is not None:
            metrics = csv_metrics

    dt = time.time() - t0
    print(f"⏱ Duration for run '{name}': {format_time(dt)}")

    if metrics and metrics.get("precision") is not None:
        print(
            f"📊 Val(M): P={metrics.get('precision'):.3f}  "
            f"R={metrics.get('recall'):.3f}  "
            f"F1={metrics.get('f1'):.3f}  "
            f"mAP50={metrics.get('mAP50'):.3f}  "
            f"mAP50-95={metrics.get('mAP50-95'):.3f}"
        )
    else:
        print("⚠️ Could not read metrics from results.csv (column names may differ).")

    return {
        "name": name,
        "train_dir": train_dir,
        "val_dir": val_dir,
        "best_pt": best_pt,
        "exported": exported,
        "metrics": metrics,
        "duration_s": dt,
    }


# ============================================================
# 6) Main: run experiments sequentially + summary CSV + best model
# ============================================================
def main():
    # Parse Command-Line Arguments
    parser = argparse.ArgumentParser(description='Train YOLO Segmentation Model')
    parser.add_argument(
        '--model',
        type=str,
        choices=['yolo11n-seg', 'yolo26n-seg'],
        default='yolo11n-seg',
        help='YOLO model choice: yolo11n-seg or yolo26n-seg (default: yolo11n-seg)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['data_420', 'data_640', 'fruit_dataset_640', 'data_640_demo_day'],
        default='data_640',
        help='Dataset choice: data_420, data_640, fruit_dataset_640, or data_640_demo_day (default: data_640)'
    )
    parser.add_argument(
        '--augmentation',
        type=str,
        choices=['no_augmentation', 'baseline', 'moderate_geom', 'strong_geom', 'strong_geom_fruit', 'all'],
        default='all',
        help='Augmentation choice: no_augmentation, baseline, moderate_geom, strong_geom, strong_geom_fruit, or all (default: all)'
    )
    args = parser.parse_args()
    
    # Filter experiments based on --augmentation
    if args.augmentation == 'all':
        experiments_to_run = EXPERIMENTS
    else:
        experiments_to_run = [exp for exp in EXPERIMENTS if exp['name'] == args.augmentation]
        if not experiments_to_run:
            raise ValueError(f"Augmentation '{args.augmentation}' not found!")
    
    # Set dataset path
    dataset_name = args.dataset
    DATA_YAML = project_root / dataset_name / "data.yaml"
    
    # Set weights path based on model selection
    base_model_name = args.model
    # Model name with dataset: yolo11n-seg_data_640
    model_name = f"{base_model_name}_{dataset_name}"
    WEIGHTS_PATH = (project_root / "yolo_weights" / f"{base_model_name}.pt").resolve()
    
    # Quick sanity checks to avoid running with invalid paths
    print("=" * 90)
    print(f"🚀 YOLO training with model: {base_model_name}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🔧 Augmentation: {args.augmentation} ({len(experiments_to_run)} Experiment(e))")
    print(f"🏷️  Model name (with dataset): {model_name}")
    print("=" * 90)
    print("project_root:", project_root)
    print("WEIGHTS_PATH:", WEIGHTS_PATH, "exists:", WEIGHTS_PATH.exists())
    print("DATA_YAML   :", DATA_YAML, "exists:", DATA_YAML.exists())
    
    # Check that the weights exist
    if not WEIGHTS_PATH.exists():
        WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(
            f"\n❌ Error: Weights not found: {WEIGHTS_PATH}\n\n"
            f"📥 Please download the {model_name} weights manually:\n"
            f"   URL: https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}.pt\n"
            f"   Or: https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_name}.pt\n\n"
            f"   Save the file to: {WEIGHTS_PATH}\n\n"
            f"   Example command (on a system with internet):\n"
            f"   wget https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}.pt -O {WEIGHTS_PATH}\n"
        )
    
    print(f"✅ Weights found: {WEIGHTS_PATH} ({WEIGHTS_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Copy `yolo26n.pt` into the Ultralytics weights cache, if needed for AMP checks.
    yolo26n_pt = project_root / "yolo_weights" / "yolo26n.pt"
    if yolo26n_pt.exists():
        try:
            from ultralytics.utils import SETTINGS
            cache_dir = Path(SETTINGS.get("weights_dir", Path.home() / ".ultralytics" / "weights"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "yolo26n.pt"
            if not cache_file.exists():
                shutil.copy2(yolo26n_pt, cache_file)
                print(f"✅ yolo26n.pt copied into the Ultralytics cache: {cache_file}")
        except Exception as e:
            print(f"⚠️ Could not copy yolo26n.pt into the cache: {e}")
    
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml not found: {DATA_YAML}\n"
            f"Please ensure the dataset directory '{dataset_name}' exists and contains a data.yaml."
        )

    # Unique timestamp for this training run (all experiments share the same timestamp)
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"📅 Run timestamp: {run_timestamp}")

    overall0 = time.time()
    all_results = []

    # Training runs (filtered by --augmentation)
    for i, exp in enumerate(experiments_to_run):
        all_results.append(run_experiment(i, exp, WEIGHTS_PATH, model_name, dataset_name, run_timestamp, DATA_YAML, len(experiments_to_run)))

        elapsed = time.time() - overall0
        avg = elapsed / (i + 1)
        remaining = avg * (len(experiments_to_run) - (i + 1))
        print("-" * 90)
        print(f"⏱ Total so far: {format_time(elapsed)} | ⏳ Remaining (~): {format_time(remaining)}")
        print("-" * 90)

    # -----------------------
    # Summary CSV (1 row per run) -> for Excel/presentation
    # -----------------------
    out_dir = project_root / "yolo_weights"
    out_dir.mkdir(exist_ok=True)

    summary_csv = out_dir / f"summary_{model_name}_{EPOCHS}epochs_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["experiment", "precision_M", "recall_M", "f1_M", "mAP50_M", "mAP50-95_M", "epochs", "weights_file"])
        for r in all_results:
            m = r.get("metrics", {})
            writer.writerow([
                r["name"],
                m.get("precision"),
                m.get("recall"),
                m.get("f1"),
                m.get("mAP50"),
                m.get("mAP50-95"),
                EPOCHS,
                str(r["exported"]),
            ])

    print(f"\n📄 Summary CSV saved: {summary_csv}")

    # -----------------------
    # Select the best model by validation F1
    # -----------------------
    best_name = None
    best_f1 = float("-inf")
    best_pt = None

    print("\n" + "#" * 90)
    print("📊 SUMMARY (only val() matters)")
    print("#" * 90)

    for r in all_results:
        m = r.get("metrics", {})
        f1v = m.get("f1")
        print(f"- {r['name']}: P={m.get('precision')} R={m.get('recall')} F1={m.get('f1')} mAP50={m.get('mAP50')}")

        if isinstance(f1v, float) and f1v > best_f1:
            best_f1 = f1v
            best_name = r["name"]
            best_pt = r["best_pt"]

    if best_pt is not None:
        stable = out_dir / f"{model_name}-demo-dataset-BEST-overall.pt"
        shutil.copy2(best_pt, stable)
        print("\n🏆 BEST (based on validation F1)")
        print(f"   Name: {best_name}")
        print(f"   F1  : {best_f1:.3f}")
        print(f"   Saved as: {stable}")
    else:
        print("\n⚠️ Could not determine the best model (F1 not available).")

    print(f"\n🎉 Done. Total runtime: {format_time(time.time() - overall0)}")


if __name__ == "__main__":
    main()
