"""
Compare trained YOLO segmentation models across augmentation experiments.

The script loads `best.pt` checkpoints, runs evaluation on the validation set with fixed
parameters, and writes a comparison table with all important metrics.
"""

import csv
from pathlib import Path
from typing import Dict, Optional, List
from ultralytics import YOLO

# Project root: repository root (this file lives in scripts/local/)
project_root = Path(__file__).resolve().parents[2]
DATA_YAML = project_root / "data_640_demo_day" / "data.yaml"

# Validation parameters (kept identical for a fair comparison)
VAL_ARGS = dict(
    data=str(DATA_YAML),
    imgsz=640,
    batch=16,
    device=0,
    conf=0.25,      # fixed confidence threshold
    iou=0.50,       # fixed IoU threshold
    split="val",
    save_json=False,  # no JSON output
    plots=False,     # no plots (faster)
)

# Models and experiments
# IMPORTANT: model names must end with `_fruit_dataset_640` because those folders are named that way.
# MODELS = ["yolo11n-seg_fruit_dataset_640", "yolo11n-seg_data_640_demo_day", "yolo26n-seg_fruit_dataset_640"]
MODELS = ["yolo11n-seg_fruit_dataset_640", "yolo11n-seg_data_640_demo_day"]
EXPERIMENTS = ["no_augmentation", "baseline", "moderate_geom", "strong_geom", "strong_geom_fruit"]

# Timestamps per model (different for each model)
MODEL_TIMESTAMPS = {
    "yolo11n-seg_fruit_dataset_640": "20260128_190716",
    "yolo26n-seg_fruit_dataset_640": "20260128_193214",
    "yolo11n-seg_data_640_demo_day": "20260312_121409",
}


def safe_float(x: any) -> Optional[float]:
    """Robust float-Parsing."""
    try:
        return float(x)
    except (ValueError, TypeError):
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
    col_p_m = find_col(keys, ["metrics/precision(m)", "metrics/precision"])
    col_r_m = find_col(keys, ["metrics/recall(m)", "metrics/recall"])
    col_m50_m = find_col(keys, ["metrics/segm_map50", "metrics/segm_mAP50", "metrics/map50"])
    col_m5095_m = find_col(keys, ["metrics/segm_map50-95", "metrics/segm_mAP50-95", "metrics/map50-95"])
    
    # Box metrics
    col_p_b = find_col(keys, ["metrics/precision(b)", "metrics/precision"])
    col_r_b = find_col(keys, ["metrics/recall(b)", "metrics/recall"])
    col_m50_b = find_col(keys, ["metrics/box_map50", "metrics/box_mAP50", "metrics/map50"])
    col_m5095_b = find_col(keys, ["metrics/box_map50-95", "metrics/box_mAP50-95", "metrics/map50-95"])

    def best(col: Optional[str]) -> Optional[float]:
        if col is None:
            return None
        vals = []
        for r in rows:
            v = safe_float(r.get(col))
            if v is not None:
                vals.append(v)
        return max(vals) if vals else None

    p_m = best(col_p_m)
    r_m = best(col_r_m)
    p_b = best(col_p_b)
    r_b = best(col_r_b)

    return {
        "precision_M": p_m,
        "recall_M": r_m,
        "f1_M": compute_f1(p_m, r_m),
        "mAP50_M": best(col_m50_m),
        "mAP50-95_M": best(col_m5095_m),
        "precision_B": p_b,
        "recall_B": r_b,
        "mAP50_B": best(col_m50_b),
        "mAP50-95_B": best(col_m5095_b),
    }


def evaluate_model(model_path: Path, model_name: str, experiment: str) -> Dict:
    """
    Evaluates a single model and returns metrics.
    """
    print(f"\n🔍 Evaluating: {model_name} - {experiment}")
    print(f"   Model: {model_path.name}")
    
    if not model_path.exists():
        print(f"   ⚠️ Model not found!")
        return {
            "model": model_name,
            "experiment": experiment,
            "precision_M": None,
            "recall_M": None,
            "f1_M": None,
            "mAP50_M": None,
            "mAP50-95_M": None,
            "precision_B": None,
            "recall_B": None,
            "mAP50_B": None,
            "mAP50-95_B": None,
        }
    
    try:
        # Retrieve timestamp for this model
        timestamp = MODEL_TIMESTAMPS.get(model_name)
        if not timestamp:
            raise ValueError(
                f"No timestamp found for model '{model_name}'. Available: {list(MODEL_TIMESTAMPS.keys())}"
            )
        
        # Validate that `model_path` is inside the expected runs directory
        expected_base = project_root / "runs" / "segment" / model_name / timestamp
        if not str(model_path).startswith(str(expected_base)):
            raise ValueError(f"Model path is outside the expected runs directory: {model_path}")
        
        # Load model
        model = YOLO(str(model_path))
        
        # Extract metrics from `results.csv` in the training directory.
        # Expected structure: runs/segment/{model_name}/{timestamp}/{experiment}/results.csv
        # model_path is: runs/segment/{model_name}/{timestamp}/{experiment}/weights/best.pt
        train_dir = model_path.parent.parent  # Von weights/best.pt zu experiment/
        
        # Validate that `train_dir` is inside the expected runs directory
        if not str(train_dir).startswith(str(expected_base)):
            raise ValueError(f"Train directory is outside the expected runs directory: {train_dir}")
        
        results_csv = train_dir / "results.csv"
        
        # Try reading metrics from the training results directory first
        metrics = read_val_metrics(results_csv)
        
        # Fallback: if the CSV does not exist or is empty, run a new validation pass
        if not metrics.get("f1_M"):
            print("   🔄 Running a new validation pass (CSV not found or empty)...")
            val_args = VAL_ARGS.copy()
            val_args['plots'] = True
            # Speichere Validation-Output im evaluation Verzeichnis
            val_args['project'] = str(project_root / "evaluation")
            val_args['name'] = f"val_{model_name}_{experiment}"
            results = model.val(**val_args, verbose=False)
            
            # Try extracting metrics directly from the `results` object
            if hasattr(results, 'seg'):
                seg = results.seg
                p_m = safe_float(getattr(seg, 'p', None))
                r_m = safe_float(getattr(seg, 'r', None))
                map50_m = safe_float(getattr(seg, 'map50', None))
                map5095_m = safe_float(getattr(seg, 'map', None))
                
                if p_m is not None and r_m is not None:
                    metrics["precision_M"] = p_m
                    metrics["recall_M"] = r_m
                    metrics["f1_M"] = compute_f1(p_m, r_m)
                    metrics["mAP50_M"] = map50_m
                    metrics["mAP50-95_M"] = map5095_m
            
            # Box metrics
            if hasattr(results, 'box'):
                box = results.box
                metrics["precision_B"] = safe_float(getattr(box, 'p', None))
                metrics["recall_B"] = safe_float(getattr(box, 'r', None))
                metrics["mAP50_B"] = safe_float(getattr(box, 'map50', None))
                metrics["mAP50-95_B"] = safe_float(getattr(box, 'map', None))
        
        result_dict = {
            "model": model_name,
            "experiment": experiment,
            **metrics
        }
        
        if metrics.get("f1_M") is not None:
            print(f"   ✅ Precision(M): {metrics.get('precision_M', 'N/A'):.3f}")
            print(f"   ✅ Recall(M): {metrics.get('recall_M', 'N/A'):.3f}")
            print(f"   ✅ F1(M): {metrics.get('f1_M', 'N/A'):.3f}")
            print(f"   ✅ mAP50(M): {metrics.get('mAP50_M', 'N/A'):.3f}")
            print(f"   ✅ mAP50-95(M): {metrics.get('mAP50-95_M', 'N/A'):.3f}")
        else:
            print(f"   ⚠️ No metrics available")
        
        return result_dict
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return {
            "model": model_name,
            "experiment": experiment,
            "precision_M": None,
            "recall_M": None,
            "f1_M": None,
            "mAP50_M": None,
            "mAP50-95_M": None,
            "precision_B": None,
            "recall_B": None,
            "mAP50_B": None,
            "mAP50-95_B": None,
            "error": str(e),
        }


def main():
    print("=" * 90)
    print("📊 MODEL COMPARISON: All augmentation variants")
    print("=" * 90)
    print(f"Validation parameters: conf={VAL_ARGS['conf']}, iou={VAL_ARGS['iou']}")
    print(f"Dataset: {DATA_YAML}")
    print("Timestamps per model:")
    for model_name, timestamp in MODEL_TIMESTAMPS.items():
        print(f"  - {model_name}: {timestamp}")
    
    # Create evaluation directory for all outputs
    evaluation_dir = project_root / "evaluation"
    evaluation_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {evaluation_dir}")
    
    all_results = []
    
    # Evaluate all models
    for model_name in MODELS:
        timestamp = MODEL_TIMESTAMPS.get(model_name)
        if not timestamp:
            print(f"⚠️ No timestamp found for {model_name}, skipping...")
            continue
            
        for experiment in EXPERIMENTS:
            # Find the `best.pt` checkpoint for this run/experiment
            # Expected structure: runs/segment/{model_name}/{timestamp}/{experiment}/weights/best.pt
            # IMPORTANT: only use runs from the configured folders/timestamps
            model_path = (
                project_root / "runs" / "segment" / model_name / 
                timestamp / experiment / "weights" / "best.pt"
            )
            
            # Validate that the path is inside the expected runs directory
            expected_base = project_root / "runs" / "segment" / model_name / timestamp
            if not str(model_path).startswith(str(expected_base)):
                raise ValueError(f"Path is outside the expected runs directory: {model_path}")
            
            result = evaluate_model(model_path, model_name, experiment)
            all_results.append(result)
    
    # Speichere als CSV im evaluation Verzeichnis
    # Use a combined timestamp for the filename
    combined_timestamp = "_".join(sorted(MODEL_TIMESTAMPS.values()))
    output_csv = evaluation_dir / f"model_comparison_{combined_timestamp}.csv"
    
    fieldnames = [
        "model", "experiment",
        "precision_M", "recall_M", "f1_M", "mAP50_M", "mAP50-95_M",
        "precision_B", "recall_B", "mAP50_B", "mAP50-95_B"
    ]
    
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            # Nur relevante Felder schreiben
            row = {k: result.get(k) for k in fieldnames}
            writer.writerow(row)
    
    print(f"\n📄 Comparison saved: {output_csv}")
    
    # Create a nice overview
    print("\n" + "=" * 90)
    print("📊 COMPARISON OVERVIEW")
    print("=" * 90)
    
    # Sort by F1 (Mask) for a clearer overview
    valid_results = [r for r in all_results if r.get("f1_M") is not None]
    sorted_results = sorted(valid_results, key=lambda x: x.get("f1_M", 0), reverse=True)
    
    print("\n🏆 Ranking by F1-score (Mask):")
    print("-" * 90)
    print(f"{'Experiment':<20} | {'Model':<15} | {'F1':<6} | {'P':<6} | {'R':<6} | {'mAP50':<7} | {'mAP50-95':<8}")
    print("-" * 90)
    
    for r in sorted_results:
        exp = r["experiment"]
        model = r["model"]
        f1 = r.get("f1_M", 0)
        p = r.get("precision_M", 0)
        r_val = r.get("recall_M", 0)
        mAP50 = r.get("mAP50_M", 0)
        mAP50_95 = r.get("mAP50-95_M", 0)
        
        print(f"{exp:<20} | {model:<15} | {f1:<6.3f} | {p:<6.3f} | {r_val:<6.3f} | {mAP50:<7.3f} | {mAP50_95:<8.3f}")
    
    # Comparison by experiment
    print("\n" + "=" * 90)
    print("📈 COMPARISON BY EXPERIMENT")
    print("=" * 90)
    
    for exp in EXPERIMENTS:
        exp_results = [r for r in all_results if r["experiment"] == exp and r.get("f1_M") is not None]
        if exp_results:
            exp_results_sorted = sorted(exp_results, key=lambda x: x.get("f1_M", 0), reverse=True)
            print(f"\n🔬 {exp.upper()}:")
            print("-" * 90)
            print(f"{'Model':<15} | {'F1':<6} | {'P':<6} | {'R':<6} | {'mAP50':<7} | {'mAP50-95':<8}")
            print("-" * 90)
            for r in exp_results_sorted:
                model = r["model"]
                f1 = r.get("f1_M", 0)
                p = r.get("precision_M", 0)
                r_val = r.get("recall_M", 0)
                mAP50 = r.get("mAP50_M", 0)
                mAP50_95 = r.get("mAP50-95_M", 0)
                print(f"{model:<15} | {f1:<6.3f} | {p:<6.3f} | {r_val:<6.3f} | {mAP50:<7.3f} | {mAP50_95:<8.3f}")
        else:
            print(f"\n🔬 {exp.upper()}: ⚠️ No metrics available")
    
    # Comparison by model
    print("\n" + "=" * 90)
    print("🤖 COMPARISON BY MODEL")
    print("=" * 90)
    
    for model in MODELS:
        model_results = [r for r in all_results if r["model"] == model and r.get("f1_M") is not None]
        if model_results:
            model_results_sorted = sorted(model_results, key=lambda x: x.get("f1_M", 0), reverse=True)
            print(f"\n🔬 {model.upper()}:")
            print("-" * 90)
            print(f"{'Experiment':<20} | {'F1':<6} | {'P':<6} | {'R':<6} | {'mAP50':<7} | {'mAP50-95':<8}")
            print("-" * 90)
            for r in model_results_sorted:
                exp = r["experiment"]
                f1 = r.get("f1_M", 0)
                p = r.get("precision_M", 0)
                r_val = r.get("recall_M", 0)
                mAP50 = r.get("mAP50_M", 0)
                mAP50_95 = r.get("mAP50-95_M", 0)
                print(f"{exp:<20} | {f1:<6.3f} | {p:<6.3f} | {r_val:<6.3f} | {mAP50:<7.3f} | {mAP50_95:<8.3f}")
        else:
            print(f"\n🔬 {model.upper()}: ⚠️ No metrics available")
    
    # Best overall model
    if sorted_results:
        best_row = sorted_results[0]
        print("\n" + "=" * 90)
        print("🏆 BEST OVERALL MODEL")
        print("=" * 90)
        print(f"Model: {best_row['model']}")
        print(f"Experiment: {best_row['experiment']}")
        print(f"F1-Score (Mask): {best_row['f1_M']:.3f}")
        print(f"Precision (Mask): {best_row['precision_M']:.3f}")
        print(f"Recall (Mask): {best_row['recall_M']:.3f}")
        print(f"mAP50 (Mask): {best_row['mAP50_M']:.3f}")
        print(f"mAP50-95 (Mask): {best_row['mAP50-95_M']:.3f}")
        
        # Path to the best model
        best_timestamp = MODEL_TIMESTAMPS.get(best_row['model'])
        if best_timestamp:
            best_model_path = (
                project_root / "runs" / "segment" / best_row['model'] / 
                best_timestamp / best_row['experiment'] / "weights" / "best.pt"
            )
            print(f"\nBest model path: {best_model_path}")
    
    print("\n" + "=" * 90)
    print("✅ Comparison completed!")
    print("=" * 90)
    print(f"\n💡 Tips for interpretation:")
    print("   - F1-score: balance between Precision and Recall (higher = better)")
    print("   - mAP50: mean average precision at IoU=0.5 (important for detection)")
    print("   - mAP50-95: mean average precision averaged over IoU 0.5-0.95 (stricter)")
    print("   - Precision: how many predicted objects are correct?")
    print("   - Recall: how many ground-truth objects were found?")
    print("\n   For segmentation, mask metrics (M) are more important than box metrics (B)")
    print("   A high F1-score together with a high mAP50-95 is ideal!")
    print(f"\n📄 Detailed results in: {output_csv}")
    print(f"📁 All outputs saved in: {evaluation_dir}")


if __name__ == "__main__":
    main()
