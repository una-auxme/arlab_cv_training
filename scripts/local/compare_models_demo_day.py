"""
Compare trained YOLO segmentation models for the `data_640_demo_day` dataset.

The script loads `best.pt` checkpoints, evaluates them on the validation set using fixed
parameters, and writes a comparison table with the main metrics.
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
    plots=True,      # plots for visualization
)

# Models and experiments for the demo_day dataset
MODELS = ["yolo11n-seg_data_640_demo_day"]
EXPERIMENTS = ["no_augmentation", "baseline", "moderate_geom", "strong_geom", "strong_geom_fruit"]

# Timestamps per model
MODEL_TIMESTAMPS = {
    "yolo11n-seg_data_640_demo_day": "20260312_121409",
    # Füge hier weitere Timestamps hinzu, wenn du mehr Modelle trainierst:
    # "yolo26n-seg_data_640_demo_day": "TIMESTAMP_HIER",
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
    print(f"   Model: {model_path}")
    
    if not model_path.exists():
        print(f"   ⚠️ Model not found: {model_path}")
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
            raise ValueError(f"Kein Timestamp für Modell '{model_name}' gefunden. Verfügbare: {list(MODEL_TIMESTAMPS.keys())}")
        
        # Lade Modell
        model = YOLO(str(model_path))
        
        # Extract metrics from `results.csv` in the training directory
        train_dir = model_path.parent.parent  # Von weights/best.pt zu experiment/
        results_csv = train_dir / "results.csv"
        
        # Try reading metrics from the training results directory first
        metrics = read_val_metrics(results_csv)
        
        # Fallback: if the CSV is missing/empty, run a new validation pass
        if not metrics.get("f1_M"):
            print(f"   🔄 Führe neue Validation durch (CSV nicht gefunden oder leer)...")
            val_args = VAL_ARGS.copy()
            # Store validation output in the evaluation directory
            val_args['project'] = str(project_root / "evaluation" / "demo_day")
            val_args['name'] = f"val_{model_name}_{experiment}"
            results = model.val(**val_args, verbose=False)
            
            # Versuche Metriken direkt aus results zu extrahieren
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
    print("📊 MODEL COMPARISON: data_640_demo_day Dataset")
    print("=" * 90)
    print(f"Validation-Parameter: conf={VAL_ARGS['conf']}, iou={VAL_ARGS['iou']}")
    print(f"Dataset: {DATA_YAML}")
    print(f"Timestamps pro Modell:")
    for model_name, timestamp in MODEL_TIMESTAMPS.items():
        print(f"  - {model_name}: {timestamp}")
    
    # Create evaluation directory for all outputs
    evaluation_dir = project_root / "evaluation" / "demo_day"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    print(f"📁 Output-Verzeichnis: {evaluation_dir}")
    
    all_results = []
    
    # Evaluiere alle Modelle
    for model_name in MODELS:
        timestamp = MODEL_TIMESTAMPS.get(model_name)
        if not timestamp:
            print(f"⚠️ Kein Timestamp für {model_name} gefunden, überspringe...")
            continue
            
        for experiment in EXPERIMENTS:
            # Find the `best.pt` checkpoint for this run/experiment
            # Expected structure: runs/segment/{model_name}/{timestamp}/{experiment}/weights/best.pt
            model_path = (
                project_root / "runs" / "segment" / model_name / 
                timestamp / experiment / "weights" / "best.pt"
            )
            
            result = evaluate_model(model_path, model_name, experiment)
            all_results.append(result)
    
    # Speichere als CSV im evaluation Verzeichnis
    combined_timestamp = "_".join(sorted(MODEL_TIMESTAMPS.values()))
    output_csv = evaluation_dir / f"model_comparison_demo_day_{combined_timestamp}.csv"
    
    fieldnames = [
        "model", "experiment",
        "precision_M", "recall_M", "f1_M", "mAP50_M", "mAP50-95_M",
        "precision_B", "recall_B", "mAP50_B", "mAP50-95_B"
    ]
    
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in all_results:
            # Write only relevant fields
            row = {k: result.get(k) for k in fieldnames}
            writer.writerow(row)
    
    print(f"\n📄 Vergleich gespeichert: {output_csv}")
    
    # Create a nice overview
    print("\n" + "=" * 90)
    print("📊 VERGLEICHSÜBERSICHT - Demo Day Dataset")
    print("=" * 90)
    
    # Sortiere nach F1 (Mask) für bessere Übersicht
    valid_results = [r for r in all_results if r.get("f1_M") is not None]
    sorted_results = sorted(valid_results, key=lambda x: x.get("f1_M", 0), reverse=True)
    
    print("\n🏆 RANKING nach F1-Score (Mask):")
    print("-" * 90)
    print(f"{'Rank':<5} | {'Experiment':<20} | {'Model':<30} | {'F1':<6} | {'P':<6} | {'R':<6} | {'mAP50':<7}")
    print("-" * 90)
    
    for i, r in enumerate(sorted_results, 1):
        exp = r["experiment"]
        model = r["model"]
        f1 = r.get("f1_M", 0)
        p = r.get("precision_M", 0)
        r_val = r.get("recall_M", 0)
        mAP50 = r.get("mAP50_M", 0)
        
        rank_emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else f" {i}"
        print(f"{rank_emoji:<5} | {exp:<20} | {model:<30} | {f1:<6.3f} | {p:<6.3f} | {r_val:<6.3f} | {mAP50:<7.3f}")
    
    # Nicht gefundene Experimente
    missing = [r for r in all_results if r.get("f1_M") is None]
    if missing:
        print("\n⚠️ NICHT GEFUNDENE EXPERIMENTE:")
        print("-" * 90)
        for r in missing:
            print(f"   - {r['model']} / {r['experiment']}")
    
    # Bestes Modell insgesamt
    if sorted_results:
        best_row = sorted_results[0]
        print("\n" + "=" * 90)
        print("🏆 BESTES MODELL")
        print("=" * 90)
        print(f"Modell: {best_row['model']}")
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
            print(f"\nModel-Pfad: {best_model_path}")
    
    print("\n" + "=" * 90)
    print("✅ Vergleich abgeschlossen!")
    print("=" * 90)
    print(f"\n📄 Detaillierte Ergebnisse in: {output_csv}")
    print(f"📁 Alle Ausgaben gespeichert in: {evaluation_dir}")


if __name__ == "__main__":
    main()
