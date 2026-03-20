"""
Trainiert 3 Varianten (baseline / moderate_geom / strong_geom) für demo-dataset Segmentation
- 50 Epochen pro Run (schneller Vergleich)
- ohne Mosaic
- nach jedem Training: best.pt versioniert speichern
- danach: sauberes model.val() mit fixen Parametern (conf/iou) -> fairer Vergleich
- liest Maskenmetriken aus val/results.csv und berechnet F1
- schreibt zusätzlich eine Summary-CSV (1 Zeile pro Run) für Excel/Präsi
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

# yolo26n.pt wird für AMP-Checks benötigt und wird automatisch in den Cache kopiert


# ============================================================
# 1) PFADE / SETTINGS
# ============================================================

# Projekt-Root: befindet sich im Repo-Root, die Datei liegt aber in scripts/local/
project_root = Path(__file__).resolve().parents[2]

# DATA_YAML wird dynamisch basierend auf --dataset Parameter gesetzt

# Trainingseinstellungen
EPOCHS = 5000
IMGSZ = 640
BATCH = 16          # RTX 3080: sicher. Wenn OOM: 4
DEVICE = 0         # GPU. CPU wäre -1
PATIENCE = 100

# ============================================================
# 2) VAL-SETTINGS (wichtig: fix für alle Runs!)
# ============================================================
# Damit Precision/Recall/F1 fair vergleichbar sind, müssen conf/iou fix sein.
# data wird dynamisch gesetzt
def get_val_args(data_yaml: Path) -> dict:
    return dict(
        data=str(data_yaml),
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        conf=0.25,      # feste Conf-Schwelle (für P/R/F1 Vergleich!)
        iou=0.50,       # feste IoU-Schwelle
        split="val",
    )

# ============================================================
# 3) 3 EXPERIMENTE (ohne mosaic)
# ============================================================
# Idee: nur GEOMETRIE variiert. Farbe bleibt realistisch aktiv.
EXPERIMENTS = [
    {
        "name": "no_augmentation",
        "params": dict(),
    },
    {
        "name": "baseline",
        "params": dict(
            # Farb- / Beleuchtungsvariation
            hsv_h=0.03, hsv_s=0.6, hsv_v=0.5,
            # leichte Geometrie
            fliplr=0.5,
            translate=0.1,
            scale=0.1,
            # mosaic bewusst NICHT gesetzt
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
            # shear=5.0 ENTFERNEN - unpassend für Obst!
            perspective=0.0003,  # Leicht reduziert
        ),
    },
]


# ============================================================
# 4) HILFSFUNKTIONEN
# ============================================================
def format_time(seconds: float) -> str:
    """Zeit in h/m/s formatieren."""
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
    """F1 aus Precision und Recall."""
    if p is None or r is None:
        return None
    if p + r == 0:
        return 0.0
    return 2.0 * p * r / (p + r)


def find_col(keys: List[str], candidates: List[str]) -> Optional[str]:
    """
    Findet eine Spalte in results.csv.
    Ultralytics-Spaltennamen können je nach Version leicht variieren.
    Deshalb matchen wir per substring (case-insensitive).
    """
    for cand in candidates:
        cl = cand.lower()
        for k in keys:
            if cl in k.lower():
                return k
    return None


def read_val_metrics(results_csv: Path) -> Dict[str, Optional[float]]:
    """
    Liest results.csv aus dem val()-Run und extrahiert Maskenmetriken:
    - precision(M), recall(M)
    - segm mAP50, segm mAP50-95
    und berechnet F1.
    """
    if not results_csv.exists():
        return {}

    with results_csv.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows or reader.fieldnames is None:
        return {}

    keys = list(reader.fieldnames)

    # Kandidatenlisten (Masken zuerst, dann fallback)
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
# 5) EIN RUN: TRAIN -> best.pt sichern -> VAL -> Metriken lesen
# ============================================================
def run_experiment(idx: int, exp: Dict, weights_path: Path, model_name: str, dataset_name: str, run_timestamp: str, data_yaml: Path, total_experiments: int) -> Dict:
    name = exp["name"]
    params = exp["params"]

    print("\n" + "=" * 90)
    print(f"▶ Training {idx + 1}/{total_experiments}: {name}")
    print("=" * 90)

    t0 = time.time()

    # Modell frisch laden (kein Weitertrainieren vom vorherigen Run)
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights nicht gefunden: {weights_path}")
    
    print(f"📦 Lade Modell von: {weights_path}")
    model = YOLO(str(weights_path))

    # Zeitstempel mit Unterstrichen für Ordnerstruktur
    timestamp_underscore = run_timestamp.replace("-", "_")
    
    # Ultralytics-Standardstruktur: runs/{project}/{task}/train/{name}/
    # Für Segmentation ist task="segment" automatisch
    # Wir setzen: project="yolo11n-seg", name="20260125_143000/no_augmentation"
    # Ergebnis: runs/yolo11n-seg/segment/train/20260125_143000/no_augmentation/
    # Die Modell-Unterscheidung ist im obersten Projekt-Ordner klar erkennbar
    project_name = model_name  # -> runs/yolo11n-seg/ oder runs/yolo26n-seg/
    unique_name = f"{timestamp_underscore}/{name}"

    # Train-Args: Basis + Augmentationsparameter
    train_args = dict(
        data=str(data_yaml),
        epochs=EPOCHS,
        patience=PATIENCE,
        imgsz=IMGSZ,
        batch=BATCH,
        device=DEVICE,
        project=project_name,  # -> runs/yolo11n-seg/ oder runs/yolo26n-seg/
        name=unique_name,  # -> runs/{model_name}/segment/train/20260125_143000/no_augmentation/
        amp=True,  # AMP aktiviert (yolo26n.pt ist jetzt verfügbar)
        plots=True,  # YOLO-Plots aktivieren (train_batch*.jpg, labels.jpg, confusion matrix, etc.)
    )
    train_args.update(params)

    # 1) TRAIN
    results = model.train(**train_args)
    train_dir = Path(results.save_dir)

    # 2) best.pt finden
    best_pt = train_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt nicht gefunden: {best_pt}")

    # 3) best.pt versioniert kopieren
    out_dir = project_root / "yolo_weights"
    out_dir.mkdir(exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    # model_name enthält bereits dataset_name (z.B. "yolo11n-seg_data_640"), daher nicht nochmal hinzufügen
    exported = out_dir / f"{model_name}-demo-dataset-{name}-{ts}.pt"
    shutil.copy2(best_pt, exported)
    print(f"✅ best.pt exportiert nach: {exported}")

    # 4) VAL (fair: gleiche conf/iou/imgsz/batch für ALLE Runs)
    print("🔍 Validation (fixed conf/iou) ...")
    val_model = YOLO(str(best_pt))
    # Validation-Name: gleiche Struktur wie Training, aber mit VAL_ Präfix
    # Ultralytics erstellt: runs/{project}/{task}/val/{name}/
    val_name = f"{timestamp_underscore}/VAL_{name}"
    val_args = {**get_val_args(data_yaml), "project": project_name, "name": val_name}
    val_results = val_model.val(**val_args)
    val_dir = Path(val_results.save_dir)

    # 5) Metriken extrahieren (zuerst aus val_results Objekt, dann Fallback auf CSV)
    metrics = {}
    
    # Versuche Metriken direkt aus val_results zu extrahieren
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
    
    # Fallback: Versuche aus results.csv zu lesen (falls vorhanden)
    if not metrics.get("precision"):
        csv_metrics = read_val_metrics(val_dir / "results.csv")
        if csv_metrics.get("precision") is not None:
            metrics = csv_metrics

    dt = time.time() - t0
    print(f"⏱ Dauer Run '{name}': {format_time(dt)}")

    if metrics and metrics.get("precision") is not None:
        print(
            f"📊 Val(M): P={metrics.get('precision'):.3f}  "
            f"R={metrics.get('recall'):.3f}  "
            f"F1={metrics.get('f1'):.3f}  "
            f"mAP50={metrics.get('mAP50'):.3f}  "
            f"mAP50-95={metrics.get('mAP50-95'):.3f}"
        )
    else:
        print("⚠️ Konnte keine Metriken aus results.csv lesen (Spaltennamen evtl. anders).")

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
# 6) MAIN: 3 Runs nacheinander + Summary CSV + bestes Modell
# ============================================================
def main():
    # Parse Command-Line Arguments
    parser = argparse.ArgumentParser(description='Train YOLO Segmentation Model')
    parser.add_argument(
        '--model',
        type=str,
        choices=['yolo11n-seg', 'yolo26n-seg'],
        default='yolo11n-seg',
        help='YOLO Modell auswählen: yolo11n-seg oder yolo26n-seg (default: yolo11n-seg)'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['data_420', 'data_640', 'fruit_dataset_640', 'data_640_demo_day'],
        default='data_640',
        help='Dataset auswählen: data_420, data_640, fruit_dataset_640 oder data_640_demo_day (default: data_640)'
    )
    parser.add_argument(
        '--augmentation',
        type=str,
        choices=['no_augmentation', 'baseline', 'moderate_geom', 'strong_geom', 'strong_geom_fruit', 'all'],
        default='all',
        help='Augmentation-Typ auswählen: no_augmentation, baseline, moderate_geom, strong_geom, strong_geom_fruit oder all (default: all)'
    )
    args = parser.parse_args()
    
    # Filtere Experimente basierend auf --augmentation
    if args.augmentation == 'all':
        experiments_to_run = EXPERIMENTS
    else:
        experiments_to_run = [exp for exp in EXPERIMENTS if exp['name'] == args.augmentation]
        if not experiments_to_run:
            raise ValueError(f"Augmentation '{args.augmentation}' nicht gefunden!")
    
    # Setze Dataset-Pfad
    dataset_name = args.dataset
    DATA_YAML = project_root / dataset_name / "data.yaml"
    
    # Setze Weights-Pfad basierend auf Modell-Auswahl
    base_model_name = args.model
    # Modell-Name mit Dataset: yolo11n-seg_data_640
    model_name = f"{base_model_name}_{dataset_name}"
    WEIGHTS_PATH = (project_root / "yolo_weights" / f"{base_model_name}.pt").resolve()
    
    # Mini-Checks, damit du nicht mit falschen Pfaden losläufst
    print("=" * 90)
    print(f"🚀 YOLO Training mit Modell: {base_model_name}")
    print(f"📊 Dataset: {dataset_name}")
    print(f"🔧 Augmentation: {args.augmentation} ({len(experiments_to_run)} Experiment(e))")
    print(f"🏷️  Modell-Name (mit Dataset): {model_name}")
    print("=" * 90)
    print("project_root:", project_root)
    print("WEIGHTS_PATH:", WEIGHTS_PATH, "exists:", WEIGHTS_PATH.exists())
    print("DATA_YAML   :", DATA_YAML, "exists:", DATA_YAML.exists())
    
    # Prüfe, ob Weights vorhanden sind
    if not WEIGHTS_PATH.exists():
        WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        raise FileNotFoundError(
            f"\n❌ FEHLER: Weights nicht gefunden: {WEIGHTS_PATH}\n\n"
            f"📥 Bitte lade die {model_name} Weights manuell herunter:\n"
            f"   URL: https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}.pt\n"
            f"   Oder: https://github.com/ultralytics/assets/releases/download/v8.3.0/{model_name}.pt\n\n"
            f"   Speichere die Datei nach: {WEIGHTS_PATH}\n\n"
            f"   Beispiel-Befehl (auf einem System mit Internet):\n"
            f"   wget https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}.pt -O {WEIGHTS_PATH}\n"
        )
    
    print(f"✅ Weights gefunden: {WEIGHTS_PATH} ({WEIGHTS_PATH.stat().st_size / 1024 / 1024:.1f} MB)")
    
    # Kopiere yolo26n.pt in Ultralytics Cache, falls es für AMP-Checks benötigt wird
    yolo26n_pt = project_root / "yolo_weights" / "yolo26n.pt"
    if yolo26n_pt.exists():
        try:
            from ultralytics.utils import SETTINGS
            cache_dir = Path(SETTINGS.get("weights_dir", Path.home() / ".ultralytics" / "weights"))
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_file = cache_dir / "yolo26n.pt"
            if not cache_file.exists():
                shutil.copy2(yolo26n_pt, cache_file)
                print(f"✅ yolo26n.pt in Ultralytics Cache kopiert: {cache_file}")
        except Exception as e:
            print(f"⚠️ Konnte yolo26n.pt nicht in Cache kopieren: {e}")
    
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"data.yaml nicht gefunden: {DATA_YAML}\n"
            f"Bitte stelle sicher, dass das Dataset-Verzeichnis '{dataset_name}' existiert und eine data.yaml enthält."
        )

    # Eindeutiger Zeitstempel für diesen Training-Run (alle Experimente teilen sich denselben)
    run_timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    print(f"📅 Run-Timestamp: {run_timestamp}")

    overall0 = time.time()
    all_results = []

    # Trainingsläufe (gefiltert nach --augmentation)
    for i, exp in enumerate(experiments_to_run):
        all_results.append(run_experiment(i, exp, WEIGHTS_PATH, model_name, dataset_name, run_timestamp, DATA_YAML, len(experiments_to_run)))

        elapsed = time.time() - overall0
        avg = elapsed / (i + 1)
        remaining = avg * (len(experiments_to_run) - (i + 1))
        print("-" * 90)
        print(f"⏱ Gesamt bisher: {format_time(elapsed)} | ⏳ Rest (≈): {format_time(remaining)}")
        print("-" * 90)

    # -----------------------
    # Summary CSV (1 Zeile pro Run) -> für Excel/Präsi
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

    print(f"\n📄 Summary CSV gespeichert: {summary_csv}")

    # -----------------------
    # Bestes Modell nach Val-F1 auswählen
    # -----------------------
    best_name = None
    best_f1 = float("-inf")
    best_pt = None

    print("\n" + "#" * 90)
    print("📊 SUMMARY (nur val() zählt)")
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
        print("\n🏆 BEST (nach Val-F1)")
        print(f"   Name: {best_name}")
        print(f"   F1  : {best_f1:.3f}")
        print(f"   Gespeichert als: {stable}")
    else:
        print("\n⚠️ Konnte kein bestes Modell bestimmen (F1 nicht verfügbar).")

    print(f"\n🎉 Fertig. Gesamtzeit: {format_time(time.time() - overall0)}")


if __name__ == "__main__":
    main()
