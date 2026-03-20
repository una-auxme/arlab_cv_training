# Beispiel-Struktur des `runs/` Ordners nach Training beider Modelle

## Annahme:
- **yolo11n-seg** Training gestartet am: `20260125-143000`
- **yolo26n-seg** Training gestartet am: `20260125-150000`
- Beide Skripte trainieren 4 Experimente: `no_augmentation`, `baseline`, `moderate_geom`, `strong_geom`

## Struktur (Ultralytics-Standard mit Modell-Unterscheidung):

```
runs/
├── yolo11n-seg/                                      # Projekt-Ordner (project="yolo11n-seg")
│   └── segment/                                      # Task-Ordner (automatisch für Segmentation)
│       ├── train/                                    # Training-Ordner
│       │   ├── 20260125_143000/                     # Timestamp-Ordner
│       │   │   └── no_augmentation/                 # Experiment 1
│       │   │       ├── weights/
│       │   │       │   ├── best.pt
│       │   │       │   ├── last.pt
│       │   │       │   └── epoch*.pt
│       │   │       ├── results.csv
│       │   │       ├── results.png                  # Metriken-Verlauf
│       │   │       ├── confusion_matrix.png
│       │   │       ├── F1_curve.png
│       │   │       ├── PR_curve.png
│       │   │       ├── train_batch0.jpg             # YOLO-Plots
│       │   │       ├── train_batch1.jpg
│       │   │       ├── train_batch2.jpg
│       │   │       ├── labels.jpg                   # Label-Verteilung
│       │   │       └── args.yaml
│       │   │
│       │   ├── 20260125_143000/
│       │   │   └── baseline/                        # Experiment 2
│       │   │       └── [gleiche Struktur wie oben]
│       │   │
│       │   ├── 20260125_143000/
│       │   │   └── moderate_geom/                    # Experiment 3
│       │   │       └── [gleiche Struktur wie oben]
│       │   │
│       │   └── 20260125_143000/
│       │       └── strong_geom/                     # Experiment 4
│       │           └── [gleiche Struktur wie oben]
│       │
│       └── val/                                      # Validation-Ordner
│           └── 20260125_143000/
│               ├── VAL_no_augmentation/               # Val für Exp 1
│               │   ├── results.csv                    # Wichtig: Metriken werden hieraus gelesen
│               │   ├── confusion_matrix.png
│               │   ├── F1_curve.png
│               │   ├── PR_curve.png
│               │   ├── val_batch0_labels.jpg
│               │   ├── val_batch0_pred.jpg
│               │   └── args.yaml
│               │
│               ├── VAL_baseline/                      # Val für Exp 2
│               │   └── [gleiche Struktur wie oben]
│               │
│               ├── VAL_moderate_geom/                 # Val für Exp 3
│               │   └── [gleiche Struktur wie oben]
│               │
│               └── VAL_strong_geom/                    # Val für Exp 4
│                   └── [gleiche Struktur wie oben]
│
└── yolo26n-seg/                                      # Projekt-Ordner (project="yolo26n-seg")
    └── segment/                                      # Task-Ordner (automatisch für Segmentation)
        ├── train/
        │   └── 20260125_150000/                      # Timestamp-Ordner
        │       ├── no_augmentation/
        │       ├── baseline/
        │       ├── moderate_geom/
        │       └── strong_geom/
        │
        └── val/
            └── 20260125_150000/
                ├── VAL_no_augmentation/
                ├── VAL_baseline/
                ├── VAL_moderate_geom/
                └── VAL_strong_geom/

```

## Zusammenfassung:

### Ordnerstruktur:
- **`runs/yolo11n-seg/`**: Alle Ergebnisse von **yolo11n-seg** Training
- **`runs/yolo26n-seg/`**: Alle Ergebnisse von **yolo26n-seg** Training

### Hierarchie:
1. **Modell-Ordner**: `yolo11n-seg/` oder `yolo26n-seg/`
2. **Timestamp-Ordner**: `20260125_143000/` (alle Experimente eines Runs teilen denselben Timestamp)
3. **Experiment-Ordner**: `no_augmentation/`, `baseline/`, `moderate_geom/`, `strong_geom/`
4. **Validation-Ordner**: `VAL_no_augmentation/`, `VAL_baseline/`, etc. (im gleichen Timestamp-Ordner)

### Beispiel-Pfad (Ultralytics-Standard):
- Training: `runs/yolo11n-seg/segment/train/20260125_143000/no_augmentation/`
- Validation: `runs/yolo11n-seg/segment/val/20260125_143000/VAL_no_augmentation/`

### Wichtige Dateien für Analyse:
1. **results.png** in jedem Experiment-Ordner → Metriken-Verlauf während Training
2. **results.csv** in jedem `VAL_*` Ordner → Finale Metriken (wird für Summary-CSV verwendet)
3. **confusion_matrix.png** → Confusion Matrix
4. **train_batch*.jpg** → Trainingsbilder mit Annotationen
5. **labels.jpg** → Label-Verteilung im Dataset

### Vorteile der Ultralytics-Standardstruktur:
- ✅ **Standardkonform**: Verwendet die native Ultralytics-Ordnerstruktur ohne Manipulation
- ✅ **Keine manuellen Verschiebungen**: Alles wird automatisch von Ultralytics erstellt
- ✅ **Eindeutige Modell-Identifikation**: Der oberste Ordner (`yolo11n-seg/` oder `yolo26n-seg/`) zeigt sofort, welches Modell verwendet wurde
- ✅ **Klare Trennung**: Training (`train/`) und Validation (`val/`) sind getrennt
- ✅ **Zeitliche Organisation**: Alle Experimente eines Runs teilen denselben Timestamp-Ordner
- ✅ **Kompatibel**: Funktioniert mit allen Ultralytics-Tools und -Features
- ✅ **Einfache Navigation**: Modell-Unterscheidung auf oberster Ebene, keine verschachtelten Projekt-Ordner
