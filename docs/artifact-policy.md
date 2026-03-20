# Artifact Policy (What gets versioned)

Diese Regeln sind die Grundlage für ein „aufgeraeumtes“ neues Trainingspipeline-Repo.
Ziel: Code und entscheidungsrelevante Outputs nachvollziehbar versionieren, aber Run-/System-Artefakte nicht ungefiltert mitschleppen.

## Versioniert (git tracked)
- Quellcode und Training-/Evaluation-Skripte (z.B. `train_models.py`, `compare_models*.py`)
- `requirements.txt`
- Ultralytics Run-Checkpoints (gezielt):
  - `runs/**/weights/*.pt` (z.B. `best.pt`, `last.pt`)
- Kleine, reviewbare Run-Metadaten:
  - `runs/**/results.csv`
  - `runs/**/args.yaml`
- Optional vorhandene Struktur-/Dokumentationsdateien unter `runs/`:
  - `runs/STRUKTUR_BEISPIEL.md`
- Vergleichsergebnisse/Dokumentation:
  - `evaluation/**/*.md` (falls vorhanden)

## Nicht versioniert (git ignored)
- Virtuelle Umgebung:
  - `.venv/`
- Log-/Job-Output:
  - `logs/`
  - `slurm-*.out`, `slurm-*.err`
- Caches:
  - `*.cache`, `**/*.cache`
- Run-Artefakte außer den explizit „versionierten“ Selektoren (z.B. Plots/Images innerhalb `runs/`)
- Exporte/Exports und Baseline-Weights:
  - `yolo_weights/`
  - `weights/`
- Datasets:
  - `data_*/`, `fruit_dataset_*/`, `fruits_testset/`
  - Begründung: Datasets sollen außerhalb des Repos verwaltet werden (oder über Setup-Schritte bereitgestellt werden), um Repo-Größe zu kontrollieren.

## Warum diese Auswahl?
- `runs/**/weights/*.pt` ist für spätere Evaluations-/Vergleichsschritte der wichtigste „Beweis“.
- `results.csv` und `args.yaml` reichen oft, um die wichtigsten Hyperparameter und Metrikverläufe nachvollziehen zu können.
- Große/zeitraubende Artefakte wie Plots/Images und Job-Logs werden ignoriert, damit das Repo schlank bleibt.

