# Structure Status & Target (Punkt 2/3/4)

## Aktueller Stand
1. **Punkt 2 – Struktur/Nachvollziehbarkeit**
   - Code liegt aktuell im Repo-Root:
     - `train_models.py`
     - `compare_models.py`
     - `compare_models_demo_day.py`
   - Einstieg/Übersicht für neue Teammitglieder existiert primär über die Guides in `docs/`.
2. **Punkt 3 – Getrennte Struktur Local vs. SLURM**
   - Im Dateisystem gibt es derzeit keine getrennten Ordner für Local vs. SLURM.
   - Es gibt aber getrennte Doku:
     - `docs/startup-local.md`
     - `docs/startup-slurm.md`
3. **Punkt 4 – Readmes + Startup guides**
   - `README.md` und die Startup-Guides existieren bereits.
   - Nach einer eventuellen Struktur-Umstellung müssen Pfade/Beispielkommandos in diesen Dokumenten angepasst werden.

## Zielbild (minimal-invasiv)
- **Entry Points bleiben klar**:
  - Root-`README.md` als Einstieg
  - `docs/startup-local.md` und `docs/startup-slurm.md` als konkrete Startanleitungen
- **Ordnertrennung Local vs. SLURM im Dateisystem**:
  - `scripts/local/` enthält die lokalen Python-Entry-Points (Training/Compare)
  - `scripts/slurm/` enthält die SLURM Job-Skripte (`sbatch ...`)
- **Minimaler Umbau**:
  - Keine tiefen Refactors der Python-Logik.
  - Nur Moves + Pfad-Anpassungen in SLURM-Skripten und ggf. in Docs.

### Konkretes Minimal-Zielbild (empfohlen)
- `scripts/local/train_models.py` (Local Training)
- `scripts/local/compare_models.py` (Local Vergleich/Evaluation)
- `scripts/local/compare_models_demo_day.py` (Local Vergleich/Evaluation für Demo Day)
- `scripts/slurm/slurm_train_*.sh` (SLURM Training Jobs)
- `scripts/slurm/slurm_compare_*.sh` (SLURM Comparison Jobs)

