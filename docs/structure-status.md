# Structure Status & Target (points 2/3/4)

## Current status
1. **Point 2 – structure / traceability**
   - Code currently lives in the repo root:
     - `train_models.py`
     - `compare_models.py`
     - `compare_models_demo_day.py`
   - Onboarding / overview for new team members is mainly provided through the guides in `docs/`.
2. **Point 3 – separated Local vs. SLURM structure**
   - In the filesystem, there are currently no separate folders for Local vs. SLURM.
   - Documentation is separated instead:
     - `docs/startup-local.md`
     - `docs/startup-slurm.md`
3. **Point 4 – READMEs + startup guides**
   - `README.md` and the startup guides already exist.
   - After any structure change, paths / example commands in these documents may need to be updated.

## Target state (minimal-invasiveness)
- **Entry points stay clear**
  - Root `README.md` as the entry point
  - `docs/startup-local.md` and `docs/startup-slurm.md` as concrete starting guides
- **Local vs. SLURM separation in the filesystem**
  - `scripts/local/` contains the local Python entry points (training / comparison)
  - `scripts/slurm/` contains the SLURM job scripts (`sbatch ...`)
- **Minimal refactoring**
  - No deep refactors of the Python logic
  - Only moves + path updates in SLURM scripts and, if needed, in docs

### Concrete minimal target (recommended)
- `scripts/local/train_models.py` (Local training)
- `scripts/local/compare_models.py` (Local comparison/evaluation)
- `scripts/local/compare_models_demo_day.py` (Local comparison/evaluation for Demo Day)
- `scripts/slurm/slurm_train_*.sh` (SLURM training jobs)
- `scripts/slurm/slurm_compare_*.sh` (SLURM comparison jobs)

