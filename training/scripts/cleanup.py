"""
Cleanup script - removes old test artifacts, temp files, and unused scripts.

Usage:
    python training/scripts/cleanup.py
"""
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
SCRIPTS_DIR = TRAINING_DIR / "scripts"


def safe_remove(path: Path):
    if path.exists():
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            print(f"  Removed dir:  {path}")
        else:
            path.unlink(missing_ok=True)
            print(f"  Removed file: {path}")


def main():
    print("=== Cleaning old test artifacts and temp files ===\n")

    # 1. Old test model artifacts
    print("[1] Old test models and training state...")
    safe_remove(TRAINING_DIR / "models" / "quick-sample")
    safe_remove(TRAINING_DIR / "train" / "quick-sample")
    safe_remove(TRAINING_DIR / "torchmodels_toexport" / "quick-sample-s999936-d2655")

    # 2. Old config files
    print("[2] Old config files...")
    safe_remove(TRAINING_DIR / "test-selfplay.cfg")

    # 3. Old PowerShell scripts (replaced by Python)
    print("[3] Old PowerShell scripts...")
    for ps1 in SCRIPTS_DIR.glob("*.ps1"):
        safe_remove(ps1)

    # 4. Old log files
    print("[4] Old log files...")
    for txt in SCRIPTS_DIR.glob("*.txt"):
        safe_remove(txt)
    safe_remove(TRAINING_DIR / "selfplay-stdout.txt")
    safe_remove(TRAINING_DIR / "selfplay-stderr.txt")

    # 5. Old counter files
    print("[5] Old counter files...")
    safe_remove(TRAINING_DIR / "iter_counter.txt")
    safe_remove(TRAINING_DIR / "run_counter.txt")

    # 6. Old shuffled data
    print("[6] Old shuffled data...")
    safe_remove(TRAINING_DIR / "shuffleddata")

    # 7. Old selfplay temp models
    print("[7] Old selfplay temp models...")
    safe_remove(TRAINING_DIR / "selfplay-models")

    # 8. Old selfplay data
    print("[8] Old selfplay data...")
    safe_remove(TRAINING_DIR / "data")

    # 9. Old temp files
    print("[9] Old temp files...")
    safe_remove(TRAINING_DIR / "tmp")

    # 10. Old export checkpoints
    print("[10] Old export checkpoints...")
    safe_remove(TRAINING_DIR / "torchmodels_toexport")

    # 11. Duplicate KataGomo-Amazons directory
    print("[11] Duplicate source directory...")
    safe_remove(PROJECT_ROOT / "KataGomo-Amazons")

    print("\n=== Cleanup complete! ===")
    print("\nRemaining structure:")
    print("  training/")
    print("    models/bootstrap-10x10/   <- bootstrap model")
    print("    train/iterative-run/      <- training checkpoint (KEPT)")
    print("    output/                   <- exported models (KEPT)")
    print("    scripts/")
    print("      train.py                <- main training script")
    print("      cleanup.py              <- this script")
    print("      bootstrap_model.py      <- model generator")
    print("      check_env.py            <- environment checker")
    print("    selfplay.cfg              <- selfplay config")
    print("  cpp/                        <- C++ source + katago.exe")
    print("  python/                     <- training library")
    print("  venv/                       <- Python environment")


if __name__ == "__main__":
    main()
