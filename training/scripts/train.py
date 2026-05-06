"""
KataGomo Amazons iterative self-play training script.

Usage:
    python training/scripts/train.py                  # full run
    python training/scripts/train.py --skip-selfplay   # resume from shuffle
    python training/scripts/train.py --skip-selfplay --skip-shuffle  # resume from training

Each run: selfplay -> shuffle -> train -> export
Output: training/output/run-{N}/
"""
import argparse
import os
import sys
import json
import time
import shutil
import signal
import smtplib
import socket
import subprocess
import traceback
import threading
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path


# ============================================================
# Configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
TRAINING_DIR = PROJECT_ROOT / "training"
SCRIPTS_DIR = TRAINING_DIR / "scripts"
OUTPUT_DIR = TRAINING_DIR / "output"
MODELS_DIR = TRAINING_DIR / "models"
BOOTSTRAP_DIR = MODELS_DIR / "bootstrap-10x10"
RUN_COUNTER_FILE = TRAINING_DIR / "run_counter.txt"

KATAGO_EXE = PROJECT_ROOT / "cpp" / "build" / "Release" / "katago.exe"
VENV_PYTHON = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
PYTHON_DIR = PROJECT_ROOT / "python"
SELFPLAY_CONFIG = TRAINING_DIR / "selfplay.cfg"

SELFPLAY_MODELS_DIR = TRAINING_DIR / "selfplay-models"
SELFPLAY_DATA_DIR = TRAINING_DIR / "data" / "selfplay"
SHUFFLED_BASE_DIR = TRAINING_DIR / "shuffleddata"
TRAIN_DIR = TRAINING_DIR / "train" / "iterative-run"
EXPORT_DIR = TRAINING_DIR / "torchmodels_toexport"
TMP_DIR = TRAINING_DIR / "tmp"

SELFPLAY_HOURS = 20
TRAIN_EPOCHS = 80
BATCH_SIZE = 128

# ============================================================
# Email configuration
# ============================================================

EMAIL_CONFIG = {
    "host": "smtp.qq.com",
    "port": 465,
    "username": "2975194966@qq.com",
    "password": "aclnbepknekddejb",
    "to": "2975194966@qq.com",
}

# ============================================================
# Utility functions
# ============================================================


def log(msg: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)


def run(cmd: list[str], cwd=None, check=True) -> subprocess.CompletedProcess:
    log(f"  Running: {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if check and result.returncode != 0:
        raise RuntimeError(f"Command failed with code {result.returncode}: {cmd}")
    return result


def run_python(script: str | Path, args: list[str], check=True) -> subprocess.CompletedProcess:
    script_path = Path(script)
    if not script_path.is_absolute():
        script_path = PROJECT_ROOT / script_path
    return run([str(VENV_PYTHON), str(script_path)] + args, check=check)


def clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def clean_dir_contents(path: Path):
    if path.exists():
        for item in path.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
    path.mkdir(parents=True, exist_ok=True)


def send_email(subject: str, body: str):
    try:
        msg = MIMEMultipart()
        msg["From"] = EMAIL_CONFIG["username"]
        msg["To"] = EMAIL_CONFIG["to"]
        msg["Subject"] = subject
        msg.attach(MIMEText(body, "plain", "utf-8"))

        with smtplib.SMTP_SSL(EMAIL_CONFIG["host"], EMAIL_CONFIG["port"]) as server:
            server.login(EMAIL_CONFIG["username"], EMAIL_CONFIG["password"])
            server.sendmail(EMAIL_CONFIG["username"], EMAIL_CONFIG["to"], msg.as_string())
        log(f"  Email sent: {subject}")
    except Exception as e:
        log(f"  WARNING: Failed to send email: {e}")


def disable_sleep():
    try:
        subprocess.run(["powercfg", "/change", "standby-timeout-ac", "0"], capture_output=True)
        subprocess.run(["powercfg", "/change", "hibernate-timeout-ac", "0"], capture_output=True)
        log("  Sleep disabled.")
        return True
    except Exception:
        log("  WARNING: Could not disable sleep (run as admin).")
        return False


def restore_sleep():
    try:
        subprocess.run(["powercfg", "/change", "standby-timeout-ac", "30"], capture_output=True)
        subprocess.run(["powercfg", "/change", "hibernate-timeout-ac", "120"], capture_output=True)
        log("  Sleep settings restored.")
    except Exception:
        pass


def get_run_number() -> int:
    if RUN_COUNTER_FILE.exists():
        num = int(RUN_COUNTER_FILE.read_text().strip()) + 1
    else:
        num = 1
    RUN_COUNTER_FILE.write_text(str(num))
    return num


def get_run_dir(run_num: int) -> Path:
    return OUTPUT_DIR / f"run-{run_num:03d}"


def find_previous_checkpoint() -> Path | None:
    """Find the most recent checkpoint from a previous run."""
    if not OUTPUT_DIR.exists():
        return None
    run_dirs = sorted(
        [d for d in OUTPUT_DIR.iterdir() if d.is_dir() and d.name.startswith("run-")],
        key=lambda d: d.name,
        reverse=True,
    )
    for rd in run_dirs:
        ckpt = rd / "model.ckpt"
        if ckpt.exists():
            return ckpt
    return None


def cleanup_temp_files(skip_selfplay=False, skip_shuffle=False):
    """Clean all temporary and intermediate files."""
    log("Cleaning temporary files...")
    dirs_to_clean = [SELFPLAY_MODELS_DIR, TMP_DIR]
    if not skip_selfplay:
        dirs_to_clean.append(SELFPLAY_DATA_DIR)
    if not skip_shuffle:
        dirs_to_clean.append(SHUFFLED_BASE_DIR)
    for d in dirs_to_clean:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            log(f"  Removed: {d}")

    files_to_remove = [
        TRAINING_DIR / "selfplay-stdout.txt",
        TRAINING_DIR / "selfplay-stderr.txt",
    ]
    for f in files_to_remove:
        if f.exists():
            f.unlink(missing_ok=True)

    log("  Temp files cleaned.")


# ============================================================
# Training steps
# ============================================================


def step_prepare_model(run_num: int, run_dir: Path) -> tuple[Path, bool]:
    log("[Step 1] Preparing model for selfplay...")

    prev_ckpt = find_previous_checkpoint()
    if prev_ckpt is not None:
        log(f"  Using previous checkpoint: {prev_ckpt}")
        source_ckpt = prev_ckpt
        is_first_run = False
    else:
        log("  First run: generating bootstrap model...")
        if not (BOOTSTRAP_DIR / "model.ckpt").exists():
            run_python("training/scripts/bootstrap_model.py", ["b6c96", str(BOOTSTRAP_DIR)])
        if not (BOOTSTRAP_DIR / "model.bin").exists():
            run_python(
                "python/export_model_pytorch.py",
                [
                    "-checkpoint", str(BOOTSTRAP_DIR / "model.ckpt"),
                    "-export-dir", str(BOOTSTRAP_DIR),
                    "-model-name", "amazons-bootstrap-10x10",
                    "-filename-prefix", "model",
                ],
            )
        source_ckpt = BOOTSTRAP_DIR / "model.ckpt"
        is_first_run = True

    clean_dir(SELFPLAY_MODELS_DIR)
    run_python(
        "python/export_model_pytorch.py",
        [
            "-checkpoint", str(source_ckpt),
            "-export-dir", str(SELFPLAY_MODELS_DIR),
            "-model-name", "selfplay-model",
            "-filename-prefix", "model",
        ],
    )
    log("  Model ready for selfplay.")
    return source_ckpt, is_first_run


def step_selfplay(run_num: int):
    log(f"[Step 2] Starting selfplay ({SELFPLAY_HOURS} hours)...")

    clean_dir_contents(SELFPLAY_DATA_DIR)

    log_file = TRAINING_DIR / "selfplay-stdout.txt"
    err_file = TRAINING_DIR / "selfplay-stderr.txt"

    proc = subprocess.Popen(
        [
            str(KATAGO_EXE),
            "selfplay",
            "-config", str(SELFPLAY_CONFIG),
            "-models-dir", str(SELFPLAY_MODELS_DIR),
            "-output-dir", str(SELFPLAY_DATA_DIR),
        ],
        cwd=str(KATAGO_EXE.parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )

    def _filter_stdout():
        with open(log_file, "w") as out_f:
            for line in proc.stdout:
                if "Not finished game" not in line:
                    out_f.write(line)

    def _filter_stderr():
        with open(err_file, "w") as err_f:
            for line in proc.stderr:
                err_f.write(line)

    t_out = threading.Thread(target=_filter_stdout, daemon=True)
    t_err = threading.Thread(target=_filter_stderr, daemon=True)
    t_out.start()
    t_err.start()

    timeout = SELFPLAY_HOURS * 3600
    elapsed = 0
    try:
        while elapsed < timeout and proc.poll() is None:
            time.sleep(60)
            elapsed += 60
            pct = round(elapsed / timeout * 100)
            file_count = sum(1 for _ in SELFPLAY_DATA_DIR.rglob("*")) if SELFPLAY_DATA_DIR.exists() else 0
            log(f"  Selfplay: {elapsed / 3600:.1f}h / {SELFPLAY_HOURS}h ({pct}%) | files: {file_count}")

        if proc.poll() is None:
            log("  Time reached, stopping selfplay...")
            proc.terminate()
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
    except KeyboardInterrupt:
        log("  Interrupted, stopping selfplay...")
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise

    t_out.join(timeout=5)
    t_err.join(timeout=5)
    log("  Selfplay done.")


def step_shuffle(run_num: int):
    log("[Step 3] Shuffling data...")

    shuffle_dir = SHUFFLED_BASE_DIR / f"run-{run_num:03d}"
    clean_dir(shuffle_dir / "train")
    clean_dir(shuffle_dir / "val")

    common_args = [
        str(SELFPLAY_DATA_DIR),
        "-min-rows", "10000",
        "-keep-target-rows", "1000000",
        "-expand-window-per-row", "4000",
        "-taper-window-exponent", "0.3",
        "-approx-rows-per-out-file", "50000",
        "-num-processes", "6",
        "-batch-size", "128",
        "-output-npz",
    ]

    run_python("python/shuffle.py", common_args + [
        "-out-dir", str(shuffle_dir / "train"),
        "-out-tmp-dir", str(TMP_DIR / "train"),
        "-only-include-md5-path-prop-lbound", "0.00",
        "-only-include-md5-path-prop-ubound", "0.95",
    ])

    run_python("python/shuffle.py", common_args + [
        "-out-dir", str(shuffle_dir / "val"),
        "-out-tmp-dir", str(TMP_DIR / "val"),
        "-only-include-md5-path-prop-lbound", "0.95",
        "-only-include-md5-path-prop-ubound", "1.00",
    ])

    train_json = shuffle_dir / "train.json"
    val_json = shuffle_dir / "val.json"
    if train_json.exists():
        shutil.copy(train_json, shuffle_dir / "train" / "train.json")
    if val_json.exists():
        shutil.copy(val_json, shuffle_dir / "val" / "train.json")

    val_train_dir = shuffle_dir / "val" / "train"
    val_train_dir.mkdir(parents=True, exist_ok=True)
    for item in (shuffle_dir / "val").iterdir():
        if item.is_file() and item.suffix in (".npz", ".json"):
            shutil.move(str(item), str(val_train_dir / item.name))

    log("  Shuffle done.")


def step_train(run_num: int, is_first_run: bool, shuffle_dir_override: Path | None = None):
    log(f"[Step 4] Training ({TRAIN_EPOCHS} epochs, batch={BATCH_SIZE})...")

    shuffle_dir = shuffle_dir_override or (SHUFFLED_BASE_DIR / f"run-{run_num:03d}")
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    args = [
        str(PYTHON_DIR / "train.py"),
        "-traindir", str(TRAIN_DIR),
        "-latestdatadir", str(shuffle_dir),
        "-exportdir", str(EXPORT_DIR),
        "-exportprefix", "iterative-run",
        "-pos-len", "10",
        "-batch-size", str(BATCH_SIZE),
        "-samples-per-epoch", "100000",
        "-max-epochs-this-instance", str(TRAIN_EPOCHS),
        "-no-compile",
        "-no-export",
    ]
    if is_first_run:
        args += ["-model-kind", "b6c96"]

    run([str(VENV_PYTHON)] + args)
    log("  Training done.")


def step_export(run_num: int, run_dir: Path):
    log("[Step 5] Exporting checkpoint...")

    ckpt_src = TRAIN_DIR / "checkpoint.ckpt"
    if not ckpt_src.exists():
        raise RuntimeError(f"Checkpoint not found: {ckpt_src}")

    ckpt_dst = run_dir / "model.ckpt"
    shutil.copy2(ckpt_src, ckpt_dst)
    log(f"  Checkpoint: {ckpt_dst}")


# ============================================================
# Main
# ============================================================


def main():
    parser = argparse.ArgumentParser(description="KataGomo Amazons training script")
    parser.add_argument("--skip-selfplay", action="store_true", help="Skip selfplay (use existing data)")
    parser.add_argument("--skip-shuffle", action="store_true", help="Skip shuffle (use existing shuffled data)")
    parser.add_argument("--shuffle-dir", type=str, default=None, help="Path to existing shuffle dir (for --skip-shuffle)")
    args = parser.parse_args()

    shuffle_dir_override = Path(args.shuffle_dir) if args.shuffle_dir else None

    run_num = get_run_number()
    run_dir = get_run_dir(run_num)
    run_dir.mkdir(parents=True, exist_ok=True)

    hostname = socket.gethostname()
    start_time = datetime.now()

    log("=" * 60)
    log(f"  RUN {run_num:03d}")
    log(f"  Selfplay : {'SKIP' if args.skip_selfplay else f'{SELFPLAY_HOURS} hours'}")
    log(f"  Shuffle  : {'SKIP' if args.skip_shuffle else 'yes'}")
    log(f"  Training : {TRAIN_EPOCHS} epochs")
    log(f"  Batch    : {BATCH_SIZE}")
    log(f"  Output   : {run_dir}")
    log("=" * 60)

    send_email(
        f"[KataGomo Run {run_num:03d}] Training started",
        f"Run {run_num:03d} started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Host: {hostname}\n"
        f"Selfplay: {'SKIP' if args.skip_selfplay else f'{SELFPLAY_HOURS}h'} | "
        f"Shuffle: {'SKIP' if args.skip_shuffle else 'yes'} | "
        f"Epochs: {TRAIN_EPOCHS} | Batch: {BATCH_SIZE}\n"
        f"Output: {run_dir}",
    )

    sleep_disabled = disable_sleep()
    success = False

    try:
        cleanup_temp_files(args.skip_selfplay, args.skip_shuffle)

        source_ckpt, is_first_run = step_prepare_model(run_num, run_dir)

        if args.skip_selfplay:
            log("[Step 2] Selfplay SKIPPED (--skip-selfplay)")
        else:
            step_selfplay(run_num)
            send_email(
                f"[KataGomo Run {run_num:03d}] Selfplay completed",
                f"Run {run_num:03d} selfplay finished.\n"
                f"Host: {hostname}\n"
                f"Starting shuffle and training...",
            )

        if args.skip_shuffle:
            log("[Step 3] Shuffle SKIPPED (--skip-shuffle)")
        else:
            step_shuffle(run_num)

        step_train(run_num, is_first_run, shuffle_dir_override)
        step_export(run_num, run_dir)

        duration = datetime.now() - start_time
        hours, remainder = divmod(int(duration.total_seconds()), 3600)
        minutes = remainder // 60

        log("=" * 60)
        log(f"  RUN {run_num:03d} COMPLETE!")
        log(f"  Duration : {hours}h {minutes}m")
        log(f"  Model    : {run_dir / 'model.bin'}")
        log(f"  Next run : python training/scripts/train.py")
        log("=" * 60)

        send_email(
            f"[KataGomo Run {run_num:03d}] Training complete",
            f"Run {run_num:03d} completed successfully!\n"
            f"Host: {hostname}\n"
            f"Duration: {hours}h {minutes}m\n"
            f"Model: {run_dir / 'model.bin'}\n"
            f"\n"
            f"Run again: python training/scripts/train.py",
        )

        success = True

    except Exception as e:
        duration = datetime.now() - start_time
        error_msg = f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"

        log(f"  ERROR: {e}")
        traceback.print_exc()

        send_email(
            f"[KataGomo Run {run_num:03d}] FAILED",
            f"Run {run_num:03d} failed after {duration}.\n"
            f"Host: {hostname}\n"
            f"\n"
            f"{error_msg}",
        )

    finally:
        cleanup_temp_files(args.skip_selfplay, args.skip_shuffle)
        if sleep_disabled:
            restore_sleep()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
