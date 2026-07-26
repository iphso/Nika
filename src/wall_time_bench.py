"""
External runner: launches nika.py training and reports PSNR at fixed wall-clock
intervals (e.g. every 5 minutes) instead of by epoch count, since epoch wall-time
varies with batch size, OOM retries, and GPU contention.

Usage:
    python wall_time_bench.py --name bunny --config xxs4 --interval_min 5

Writes:
    - a raw passthrough log of the training stdout
    - a CSV with one row per interval: elapsed time, epoch reached, current PSNR,
      best PSNR seen so far
"""
import argparse
import csv
import glob
import os
import re
import shutil
import subprocess
import sys
import time

def gpu_index_from_device(device):
    m = re.search(r"(\d+)", device)
    return int(m.group(1)) if m else 0


def query_gpu_mem_mib(gpu_index):
    try:
        out = subprocess.check_output(
            ["nvidia-smi", f"--id={gpu_index}", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True, timeout=5,
        )
        return int(out.strip().splitlines()[0])
    except Exception:
        return None


EPOCH_RE = re.compile(
    r"^\[(?P<name>[^\]]+)\] Epoch (?P<epoch>\d+) loss: (?P<loss>-?[\d.]+), "
    r"time: (?P<ftime>[\d.]+)s, PSNR: (?P<psnr>-?[\d.]+)$"
)
BEST_RE = re.compile(
    r"^\[(?P<name>[^\]]+)\] New best model saved at epoch (?P<epoch>\d+) "
    r"with PSNR: (?P<psnr>-?[\d.]+)$"
)


def main():
    parser = argparse.ArgumentParser(description="Wall-clock PSNR reporter for nika.py")
    parser.add_argument("--interval_min", type=float, default=5.0,
                        help="Report cadence in minutes (default: every 5 min)")
    parser.add_argument("--max_min", type=float, default=None,
                        help="Terminate training after this many wall-clock minutes (default: run to completion)")
    parser.add_argument("--out", default=None, help="CSV output path")
    parser.add_argument("--log", default=None, help="Raw stdout log path")

    # pass-through args for nika.py
    parser.add_argument("--basedir", default="static/benchmarks")
    parser.add_argument("--name", default="bunny")
    parser.add_argument("--config", default="small")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--segments", type=int, default=None)
    parser.add_argument("--operator_steps", type=int, default=None)
    parser.add_argument("--efficient_mode", action="store_true")
    args = parser.parse_args()

    src_dir = os.path.dirname(os.path.abspath(__file__))
    ts = time.strftime("%Y%m%d-%H%M%S")
    safe_name = args.name.replace("/", "_")
    out_path = args.out or os.path.join(src_dir, f"walltime_{safe_name}_{args.config}_{ts}.csv")
    log_path = args.log or os.path.join(src_dir, f"walltime_{safe_name}_{args.config}_{ts}.log")

    cmd = [
        sys.executable, "-u", "nika.py",
        "--basedir", args.basedir,
        "--name", args.name,
        "--config", args.config,
        "--device", args.device,
        "--batch_size", str(args.batch_size),
    ]
    if args.segments is not None:
        cmd += ["--segments", str(args.segments)]
    if args.operator_steps is not None:
        cmd += ["--operator_steps", str(args.operator_steps)]
    if args.efficient_mode:
        cmd += ["--efficient_mode"]

    gpu_index = gpu_index_from_device(args.device)

    print(f"Launching: {' '.join(cmd)} (cwd={src_dir})")
    print(f"Raw log:        {log_path}")
    print(f"Wall-time CSV:  {out_path}")
    print(f"Reporting every {args.interval_min} min" + (f", max {args.max_min} min" if args.max_min else "") + "\n")

    interval_sec = args.interval_min * 60.0
    start = time.time()
    next_boundary = interval_sec
    latest_epoch = None
    latest_psnr = None
    best_epoch = None
    best_psnr = None
    stopped_early = False

    proc = subprocess.Popen(
        cmd, cwd=src_dir, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )

    with open(log_path, "w") as logf, open(out_path, "w", newline="") as outf:
        writer = csv.writer(outf)
        writer.writerow(["elapsed_seconds", "elapsed_minutes", "epoch", "psnr", "best_psnr", "best_epoch", "gpu_mem_used_mib"])
        outf.flush()

        try:
            for line in proc.stdout:
                now = time.time()
                elapsed = now - start
                logf.write(line)
                logf.flush()
                sys.stdout.write(line)

                stripped = line.strip()
                m = EPOCH_RE.match(stripped)
                if m:
                    latest_epoch = int(m.group("epoch"))
                    latest_psnr = float(m.group("psnr"))
                else:
                    bm = BEST_RE.match(stripped)
                    if bm:
                        best_epoch = int(bm.group("epoch"))
                        best_psnr = float(bm.group("psnr"))

                while elapsed >= next_boundary:
                    minutes = next_boundary / 60.0
                    mem_mib = query_gpu_mem_mib(gpu_index)
                    writer.writerow([f"{elapsed:.1f}", f"{minutes:.2f}", latest_epoch, latest_psnr, best_psnr, best_epoch, mem_mib])
                    outf.flush()

                    # Snapshot whichever "new best" checkpoint is newest right now under a
                    # canonical <mm>min name, so this wall-clock mark has a fixed filename to
                    # eval standalone later instead of having to reconstruct nearest-epoch-<=-mark
                    # logic from the (already free, training-reconstruction-derived) best-checkpoint files.
                    model_name = os.path.basename(args.name)
                    save_config = f"{args.config}-eff" if args.efficient_mode else args.config
                    candidates = glob.glob(os.path.join(src_dir, "models", f"{save_config}-{model_name}-epoch*-psnr*.torch"))
                    snapshot_path = os.path.join(src_dir, "models", f"{save_config}-{model_name}-{minutes:.0f}min.torch")
                    if candidates and not os.path.isfile(snapshot_path):
                        try:
                            newest = max(candidates, key=os.path.getmtime)
                            shutil.copy2(newest, snapshot_path)
                        except OSError as e:
                            # A failed copy must never crash this monitor loop (see HiNeRV's
                            # wall_time_bench.py for the same fix -- an uncaught exception here
                            # skips the max_min check below and orphans the training subprocess).
                            print(f"    (snapshot copy failed, will be missing for this mark: {e!r})")

                    print(f">>> [{minutes:.0f} min] epoch={latest_epoch} psnr={latest_psnr} "
                          f"(best={best_psnr} @ epoch {best_epoch}) gpu_mem={mem_mib}MiB "
                          f"snapshot={'yes' if os.path.isfile(snapshot_path) else 'no'}")
                    next_boundary += interval_sec

                if args.max_min is not None and elapsed >= args.max_min * 60.0:
                    stopped_early = True
                    proc.terminate()
                    break
        except BaseException as e:
            stopped_early = True
            if not isinstance(e, KeyboardInterrupt):
                import traceback
                traceback.print_exc()
            proc.terminate()
        finally:
            proc.wait()
            total_elapsed = time.time() - start
            mem_mib = query_gpu_mem_mib(gpu_index)
            writer.writerow([f"{total_elapsed:.1f}", f"{total_elapsed/60:.2f}", latest_epoch, latest_psnr, best_psnr, best_epoch, mem_mib])
            outf.flush()
            tag = "STOPPED (max_min reached)" if stopped_early else "Done"
            print(f"\n{tag}. Total elapsed: {total_elapsed/60:.2f} min. Final: "
                  f"epoch={latest_epoch} psnr={latest_psnr} best={best_psnr} @ epoch {best_epoch} gpu_mem={mem_mib}MiB")


if __name__ == "__main__":
    main()
