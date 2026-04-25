#!/usr/bin/env python3
import subprocess
import plistlib
import sys
import argparse
from collections import defaultdict


def is_relevant_process(name):
    name_lower = name.lower()
    if 'google' in name_lower or 'chrome' in name_lower:
        return False
    keywords = ['python', 'code', 'jupyter', 'antigravity',
                'language_server', 'pyrefly', 'electron']
    return any(k in name_lower for k in keywords)


def record_power(duration_sec=60, interval_ms=1000):
    n_samples = duration_sec * 1000 // interval_ms

    cmd = [
        "sudo", "powermetrics",
        "--samplers", "cpu_power,tasks",
        "-i", str(interval_ms),
        "-n", str(n_samples),
        "-f", "plist",
    ]

    print(f"╔══════════════════════════════════════════════════════════════╗")
    print(f"║  Power Measurement — Apple Silicon                         ║")
    print(f"║  Duration : {duration_sec:>4}s ({duration_sec/60:.1f} min)                              ║")
    print(f"║  Interval : {interval_ms:>4}ms (1 reading/sec)                        ║")
    print(f"║  Samples  : {n_samples:>4}                                           ║")
    print(f"╚══════════════════════════════════════════════════════════════╝")
    print(f"\nRecording... (sudo may prompt for password)\n")

    result = subprocess.run(cmd, capture_output=True)

    if not result.stdout:
        print("ERROR: No output from powermetrics.")
        print(result.stderr.decode())
        sys.exit(1)

    raw_samples = result.stdout.split(b'\x00')

    proc_cpu_power     = defaultdict(list)
    proc_names         = {}
    sys_cpu_mw_samples = []
    sample_count       = 0

    for raw in raw_samples:
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = plistlib.loads(raw)
        except Exception:
            continue

        sample_count += 1

        processor  = data.get("processor", {})
        pkg_cpu_mw = processor.get("cpu_power", 0.0)
        sys_cpu_mw_samples.append(pkg_cpu_mw)

        tasks         = data.get("tasks", [])
        total_cputime = sum(t.get("cputime_ms_per_s", 0.0) for t in tasks)

        for task in tasks:
            pid  = task.get("pid", -1)
            name = task.get("name", "unknown")
            proc_names[pid] = name

            cpu_ms    = task.get("cputime_ms_per_s", 0.0)
            cpu_share = (cpu_ms / total_cputime) if total_cputime > 0 else 0.0
            proc_cpu_power[pid].append(cpu_share * pkg_cpu_mw)

        if sample_count % 10 == 0:
            print(f"  {sample_count}/{n_samples} samples collected...", flush=True)

    if sample_count == 0:
        print("No samples parsed. Check sudo permissions.")
        return

    dt_sec = interval_ms / 1000.0

    # ── System-level ──────────────────────────────────────────────────────
    nonzero_sys    = [p for p in sys_cpu_mw_samples if p > 0]
    n_nonzero_sys  = len(nonzero_sys) if nonzero_sys else 1
    avg_sys_cpu_mw = sum(nonzero_sys) / n_nonzero_sys

    # Energy = Σ P(tᵢ) × Δt  over ALL samples (zeros count as zero contribution)
    energy_cpu_mj       = sum(p * dt_sec for p in sys_cpu_mw_samples)
    energy_cpu_j        = energy_cpu_mj / 1000.0
    measured_minutes    = (sample_count * dt_sec) / 60.0

    # Per-sec: average power over non-zero samples × Δt
    # = (Σ P(tᵢ) for non-zero tᵢ) / n_nonzero × Δt
    sys_avg_e_per_sec   = avg_sys_cpu_mw * dt_sec   # mJ/s
    sys_avg_e_per_min   = sys_avg_e_per_sec * 60     # mJ/min
    sys_avg_e_per_30min = sys_avg_e_per_min * 30     # mJ/30min

    print(f"\n{'═' * 75}")
    print(f"  SYSTEM-LEVEL SUMMARY  ({sample_count} samples over {measured_minutes:.2f} min)")
    print(f"{'═' * 75}")
    print(f"  Average CPU Power     : {avg_sys_cpu_mw:>10.1f} mW  ({n_nonzero_sys} non-zero samples)")
    print(f"  Total Energy (CPU)    : {energy_cpu_mj:>10.1f} mJ  ({energy_cpu_j:.3f} J)")
    print(f"  Avg Energy / sec      : {sys_avg_e_per_sec:>10.3f} mJ/s   [= avg_power × Δt]")
    print(f"  Avg Energy / min      : {sys_avg_e_per_min:>10.3f} mJ/min [= /sec × 60]")
    print(f"  Avg Energy / 30 min   : {sys_avg_e_per_30min:>10.3f} mJ    [= /min × 30]")
    print(f"{'═' * 75}")

    # ── Per-process breakdown ─────────────────────────────────────────────
    pid_results = []

    for pid in proc_cpu_power:
        cpu_samples = proc_cpu_power[pid]

        # Non-zero samples only for average power
        nonzero = [p for p in cpu_samples if p > 0]
        n_nz    = len(nonzero) if nonzero else 1
        avg_cpu = sum(nonzero) / n_nz                    # avg power over active samples (mW)

        # Total energy over ALL samples (zeros = process was idle, contributes 0)
        e_cpu   = sum(p * dt_sec for p in cpu_samples)   # mJ

        # Per-sec  = avg power over non-zero samples × Δt
        # This is consistent: avg_power (mW) × Δt (s) = mJ per active second
        avg_e_per_sec  = avg_cpu * dt_sec                # mJ/s
        avg_e_per_min  = avg_e_per_sec * 60              # mJ/min  [= /sec × 60]
        avg_e_per_30min = avg_e_per_min * 30             # mJ/30min [= /min × 30]

        if avg_cpu < 0.1:
            continue

        pid_results.append({
            'pid':              pid,
            'name':             proc_names.get(pid, 'unknown')[:25],
            'avg_cpu_mw':       avg_cpu,
            'energy_cpu_mj':    e_cpu,
            'n_nonzero':        n_nz,
            'cpu_share_pct':    (avg_cpu / avg_sys_cpu_mw * 100) if avg_sys_cpu_mw > 0 else 0,
            'avg_e_per_sec':    avg_e_per_sec,
            'avg_e_per_min':    avg_e_per_min,
            'avg_e_per_30min':  avg_e_per_30min,
        })

    pid_results = [r for r in pid_results if is_relevant_process(r['name'])]
    pid_results.sort(key=lambda x: x['energy_cpu_mj'], reverse=True)
    pid_results = pid_results[:20]
    pid_results.sort(key=lambda x: x['pid'], reverse=True)

    print(f"\n{'═' * 125}")
    print(f"  PER-PROCESS BREAKDOWN (Top 20 Coding Processes, Sorted by PID)")
    print(f"{'═' * 125}")
    print(f"{'PID':<8} | {'Process':<25} | {'Avg CPU':>9} | {'E(CPU)':>10} | {'Smpls':>5} | {'CPU%':>5} | {'E/sec':>10} | {'E/min':>12} | {'E/30min':>13}")
    print(f"{'':8} | {'':25} | {'(mW)':>9} | {'(mJ)':>10} | {'non-0':>5} | {'':>5} | {'(mJ/s)':>10} | {'(mJ/min)':>12} | {'(mJ)':>13}")
    print(f"{'─' * 125}")

    for r in pid_results:
        print(
            f"{r['pid']:<8} | {r['name']:<25} | "
            f"{r['avg_cpu_mw']:>9.1f} | "
            f"{r['energy_cpu_mj']:>10.1f} | "
            f"{r['n_nonzero']:>5} | "
            f"{r['cpu_share_pct']:>4.1f}% | "
            f"{r['avg_e_per_sec']:>10.4f} | "
            f"{r['avg_e_per_min']:>12.4f} | "
            f"{r['avg_e_per_30min']:>13.4f}"
        )

    print(f"{'─' * 125}")
    print(f"\n  Derivation chain:")
    print(f"    avg_power (mW)  = Σ P(tᵢ) / n_nonzero")
    print(f"    E/sec  (mJ/s)   = avg_power × Δt           (Δt = {dt_sec}s)")
    print(f"    E/min  (mJ/min) = E/sec × 60")
    print(f"    E/30min  (mJ)   = E/min × 30")
    print(f"\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apple Silicon Power Measurement")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--interval", type=int, default=1000)
    args = parser.parse_args()
    record_power(duration_sec=args.duration, interval_ms=args.interval)