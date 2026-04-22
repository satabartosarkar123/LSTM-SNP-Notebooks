import subprocess
import plistlib
import sys
from collections import defaultdict

def record_per_process_power(duration_sec=10, interval_ms=1000):
    cmd = [
        "sudo", "powermetrics",
        "--samplers", "cpu_power,tasks",
        "-i", str(interval_ms),
        "-n", str(duration_sec),
        "-f", "plist"
    ]

    print(f"Recording for {duration_sec}s... (sudo may prompt for password)")
    print("-" * 75)

    result = subprocess.run(cmd, capture_output=True)

    if not result.stdout:
        print("ERROR: No output from powermetrics.")
        print(result.stderr.decode())
        sys.exit(1)

    raw_samples = result.stdout.split(b'\x00')

    process_power = defaultdict(float)
    process_names = {}
    sample_count  = 0
    total_cpu_mw  = 0.0
    total_gpu_mw  = 0.0
    total_ane_mw  = 0.0

    for raw in raw_samples:
        raw = raw.strip()
        if not raw:
            continue
        try:
            data = plistlib.loads(raw)
        except Exception:
            continue

        sample_count += 1

        proc         = data.get("processor", {})
        pkg_cpu_mw   = proc.get("cpu_power", 0.0)   # ← confirmed key
        pkg_gpu_mw   = proc.get("gpu_power", 0.0)
        pkg_ane_mw   = proc.get("ane_power", 0.0)
        total_cpu_mw += pkg_cpu_mw
        total_gpu_mw += pkg_gpu_mw
        total_ane_mw += pkg_ane_mw

        tasks = data.get("tasks", [])

        # ── use cputime_ms_per_s as CPU share proxy ──────────────────────
        total_cputime = sum(t.get("cputime_ms_per_s", 0.0) for t in tasks)
        if total_cputime == 0:
            continue

        for task in tasks:
            pid        = task.get("pid", -1)
            name       = task.get("name", "unknown")
            cpu_share  = task.get("cputime_ms_per_s", 0.0) / total_cputime
            est_power  = cpu_share * pkg_cpu_mw

            process_power[pid] += est_power
            process_names[pid]  = name

    if sample_count == 0:
        print("No samples parsed. Check sudo permissions.")
        return

    avg_cpu_mw = total_cpu_mw / sample_count
    avg_gpu_mw = total_gpu_mw / sample_count
    avg_ane_mw = total_ane_mw / sample_count

    print(f"\nSummary over {sample_count} sample(s):")
    print(f"  Avg CPU Power : {avg_cpu_mw:.1f} mW")
    print(f"  Avg GPU Power : {avg_gpu_mw:.1f} mW")
    print(f"  Avg ANE Power : {avg_ane_mw:.1f} mW")
    print(f"  Avg Combined  : {avg_cpu_mw + avg_gpu_mw + avg_ane_mw:.1f} mW\n")

    print(f"{'PID':<8} | {'Process Name':<30} | {'Avg CPU Power (mW)':<22} | {'CPU Share %'}")
    print("-" * 82)

    for pid, total_pw in sorted(process_power.items(), key=lambda x: x[1], reverse=True):
        avg_pw = total_pw / sample_count
        if avg_pw < 1.0:
            continue
        name      = process_names.get(pid, "unknown")[:30]
        share_pct = (avg_pw / avg_cpu_mw * 100) if avg_cpu_mw > 0 else 0
        print(f"{pid:<8} | {name:<30} | {avg_pw:<22.2f} | {share_pct:.1f}%")

    print("-" * 82)
    print("Done.")


if __name__ == "__main__":
    record_per_process_power(duration_sec=10, interval_ms=1000)