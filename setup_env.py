#!/usr/bin/env python3
"""
Setup script for LSTM-SNP Notebooks.

This script creates a Python virtual environment, installs all required
dependencies, and registers a Jupyter kernel so the notebooks can be
run on any device.

Usage:
    python setup_env.py

After running, select the "LSTM-SNP (Python 3.11)" kernel in Jupyter
to run the notebooks.
"""

import os
import platform
import subprocess
import sys

# ── Configuration ──────────────────────────────────────────────────
VENV_DIR = "venv"
KERNEL_NAME = "snp-venv"
KERNEL_DISPLAY = "LSTM-SNP (Python 3.11)"
REQUIREMENTS = "requirements.txt"
# ──────────────────────────────────────────────────────────────────

def main():
    # Work from the directory where this script lives
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    venv_path = os.path.join(script_dir, VENV_DIR)
    is_windows = platform.system() == "Windows"

    # Paths inside the venv
    if is_windows:
        python_bin = os.path.join(venv_path, "Scripts", "python.exe")
        pip_bin = os.path.join(venv_path, "Scripts", "pip.exe")
    else:
        python_bin = os.path.join(venv_path, "bin", "python")
        pip_bin = os.path.join(venv_path, "bin", "pip")

    # ── 1. Create virtual environment ─────────────────────────────
    if os.path.isdir(venv_path):
        print(f"[✓] Virtual environment already exists at '{VENV_DIR}/'")
    else:
        print(f"[*] Creating virtual environment in '{VENV_DIR}/'...")
        subprocess.check_call([sys.executable, "-m", "venv", venv_path])
        print(f"[✓] Virtual environment created.")

    # ── 2. Upgrade pip ────────────────────────────────────────────
    print("[*] Upgrading pip...")
    subprocess.check_call(
        [python_bin, "-m", "pip", "install", "--upgrade", "pip"],
        stdout=subprocess.DEVNULL,
    )
    print("[✓] pip upgraded.")

    # ── 3. Install dependencies ───────────────────────────────────
    req_path = os.path.join(script_dir, REQUIREMENTS)
    if not os.path.isfile(req_path):
        print(f"[!] {REQUIREMENTS} not found – skipping dependency install.")
    else:
        print(f"[*] Installing dependencies from {REQUIREMENTS}...")
        subprocess.check_call([pip_bin, "install", "-r", req_path])
        print("[✓] All dependencies installed.")

    # ── 4. Register Jupyter kernel ────────────────────────────────
    print(f"[*] Registering Jupyter kernel '{KERNEL_DISPLAY}'...")
    subprocess.check_call([
        python_bin, "-m", "ipykernel", "install",
        "--user",
        "--name", KERNEL_NAME,
        "--display-name", KERNEL_DISPLAY,
    ])
    print(f"[✓] Kernel '{KERNEL_DISPLAY}' registered.")

    # ── Done ──────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print("  Setup complete!")
    print(f"  Kernel: {KERNEL_DISPLAY}")
    print("  Open the notebooks in Jupyter / VS Code and select")
    print(f"  the '{KERNEL_DISPLAY}' kernel to run them.")
    print("=" * 60)


if __name__ == "__main__":
    main()
