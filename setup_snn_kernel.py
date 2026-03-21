#!/usr/bin/env python3
"""
Setup script to create a Jupyter kernel for the SNN-Transformer notebooks.

This script will:
1. Create a Python virtual environment named 'snn_venv'.
2. Install all required dependencies (torch, pandas, numpy, scikit-learn, matplotlib, ipykernel).
3. Register the virtual environment as a Jupyter kernel named 'SNN Transformer'.

Usage:
    python3 setup_snn_kernel.py
"""

import os
import sys
import subprocess
import platform

VENV_DIR = "snn_venv"
KERNEL_NAME = "snn_transformer"
DISPLAY_NAME = "SNN Transformer"

def run_cmd(cmd, desc):
    print(f"\\n---> {desc}")
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"\\n[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    print("============================================================")
    print(" SNN-Transformer Kernel Setup Script")
    print("============================================================")

    # 1. Create Virtual Environment
    if not os.path.exists(VENV_DIR):
        run_cmd([sys.executable, "-m", "venv", VENV_DIR], "Creating virtual environment...")
    else:
        print(f"\\n---> Virtual environment '{VENV_DIR}' already exists. Skipping creation.")

    # Determine paths based on OS
    if platform.system() == "Windows":
        pip_exe = os.path.join(VENV_DIR, "Scripts", "pip")
        python_exe = os.path.join(VENV_DIR, "Scripts", "python")
    else:
        pip_exe = os.path.join(VENV_DIR, "bin", "pip")
        python_exe = os.path.join(VENV_DIR, "bin", "python")

    # Double check if pip exists
    if not os.path.exists(pip_exe):
        print(f"\\n[ERROR] Cannot find pip at {pip_exe}. Virtual environment creation might have failed.")
        sys.exit(1)

    # 2. Upgrade pip
    run_cmd([python_exe, "-m", "pip", "install", "--upgrade", "pip"], "Upgrading pip...")

    # 3. Install packages
    packages = [
        "torch>=2.0.0",
        "pandas",
        "numpy",
        "scikit-learn",
        "matplotlib",
        "ipykernel",
        "jupyter"
    ]
    run_cmd([python_exe, "-m", "pip", "install"] + packages, "Installing required packages...")

    # 4. Register Kernel
    # python -m ipykernel install --user --name=snn_transformer --display-name="SNN Transformer"
    run_cmd(
        [python_exe, "-m", "ipykernel", "install", "--user", f"--name={KERNEL_NAME}", f"--display-name={DISPLAY_NAME}"],
        f"Registering Jupyter kernel '{DISPLAY_NAME}'..."
    )

    print("\\n============================================================")
    print(" Setup Complete! ✓")
    print("============================================================")
    print(f"You can now select the kernel '{DISPLAY_NAME}' when opening the notebooks.")
    print(f"If you are transferring this project to another machine, just run this script there!")

if __name__ == "__main__":
    main()
