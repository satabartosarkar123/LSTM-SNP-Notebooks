@echo off
echo ============================================================
echo   Transformer Windows Setup (RTX 3050 GPU Support)
echo ============================================================
echo.

set VENV_DIR=transformer_venv_win

:: 1. Create Virtual Environment
if exist %VENV_DIR% (
    echo [✓] Virtual environment already exists.
) else (
    echo [*] Creating virtual environment in %VENV_DIR%...
    python -m venv %VENV_DIR%
)

:: 2. Upgrade PIP and Install Dependencies
echo [*] Upgrading pip...
%VENV_DIR%\Scripts\python.exe -m pip install --upgrade pip

echo [*] Installing Windows-specific dependencies (CUDA Support)...
%VENV_DIR%\Scripts\python.exe -m pip install -r requirements_windows.txt

:: 3. Register Kernel
echo [*] Registering Jupyter Kernel...
%VENV_DIR%\Scripts\python.exe -m ipykernel install --user --name transformer-win --display-name "Transformer-Windows (CUDA)"

echo.
echo ============================================================
echo   Setup Complete!
echo.
echo   1. Open your notebook in Jupyter or VS Code.
echo   2. Select the "Transformer-Windows (CUDA)" kernel.
echo   3. Verify GPU with: torch.cuda.is_available() 
echo ============================================================
pause
