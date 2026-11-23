@echo off
setlocal

echo ============================================================================
echo  Irodori-TTS v4-Large Japanese Studio - Windows Setup [CUDA 12.x / RTX 4060]
echo ============================================================================
echo.

cd /d "%~dp0"

REM Keep all pip temp files and Hugging Face / PyTorch caches on this drive (D:) instead of C:
set "PIP_NO_CACHE_DIR=1"
set "HF_HOME=%~dp0models_cache\huggingface"
set "HUGGINGFACE_HUB_CACHE=%~dp0models_cache\huggingface\hub"
set "TORCH_HOME=%~dp0models_cache\torch"
set "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8"
set "TEMP=%~dp0.tmp_pip"
set "TMP=%~dp0.tmp_pip"
if not exist "%TEMP%" mkdir "%TEMP%"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%TORCH_HOME%" mkdir "%TORCH_HOME%"

REM 1. Verify Python is available
where python >nul 2>&1
if errorlevel 1 goto :no_python

for /f "tokens=*" %%i in ('python --version') do set "PYVER=%%i"
echo [INFO] Detected %PYVER%

REM Verify Git is available for installing dacvae and irodori-tts from GitHub
where git >nul 2>&1
if errorlevel 1 goto :no_git

REM 2. Create .venv with --system-site-packages if it does not exist
if exist ".venv\Scripts\activate.bat" goto :venv_exists
echo [STEP 1/5] Creating virtual environment .venv with --system-site-packages...
python -m venv .venv --system-site-packages
if errorlevel 1 goto :venv_fail
goto :activate_venv

:venv_exists
echo [STEP 1/5] Virtual environment .venv already exists. Reusing it.

:activate_venv
REM 3. Activate virtual environment
echo [STEP 2/5] Activating .venv...
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :activate_fail

REM Upgrade pip/setuptools/wheel inside .venv safely
python -m pip install --upgrade pip setuptools wheel

REM 4. Check if CUDA-enabled PyTorch 2.10.x is already accessible
REM    (irodori-tts + torchao 0.16.x quantized checkpoints require PyTorch 2.10.x CUDA)
echo [STEP 3/5] Checking for CUDA-enabled PyTorch 2.10.x...
python -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; v=tuple(map(int, torch.__version__.split('+')[0].split('.')[:2])); assert v == (2, 10), f'Need torch 2.10.x for torchao 0.16, found {torch.__version__}'; print('[OK] Found PyTorch', torch.__version__, 'with CUDA', torch.version.cuda, 'on', torch.cuda.get_device_name(0))"
if errorlevel 1 goto :install_torch

REM Compatible CUDA torch 2.10.x found - verify torchaudio is also 2.10.x
python -c "import torchaudio; v=tuple(map(int, torchaudio.__version__.split('+')[0].split('.')[:2])); assert v == (2, 10)" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing matching torchaudio 2.10.x with CUDA 12.8 support...
    pip install --upgrade "torchaudio>=2.10.0,<2.11.0" --index-url https://download.pytorch.org/whl/cu128
)
goto :install_reqs

:install_torch
echo [INFO] Compatible CUDA PyTorch 2.10.x was not found in global/venv environment.
echo [INFO] Installing PyTorch 2.10.x and torchaudio 2.10.x [CUDA 12.8] into .venv...
pip install --upgrade "torch>=2.10.0,<2.11.0" "torchaudio>=2.10.0,<2.11.0" --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 goto :torch_fail

:install_reqs
REM 5. Install PyPI dependencies from requirements.txt into .venv
echo [STEP 4/5] Installing PyPI dependencies from requirements.txt...
pip install -r requirements.txt
if errorlevel 1 goto :reqs_fail

REM 6. Install GitHub-only audio codec (dacvae) and official Irodori-TTS runtime
REM    using --no-deps so pip does not fail searching PyPI for `dacvae`,
REM    does not pull legacy `protobuf<3.20` from `descript-audiotools`,
REM    and does not overwrite CUDA-enabled PyTorch.
echo [STEP 5/5] Installing dacvae and irodori-tts from GitHub...
pip install --no-deps descript-audiotools
if errorlevel 1 goto :github_fail
pip install --no-deps "git+https://github.com/facebookresearch/dacvae.git"
if errorlevel 1 goto :github_fail
pip install --no-deps "git+https://github.com/Aratako/Irodori-TTS.git"
if errorlevel 1 goto :github_fail

echo [INFO] Verifying core imports...
python -c "import torch, torchao, dacvae, irodori_tts.inference_runtime, gradio; print('[OK] Core imports verified! PyTorch:', torch.__version__, '| torchao:', torchao.__version__, '| CUDA:', torch.cuda.is_available())"
if errorlevel 1 goto :verify_fail

REM Clean temporary pip build/unpack directory on D:
if exist "%~dp0.tmp_pip" rmdir /s /q "%~dp0.tmp_pip" >nul 2>&1

echo.
echo [SUCCESS] Setup complete! Launching Irodori-TTS Japanese Studio...
echo.
python app.py
if errorlevel 1 goto :app_fail

echo.
echo Application closed cleanly.
pause
exit /b 0

:no_python
echo [ERROR] Python was not found on PATH.
echo         Please install 64-bit Python 3.10 - 3.12 and check "Add Python to PATH".
pause
exit /b 1

:no_git
echo [ERROR] Git was not found on PATH.
echo         Please install Git for Windows from https://git-scm.com/download/win
echo         and reopen this script.
pause
exit /b 1

:venv_fail
echo [ERROR] Failed to create virtual environment .venv.
pause
exit /b 1

:activate_fail
echo [ERROR] Failed to activate .venv.
pause
exit /b 1

:torch_fail
echo [ERROR] Failed to install CUDA-enabled PyTorch 2.10.x.
pause
exit /b 1

:reqs_fail
echo [ERROR] Failed to install dependencies from requirements.txt.
echo         Check the error log above for details.
pause
exit /b 1

:github_fail
echo [ERROR] Failed to install dacvae or irodori-tts from GitHub.
echo         Verify Git is installed and GitHub is reachable, then re-run setup_windows.bat.
pause
exit /b 1

:verify_fail
echo [ERROR] Core Python module verification failed. See traceback above.
pause
exit /b 1

:app_fail
echo.
echo [ERROR] Application exited with an error. See details above.
pause
exit /b 1
