@echo off
setlocal

cd /d "%~dp0"

REM Keep all Hugging Face and PyTorch caches on this drive (D:) instead of C:
set "HF_HOME=%~dp0models_cache\huggingface"
set "HUGGINGFACE_HUB_CACHE=%~dp0models_cache\huggingface\hub"
set "TORCH_HOME=%~dp0models_cache\torch"
set "PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,garbage_collection_threshold:0.8"
if not exist "%HF_HOME%" mkdir "%HF_HOME%"
if not exist "%TORCH_HOME%" mkdir "%TORCH_HOME%"

if not exist ".venv\Scripts\activate.bat" goto :missing_venv

echo Activating .venv and launching Irodori-TTS Japanese Studio...
call ".venv\Scripts\activate.bat"
if errorlevel 1 goto :activate_fail

python app.py
if errorlevel 1 goto :app_fail

pause
exit /b 0

:missing_venv
echo [ERROR] Virtual environment ".venv" not found.
echo         Please run setup_windows.bat first.
pause
exit /b 1

:activate_fail
echo [ERROR] Failed to activate .venv.
pause
exit /b 1

:app_fail
echo.
echo [ERROR] Application exited with an error. See details above.
pause
exit /b 1
