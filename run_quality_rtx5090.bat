@echo off
title crispz-krea - RTX 5090 (local)
cd /d "%~dp0"
echo ============================================
echo  crispz-krea - RTX 5090 (local 127.0.0.1)
echo ============================================
echo.
REM Optimisations CUDA (sans danger, BF16)
set NVIDIA_TF32_OVERRIDE=1
set CUDA_CACHE_MAXSIZE=4294967296
set CUDA_AUTO_BOOST=1
set CUDA_DEVICE_ORDER=PCI_BUS_ID
set GRADIO_SERVER_PORT=7860
REM Console UTF-8 (evite les crashs cp1252 sur les barres de progression HF)
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
REM === GATED: FLUX.1-Krea-dev exige une auth HF. Une fois pour toutes, accepte la licence
REM sur https://huggingface.co/black-forest-labs/FLUX.1-Krea-dev puis: huggingface-cli login
REM --token hf_xxx  (sinon: 401 GatedRepoError au 1er chargement). ===
REM === VRAM: FLUX (~12B transformer + T5/CLIP) ~33 Go > 32 Go -> offload 'model' requis. ===
set CZ_OFFLOAD=model
REM Delegue au run.bat (detection venv + ESRGAN_DIR + lancement)
call "%~dp0run.bat" %*
