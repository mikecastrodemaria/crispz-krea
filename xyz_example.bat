@echo off
REM crispz-studio - X/Y/Z grid CLI example (see README_CLI.md).
REM Runs a small 2x2 comparison grid (Steps x Guidance) for a prompt, then prints
REM the annotated contact-sheet path (saved in out\xyz_<timestamp>\).
REM
REM Usage:
REM   xyz_example.bat                          -> demo prompt
REM   xyz_example.bat "your prompt here"       -> your prompt
REM
REM Tips (edit the --xyz lines below):
REM   - up to 3 axes (X, Y, Z), values comma-separated, quotes protect commas
REM   - axes: Checkpoint, Sampler, Schedule, Steps, Guidance, Seed, ESRGAN model,
REM     Factor, Denoise, Tile, Refine tile, LoRA weight, Performance, Prompt S/R
REM   - Ctrl+C assembles a partial sheet with the cells rendered so far

setlocal
cd /d "%~dp0"

set "PROMPT=%~1"
if "%PROMPT%"=="" set "PROMPT=a red fox in a snowy forest, cinematic light"

REM Python: .venv if present, else system
set "RUNPY=python"
if exist ".venv\Scripts\python.exe" set "RUNPY=.venv\Scripts\python.exe"

echo [xyz example] prompt: %PROMPT%
echo [xyz example] grid  : Steps=4,8  x  Guidance=0,3.5  (4 renders + 1 sheet)
echo.

%RUNPY% app.py --cli --txt2img --prompt "%PROMPT%" ^
    --xyz "Steps=4,8" ^
    --xyz "Guidance=0, 3.5" ^
    --save-mode local --output-dir out
if errorlevel 1 (
    echo.
    echo [xyz example] FAILED - see the error above.
    exit /b 1
)

echo.
echo [xyz example] done. The last line above is the annotated sheet path.
endlocal
