@echo off
setlocal enabledelayedexpansion
cd /d "%~dp0\.."
set PYTHONPATH=%CD%

set CONFIG=configs\baseline.yaml

echo.
echo # 1) data prep (one-time)
python -m src.data.download_tinystories --config %CONFIG%
if errorlevel 1 exit /b %errorlevel%

python -m src.data.train_tokenizer --config %CONFIG%
if errorlevel 1 exit /b %errorlevel%

python -m src.data.pretokenize_and_pack --config %CONFIG%
if errorlevel 1 exit /b %errorlevel%

echo.
echo # 2) train
python -m src.train --config %CONFIG%
if errorlevel 1 exit /b %errorlevel%

echo.
echo # 3) eval (optional standalone)
REM Pass ckpt path as first arg, e.g.:
REM scripts\run_baseline.bat artifacts\runs\...\checkpoints\best_model.pt
set CKPT=%1
if "%CKPT%"=="" goto :done

python -m src.eval --config %CONFIG% --ckpt %CKPT%
if errorlevel 1 exit /b %errorlevel%

:done
echo Done.
exit /b 0
