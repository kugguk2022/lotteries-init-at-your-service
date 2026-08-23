@echo off
setlocal
cd /d "%~dp0"
set "PYTHON=python"

where %PYTHON% >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python executable not found on PATH.
    exit /b 1
)

echo ==================================================
echo [EUROMILLIONS] Starting Pipeline
echo ==================================================

:: 1. FETCH DRAWS
:: Use the full-history archive source and bypass cache so each run can refresh
:: to the latest published draw instead of reusing an old cached snapshot.
echo.
echo [1/7] Fetching full EuroMillions history...
%PYTHON% -m euromillions.get_draws --out data/euromillions.csv --source archive --allow-stale --no-cache
if %errorlevel% neq 0 (
    echo [ERROR] get_draws failed.
    exit /b %errorlevel%
)

:: 2. LOTTO LAB (Agent/Discriminator Analysis)
echo.
echo [2/7] Running Lotto Lab Agent...
%PYTHON% euromillions_agent/lotto_lab.py --csv data/euromillions.csv --mode all --outdir outputs/euromillions/lottolab --debug
if %errorlevel% neq 0 (
    echo [ERROR] lotto_lab failed.
    exit /b %errorlevel%
)

:: 3. PHASE 2 SOBOL - FEATURES
echo.
echo [3/7] Extracting Phase 2 Features (Sobol)...
%PYTHON% euromillions_agent/phase2_sobol.py features --infile data/euromillions.csv --outdir outputs/euromillions/features --main-k 5 --star-k 2 --include-current
if %errorlevel% neq 0 (
    echo [ERROR] phase2_sobol features failed.
    exit /b %errorlevel%
)

:: 4. GARCH-X (Seasonal volatility on POI)
echo.
echo [4/7] Running Seasonal GARCH-X...
%PYTHON% -m euromillions.garchx --history data/euromillions.csv --poi outputs/euromillions/features/poi.csv --out-dir outputs/euromillions/garchx
if %errorlevel% neq 0 (
    echo [ERROR] garchx failed.
    exit /b %errorlevel%
)

:: 5. GROK (Transformer Training)
echo.
echo [5/7] Training Grok Transformer Model...
%PYTHON% euromillions/grok.py --g outputs/euromillions/features/g1.csv --poi outputs/euromillions/features/poi.csv --out-dir outputs/euromillions/grok --epochs 80
if %errorlevel% neq 0 (
    echo [ERROR] grok failed.
    exit /b %errorlevel%
)

:: 6. PHASE 2 SOBOL - TICKETS
echo.
echo [6/7] Generating Sobol Tickets...
%PYTHON% euromillions_agent/phase2_sobol.py tickets --out outputs/euromillions/sobol_tickets.csv --n 50 --main-n 50 --main-k 5 --star-n 12 --star-k 2
if %errorlevel% neq 0 (
    echo [ERROR] phase2_sobol tickets failed.
    exit /b %errorlevel%
)

:: 7. INFER (Frequency Candidates)
echo.
echo [7/7] Generating Frequency Candidates (Infer)...
%PYTHON% euromillions/infer.py --history data/euromillions.csv --n 20 --out outputs/euromillions/infer_candidates.csv --max-ball 50 --max-star 12 --num-balls 5 --num-stars 2 --ball-prefix ball_ --star-prefix star_
if %errorlevel% neq 0 (
    echo [ERROR] infer failed.
    exit /b %errorlevel%
)

echo.
echo ==================================================
echo [SUCCESS] Pipeline Completed!
echo Check outputs in outputs/euromillions/
echo ==================================================
endlocal
pause
