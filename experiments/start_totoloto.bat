@echo off
setlocal
echo ==================================================
echo [TOTOLOTO] Starting Pipeline
echo ==================================================

:: 1. FETCH DRAWS
echo.
echo [1/5] Fetching Totoloto draws...
:: Note: totoloto_get_draws.py does not support --allow-stale/partial.
python totoloto/totoloto_get_draws.py --out data/totoloto.csv 
if %errorlevel% neq 0 (
    echo [ERROR] get_draws failed.
    echo [WARNING] Totoloto fetch failed. If existing data is present, the pipeline MIGHT continue if the file remains valid.
    if exist data\totoloto.csv (
        echo [INFO] Found existing data/totoloto.csv. Continuing...
    ) else (
        exit /b %errorlevel%
    )
)


:: 2. PHASE 2 SOBOL - FEATURES
echo.
echo [2/5] Extracting Phase 2 Features (Sobol)...
:: Totoloto: 5 mains, 1 star (lucky number)
python euromillions_agent/phase2_sobol.py features --infile data/totoloto.csv --outdir outputs/totoloto/features --main-k 5 --star-k 1 --include-current
if %errorlevel% neq 0 (
    echo [ERROR] phase2_sobol features failed.
    exit /b %errorlevel%
)

:: 3. GROK (Transformer Training)
echo.
echo [3/5] Training Grok Transformer Model...
:: Using the features extracted in step 2
python euromillions/grok.py --g outputs/totoloto/features/g1.csv --poi outputs/totoloto/features/poi.csv --out-dir outputs/totoloto/grok --epochs 80
if %errorlevel% neq 0 (
    echo [ERROR] grok failed.
    exit /b %errorlevel%
)

:: 4. PHASE 2 SOBOL - TICKETS
echo.
echo [4/5] Generating Sobol Tickets...
:: Totoloto: 49 balls, 13 lucky numbers (stars)
python euromillions_agent/phase2_sobol.py tickets --out outputs/totoloto/sobol_tickets.csv --n 50 --main-n 49 --main-k 5 --star-n 13 --star-k 1
if %errorlevel% neq 0 (
    echo [ERROR] phase2_sobol tickets failed.
    exit /b %errorlevel%
)

:: 5. INFER (Frequency Candidates)
echo.
echo [5/5] Generating Frequency Candidates (Infer)...
:: Totoloto CSVs usually use ball_1...ball_5 and star_1 (or similar)
python euromillions/infer.py --history data/totoloto.csv --n 20 --out outputs/totoloto/infer_candidates.csv --max-ball 49 --max-star 13 --num-balls 5 --num-stars 1 --ball-prefix ball_ --star-prefix star_
if %errorlevel% neq 0 (
    echo [ERROR] infer failed.
    exit /b %errorlevel%
)

echo.
echo ==================================================
echo [SUCCESS] Pipeline Completed!
echo Check outputs in outputs/totoloto/
echo ==================================================
endlocal
pause
