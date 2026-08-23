@echo off
setlocal
echo ==================================================
echo [EURODREAMS] Starting Pipeline
echo ==================================================

:: 1. FETCH DRAWS
echo.
echo [1/5] Fetching EuroDreams draws...
python eurodreams/eurodreams_get_draws.py --out data/eurodreams.csv --allow-stale
if %errorlevel% neq 0 (
    echo [ERROR] get_draws failed.
    exit /b %errorlevel%
)


:: 2. PHASE 2 SOBOL - FEATURES
echo.
echo [2/5] Extracting Phase 2 Features (Sobol)...
:: EuroDreams: 6 mains, 1 star (dream number)
:: Note: check if phase2_sobol supports 6 mains. Code review showed main_k arg, but let's hope logic isn't hardcoded to 5 internally.
:: Checked code: "if k != 5: raise ValueError(...)" in build_pair_features. 
:: Fix required: phase2_sobol.py currently hardcodes 5 balls.
echo [WARNING] phase2_sobol.py currently supports only 5 main balls. EuroDreams has 6. 
echo [WARNING] Skipping Sobol/Grok steps involving 6-ball logic to prevent crash. You may need to update phase2_sobol.py logic.

:: 3. INFER (Frequency Candidates)
echo.
echo [3/5] Generating Frequency Candidates (Infer)...
:: Eurodreams typically uses n1..n6 and dream
python euromillions/infer.py --history data/eurodreams.csv --n 20 --out outputs/eurodreams/infer_candidates.csv --max-ball 40 --max-star 5 --num-balls 6 --num-stars 1 --ball-prefix n --star-prefix dream
if %errorlevel% neq 0 (
    echo [ERROR] infer failed.
    exit /b %errorlevel%
)

echo.
echo ==================================================
echo [PARTIAL SUCCESS] Pipeline Completed (Grok/Sobol skipped due to ball count incompatibility)!
echo Check outputs in outputs/eurodreams/
echo ==================================================
endlocal
pause
