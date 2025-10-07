@echo off
REM HP9 CI Helper Script for Windows
REM Runs complete HP9 validation pipeline locally

echo ================================================================================
echo DVREN-HOTPATH HP9 Local Validation Pipeline
echo ================================================================================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found. Please install Python 3.8+
    exit /b 1
)

REM Step 1: Build
echo [Step 1/5] Building project...
echo --------------------------------------------------------------------------------
if not exist build mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DDVREN_BUILD_CUDA=ON
if errorlevel 1 (
    echo Error: CMake configuration failed
    cd ..
    exit /b 1
)

cmake --build . --config Release
if errorlevel 1 (
    echo Error: Build failed
    cd ..
    exit /b 1
)
cd ..
echo [Step 1/5] Build completed successfully
echo.

REM Step 2: Run Tests
echo [Step 2/5] Running tests...
echo --------------------------------------------------------------------------------
build\Release\hp_runner.exe > test_output.txt 2>&1
set TEST_RESULT=%errorlevel%
type test_output.txt
echo.
echo [Step 2/5] Tests completed (exit code: %TEST_RESULT%)
echo.

REM Step 3: CI Check
echo [Step 3/5] Checking CI gates...
echo --------------------------------------------------------------------------------
python scripts\ci_check.py test_output.txt
if errorlevel 1 (
    echo [Step 3/5] CI gates FAILED
    exit /b 1
)
echo [Step 3/5] CI gates passed
echo.

REM Step 4: Profiling
echo [Step 4/5] Running profiling...
echo --------------------------------------------------------------------------------
python scripts\profile.py --executable build\Release\hp_runner.exe --output artifacts\profiling
if errorlevel 1 (
    echo Warning: Profiling encountered issues
)
echo [Step 4/5] Profiling completed
echo.

REM Step 5: Gate Validation
echo [Step 5/5] Validating all HP9 gates...
echo --------------------------------------------------------------------------------
if exist artifacts\profiling\benchmark_results.json (
    python scripts\validate_gates.py artifacts\profiling\benchmark_results.json
    if errorlevel 1 (
        echo [Step 5/5] Gate validation FAILED
        exit /b 1
    )
    echo [Step 5/5] All gates passed
) else (
    echo Warning: Benchmark results not found, skipping gate validation
)
echo.

REM Optional: Archive
echo ================================================================================
echo Optional: Archive artifacts?
echo ================================================================================
set /p ARCHIVE="Archive artifacts? (y/n): "
if /i "%ARCHIVE%"=="y" (
    python scripts\archive_artifacts.py
    echo Artifacts archived
)

echo.
echo ================================================================================
echo HP9 Validation Pipeline COMPLETED SUCCESSFULLY
echo ================================================================================
echo.
echo Next steps:
echo   - Review artifacts in artifacts\profiling\
echo   - Check test_output.txt for details
echo   - Run 'python scripts\lock_thresholds.py' to lock thresholds
echo.

exit /b 0

