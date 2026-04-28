@echo off
REM Orchestra SDK — Build the Musician training container (Windows)
REM ================================================================
REM Usage:
REM   deploy\build_musician.bat           -- NVIDIA CUDA image
REM   deploy\build_musician.bat --cpu     -- CPU-only image

setlocal

set "REPO_ROOT=%~dp0.."
set "TAG=orchestra-musician:latest"
set "DOCKERFILE=%REPO_ROOT%\docker\Dockerfile.musician"
set "CPU_MODE=0"

:parse_args
if "%~1"=="--cpu" (
    set "CPU_MODE=1"
    set "TAG=orchestra-musician-cpu:latest"
    set "DOCKERFILE=%REPO_ROOT%\examples\synthetic\Dockerfile"
    shift
    goto parse_args
)
if "%~1"=="--tag" (
    set "TAG=%~2"
    shift & shift
    goto parse_args
)

REM ── Pre-flight ────────────────────────────────────────────────────────────
where docker >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker not found. Install from https://docs.docker.com/get-docker/
    exit /b 1
)

docker info >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Docker daemon is not running. Start Docker Desktop and retry.
    exit /b 1
)

if not exist "%DOCKERFILE%" (
    echo [ERROR] Dockerfile not found: %DOCKERFILE%
    exit /b 1
)

REM ── NVIDIA check ─────────────────────────────────────────────────────────
if "%CPU_MODE%"=="0" (
    where nvidia-smi >nul 2>&1
    if errorlevel 1 (
        echo [WARN] nvidia-smi not found. Use --cpu for a CPU-only build.
    ) else (
        echo NVIDIA GPU detected:
        nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    )
)

REM ── Build ─────────────────────────────────────────────────────────────────
echo.
echo Building Docker image: %TAG%
echo   Dockerfile: %DOCKERFILE%
echo   Context:    %REPO_ROOT%
echo.

docker build ^
    --file "%DOCKERFILE%" ^
    --tag "%TAG%" ^
    "%REPO_ROOT%"

if errorlevel 1 (
    echo [ERROR] Docker build failed.
    exit /b 1
)

echo.
echo [OK] Image built: %TAG%
echo.
echo Test the image:
echo   docker run --rm %TAG% python3 -c "import torch; print(torch.__version__)"
endlocal
