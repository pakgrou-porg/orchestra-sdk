@echo off
REM Orchestra SDK — Windows launcher
REM ===================================
REM Activates the virtual environment, loads .env, and passes all arguments
REM to the `orchestra` CLI.
REM
REM Usage (from repo root):
REM   deploy\orchestra.bat run --config conductor_config.yaml
REM   deploy\orchestra.bat status --config conductor_config.yaml
REM   deploy\orchestra.bat migrate --config conductor_config.yaml
REM   deploy\orchestra.bat check
REM   deploy\orchestra.bat setup

setlocal EnableDelayedExpansion

set "REPO_ROOT=%~dp0.."
set "VENV=%REPO_ROOT%\.venv"
set "ENV_FILE=%REPO_ROOT%\.env"
set "PYTHON=%VENV%\Scripts\python.exe"
set "ORCHESTRA=%VENV%\Scripts\orchestra.exe"

REM ── Check virtual environment ──────────────────────────────────────────────
if not exist "%VENV%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found at %VENV%
    echo         Run: python deploy\setup.py
    exit /b 1
)

REM ── Activate virtual environment ───────────────────────────────────────────
call "%VENV%\Scripts\activate.bat"

REM ── Load .env ──────────────────────────────────────────────────────────────
if exist "%ENV_FILE%" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
        set "line=%%A"
        REM Skip blank lines and comments
        if not "!line!"=="" (
            set "first_char=!line:~0,1!"
            if not "!first_char!"=="#" (
                set "%%A=%%B"
            )
        )
    )
) else (
    echo [WARN] .env not found at %ENV_FILE%
    echo        Run: python deploy\setup.py  or  copy deploy\.env.example .env
)

REM ── Dispatch ──────────────────────────────────────────────────────────────
if "%~1"=="" goto :usage

set "COMMAND=%~1"
shift

if /i "%COMMAND%"=="setup" (
    "%PYTHON%" "%REPO_ROOT%\deploy\setup.py" %*
    goto :eof
)

if /i "%COMMAND%"=="check" (
    "%PYTHON%" "%REPO_ROOT%\deploy\check.py" --env "%ENV_FILE%" %*
    goto :eof
)

REM Default: pass through to orchestra CLI
"%ORCHESTRA%" %COMMAND% --env "%ENV_FILE%" %*
goto :eof

:usage
echo Orchestra SDK launcher
echo.
echo Usage: deploy\orchestra.bat ^<command^> [options]
echo.
echo Commands:
echo   setup                   Run the interactive setup wizard
echo   check                   Run the health-check validator
echo   run     --config FILE   Start a Conductor session
echo   status  --config FILE   Show session status
echo   migrate --config FILE   Apply database migrations
echo   inspect --config FILE   Inspect session memories and git log
echo   reset   --config FILE   Revert workspace to a previous iteration
echo.
echo Examples:
echo   deploy\orchestra.bat run --config conductor_config.yaml
echo   deploy\orchestra.bat status --config conductor_config.yaml --all
endlocal
