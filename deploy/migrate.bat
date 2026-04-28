@echo off
REM Orchestra SDK — Run database migrations (Windows)
REM ===================================================
REM Usage:
REM   deploy\migrate.bat --config conductor_config.yaml
REM   deploy\migrate.bat --config conductor_config.yaml --dry-run

setlocal EnableDelayedExpansion

set "REPO_ROOT=%~dp0.."
set "VENV=%REPO_ROOT%\.venv"
set "ENV_FILE=%REPO_ROOT%\.env"
set "ORCHESTRA=%VENV%\Scripts\orchestra.exe"

if not exist "%VENV%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. Run: python deploy\setup.py
    exit /b 1
)

call "%VENV%\Scripts\activate.bat"

REM Load .env
if exist "%ENV_FILE%" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%ENV_FILE%") do (
        set "line=%%A"
        if not "!line!"=="" (
            set "first=!line:~0,1!"
            if not "!first!"=="#" set "%%A=%%B"
        )
    )
)

REM Check service role key
if "%SUPABASE_SERVICE_ROLE_KEY%"=="" (
    echo [WARN] SUPABASE_SERVICE_ROLE_KEY is not set.
    echo        Migrations require the service role key.
    echo        Add it to .env or set it in the environment.
    exit /b 1
)

echo Running Orchestra migrations ...
"%ORCHESTRA%" migrate --env "%ENV_FILE%" %*
endlocal
