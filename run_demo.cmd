@echo off
setlocal
pushd "%~dp0"
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" "run_demo.py" %*
) else (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 "run_demo.py" %*
    ) else (
        where python >nul 2>nul
        if not errorlevel 1 (
            python "run_demo.py" %*
        ) else (
            echo Python 3.11+ is required for the first setup. Install Python, then run this file again.
            popd
            exit /b 1
        )
    )
)
set "DEMO_EXIT_CODE=%errorlevel%"
popd
exit /b %DEMO_EXIT_CODE%
