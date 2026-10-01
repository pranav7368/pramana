@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\docker_local.ps1" %*
exit /b %errorlevel%
