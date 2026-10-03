@echo off
title Black Box Universal Proxy & Flight Recorder
color 0A

echo ======================================================================
echo           BLACK BOX - AI Agent Flight Recorder Proxy
echo ======================================================================
echo.
echo Starting Black Box Proxy Server on http://127.0.0.1:8000 ...
echo Database: blackbox_traces.db
echo.

:: Move to script directory
cd /d "%~dp0"

:: Set PYTHONPATH to current directory
set PYTHONPATH=.

:: Launch uvicorn server
python -m uvicorn blackbox.proxy.server:app --host 0.0.0.0 --port 8000 --reload

pause
