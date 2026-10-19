@echo off
rem Paper Reader 一键启动（Windows）
rem 双击运行，或：start.bat [--no-browser]
cd /d "%~dp0"
python start.py %*
pause
