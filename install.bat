@echo off
setlocal enabledelayedexpansion
rem Paper Reader 环境安装脚本（Windows）
rem 自动检测并安装 Python 3.12 与 Node.js LTS（通过 winget），随后调用 start.py 部署并自动打开浏览器。
rem 只装缺失的部分；winget 需要 Windows 10 1809+ / Windows 11。
cd /d "%~dp0"

echo [install] 检查 Python / Node.js ...

rem ---- Python（优先用官方 py 启动器，避免 Windows Store 占位符干扰）----
set "PY_CMD="
py -3 --version >nul 2>nul
if not errorlevel 1 (
    set "PY_CMD=py -3"
) else (
    python --version >nul 2>nul
    if not errorlevel 1 set "PY_CMD=python"
)

if not defined PY_CMD (
    echo [install] 未检测到 Python，通过 winget 安装 Python 3.12 ...
    winget install -e --id Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    if errorlevel 1 goto :no_winget
    set "PY_CMD=python"
)

rem ---- Node.js ----
node --version >nul 2>nul
if errorlevel 1 (
    echo [install] 未检测到 Node.js，通过 winget 安装 Node.js LTS ...
    winget install -e --id OpenJS.NodeJS.LTS --accept-package-agreements --accept-source-agreements
    if errorlevel 1 goto :no_winget
)

rem winget 新装的程序在当前会话可能不在 PATH，手动补上常见安装路径
set "PATH=%LOCALAPPDATA%\Programs\Python\Python312\Scripts;%LOCALAPPDATA%\Programs\Python\Python312;%ProgramFiles%\Python312\Scripts;%ProgramFiles%\Python312;%ProgramFiles%\nodejs;%PATH%"

echo [install] 环境就绪，开始部署（自动打开浏览器）...
%PY_CMD% start.py %*
goto :eof

:no_winget
echo.
echo [error] 未找到 winget（Windows 10 1809+ / Windows 11 自带）。
echo         请手动安装后重试：
echo           Python: https://www.python.org/downloads/
echo           Node:   https://nodejs.org/
echo.
pause
exit /b 1
