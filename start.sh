#!/usr/bin/env bash
# 觅心（HeartSeeker_Med）一键启动（Linux / macOS）
# 用法：./start.sh [--no-browser]
set -e
cd "$(dirname "$0")"
exec python3 start.py "$@"
