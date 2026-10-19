#!/usr/bin/env bash
# Paper Reader 环境安装脚本（Linux）
# 自动检测并安装 Python 3.10+ 与 Node.js 18+，随后调用 start.py 一键部署（自动打开浏览器）。
# 安装系统包需要 sudo 权限；只装缺失/过旧的部分，不会动已满足要求的环境。
#
# 用法：
#   ./install.sh              # 装环境 + 部署 + 打开浏览器
#   ./install.sh --no-browser # 只装环境 + 部署，不打开浏览器
set -euo pipefail
cd "$(dirname "$0")"

log() { printf '\033[36m[install]\033[0m %s\n' "$*"; }
err() { printf '\033[31m[error]\033[0m %s\n' "$*" >&2; }

# 版本比较：ver_ge "3.10.12" "3.10" -> 真
ver_ge() {
    [ "$(printf '%s\n%s\n' "$2" "$1" | sort -V | head -n1)" = "$2" ]
}

PM=""
INSTALL=""
UPDATE=""

detect_pm() {
    if command -v apt-get >/dev/null 2>&1; then
        PM=apt; UPDATE="sudo apt-get update -y"; INSTALL="sudo apt-get install -y"
    elif command -v dnf >/dev/null 2>&1; then
        PM=dnf; UPDATE="sudo dnf -y makecache || true"; INSTALL="sudo dnf install -y"
    elif command -v yum >/dev/null 2>&1; then
        PM=yum; UPDATE="true"; INSTALL="sudo yum install -y"
    elif command -v pacman >/dev/null 2>&1; then
        PM=pacman; UPDATE="true"; INSTALL="sudo pacman -S --needed --noconfirm"
    else
        PM=unknown
    fi
}

need_sudo_hint() {
    err "未检测到受支持的包管理器（apt / dnf / yum / pacman），或缺少 sudo。"
    err "请手动安装 Python 3.10+ 与 Node.js 18+ 后，运行：python3 start.py"
    exit 1
}

ensure_python() {
    if command -v python3 >/dev/null 2>&1; then
        local v
        v="$(python3 -c 'import sys;print("%d.%d"%sys.version_info[:2])' 2>/dev/null || echo 0.0)"
        if ver_ge "$v" "3.10"; then
            log "Python $v 已就绪"
            return 0
        fi
        log "检测到 Python $v（<3.10），将安装/升级新版。"
    else
        log "未检测到 Python，将安装 Python 3。"
    fi
    [ "$PM" = unknown ] && need_sudo_hint
    log "通过 $PM 安装 Python 3 与 venv …"
    $UPDATE
    case "$PM" in
        apt)    $INSTALL python3 python3-venv python3-pip ;;
        dnf|yum) $INSTALL python3 python3-pip ;;
        pacman) $INSTALL python ;;
    esac
}

ensure_node() {
    if command -v node >/dev/null 2>&1; then
        local v
        v="$(node --version | sed 's/^v//')"
        if ver_ge "$v" "18.0"; then
            log "Node.js $v 已就绪"
            return 0
        fi
        log "检测到 Node.js $v（<18），将安装新版。"
    else
        log "未检测到 Node.js，将安装 Node.js 与 npm。"
    fi
    [ "$PM" = unknown ] && need_sudo_hint
    log "通过 $PM 安装 Node.js 与 npm …"
    $UPDATE
    case "$PM" in
        apt)    $INSTALL nodejs npm ;;
        dnf|yum) $INSTALL nodejs npm ;;
        pacman) $INSTALL nodejs npm ;;
    esac
    # Debian/Ubuntu 的 apt 仓库 nodejs 可能仍 <18，明确提示
    if command -v node >/dev/null 2>&1; then
        local v
        v="$(node --version | sed 's/^v//')"
        if ! ver_ge "$v" "18.0"; then
            err "系统仓库的 Node.js 仍为 $v（<18），Vite 需要 18+。"
            err "请改用 NodeSource 或 nvm 安装新版 Node："
            err "  curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - && sudo apt-get install -y nodejs"
            exit 1
        fi
    fi
}

detect_pm
ensure_python
ensure_node

log "环境就绪，开始部署（自动打开浏览器）…"
exec python3 start.py "$@"
