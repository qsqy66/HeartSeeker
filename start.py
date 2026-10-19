#!/usr/bin/env python3
"""一键启动 Paper Reader：后端 FastAPI + 前端 Vite。

用法：
    python start.py                # 启动后端+前端，并自动打开浏览器
    python start.py --no-browser   # 只启动，不打开浏览器

首次运行会自动：创建 .venv、安装后端依赖、安装前端依赖（若 node_modules 缺失）。
之后再次运行直接复用，速度很快。按 Ctrl+C 一键停止两个服务。
"""
from __future__ import annotations

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
BACKEND_PORT = int(os.environ.get("PAPER_READER_PORT", "8765"))
FRONTEND_PORT = int(os.environ.get("PAPER_READER_FRONTEND_PORT", "5173"))
BACKEND_URL = f"http://127.0.0.1:{BACKEND_PORT}"
FRONTEND_URL = f"http://127.0.0.1:{FRONTEND_PORT}"

_procs: list[tuple[str, subprocess.Popen]] = []


def log(tag: str, message: str) -> None:
    color = {"setup": "36", "ok": "32", "info": "37", "error": "31", "stop": "33"}.get(tag, "37")
    print(f"\033[{color}m[{tag}]\033[0m {message}", flush=True)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, **kwargs)


def ensure_venv() -> Path:
    py = venv_python()
    if not py.exists():
        log("setup", f"创建虚拟环境 {VENV} …")
        _run([sys.executable, "-m", "venv", str(VENV)], check=True)
    return py


def ensure_backend_deps(py: Path) -> None:
    req = ROOT / "backend" / "requirements.txt"
    marker = VENV / ".deps_ok"
    if marker.exists() and marker.stat().st_mtime >= req.stat().st_mtime:
        return
    log("setup", "安装后端依赖 (pip install -r backend/requirements.txt) …")
    _run([str(py), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(req)], check=True)
    marker.write_text("ok", encoding="utf-8")


def ensure_frontend_deps() -> None:
    if (ROOT / "node_modules").exists():
        return
    npm = shutil.which("npm") or ("npm.cmd" if os.name == "nt" else "npm")
    log("setup", "安装前端依赖 (npm install) …")
    _run([npm, "install"], check=True)


def _node_major() -> int | None:
    node = shutil.which("node")
    if not node:
        return None
    try:
        out = subprocess.run([node, "--version"], capture_output=True, text=True, timeout=10)
        if out.returncode != 0:
            return None
        return int(out.stdout.strip().lstrip("v").split(".")[0])
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def require_node() -> None:
    """启动前端前校验 Node.js 18+，缺失或过旧时给出明确提示。"""
    major = _node_major()
    if major is None:
        log("error", "未检测到 Node.js（需要 18+），前端无法启动。请先安装 Node.js，或运行 install 脚本。")
        raise SystemExit(1)
    if major < 18:
        log("error", f"Node.js 版本过低（当前 v{major}，需要 18+），前端无法启动。请升级后重试。")
        raise SystemExit(1)


def wait_http(url: str, timeout: float = 60.0, proc: subprocess.Popen | None = None) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(url, timeout=1.5):
                return True
        except (urllib.error.URLError, OSError):
            time.sleep(0.4)
    return False


def _spawn(cmd: list[str]) -> subprocess.Popen:
    if os.name == "nt":
        return subprocess.Popen(cmd, cwd=ROOT, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    return subprocess.Popen(cmd, cwd=ROOT, start_new_session=True)


def _stop_one(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                proc.terminate()
            except OSError:
                pass


def shutdown() -> None:
    for _, proc in reversed(_procs):
        _stop_one(proc)
    time.sleep(1.5)
    for _, proc in reversed(_procs):
        if proc.poll() is None:
            try:
                proc.kill()
            except OSError:
                pass


def _sigterm(signum, frame):
    log("stop", f"收到信号 {signum}，停止服务 …")
    shutdown()
    raise SystemExit(0)


def start_backend(py: Path) -> subprocess.Popen:
    log("backend", f"启动 {BACKEND_URL} …")
    cmd = [str(py), "-m", "uvicorn", "backend.app:app", "--host", "127.0.0.1", "--port", str(BACKEND_PORT)]
    proc = _spawn(cmd)
    _procs.append(("backend", proc))
    return proc


def start_frontend() -> subprocess.Popen:
    npm = "npm.cmd" if os.name == "nt" else "npm"
    log("frontend", f"启动 {FRONTEND_URL} …")
    cmd = [npm, "run", "dev", "--", "--host", "127.0.0.1", "--port", str(FRONTEND_PORT), "--strictPort"]
    proc = _spawn(cmd)
    _procs.append(("frontend", proc))
    return proc


def main() -> int:
    parser = argparse.ArgumentParser(description="一键启动 Paper Reader（后端 + 前端）")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        log("error", f"需要 Python 3.10+，当前为 {sys.version.split()[0]}")
        return 1

    # 收到 SIGTERM 时也干净停掉子进程，避免终端关闭/进程被 kill 后残留服务
    if hasattr(signal, "SIGTERM"):
        try:
            signal.signal(signal.SIGTERM, _sigterm)
        except (ValueError, OSError):
            pass

    # 已在运行的服务直接复用，只启动缺失的部分
    backend_up = wait_http(f"{BACKEND_URL}/api/health", timeout=1.0)
    frontend_up = wait_http(FRONTEND_URL, timeout=1.0)
    if backend_up and frontend_up:
        log("ok", "Paper Reader 已在运行，直接打开浏览器。")
        if not args.no_browser:
            webbrowser.open(FRONTEND_URL)
        return 0

    log("setup", "检查环境 …")
    py = None
    if not frontend_up:
        require_node()
    if not backend_up:
        py = ensure_venv()
        ensure_backend_deps(py)
    if not frontend_up:
        ensure_frontend_deps()

    backend_proc = None
    frontend_proc = None
    if not backend_up:
        backend_proc = start_backend(py)
    if not frontend_up:
        frontend_proc = start_frontend()

    if not backend_up:
        log("info", "等待后端就绪 …")
        if not wait_http(f"{BACKEND_URL}/api/health", proc=backend_proc):
            log("error", "后端启动失败：端口可能被占用或依赖未就绪，请查看上方日志。")
            shutdown()
            return 1
        log("ok", f"后端就绪：{BACKEND_URL}/api/health")

    if not frontend_up:
        log("info", "等待前端就绪 …")
        if not wait_http(FRONTEND_URL, proc=frontend_proc):
            log("error", "前端启动失败：端口可能被占用或依赖未就绪，请查看上方日志。")
            shutdown()
            return 1
        log("ok", f"前端就绪：{FRONTEND_URL}")

    if not args.no_browser:
        try:
            webbrowser.open(FRONTEND_URL)
        except Exception:
            pass

    print(f"\n✅ Paper Reader 已启动：{FRONTEND_URL}\n   按 Ctrl+C 停止两个服务。\n", flush=True)
    try:
        while True:
            for name, proc in _procs:
                code = proc.poll()
                if code is not None:
                    log("error", f"{name} 已退出（code={code}），正在停止另一个服务。")
                    shutdown()
                    return 1
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n", flush=True)
        log("stop", "收到退出信号，停止服务 …")
        shutdown()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        shutdown()
        raise SystemExit(0)
    except BaseException:
        shutdown()
        raise
