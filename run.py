#!/usr/bin/env python3
"""
KrishiSahyog - Single-command local development launcher.

Starts backend (FastAPI/uvicorn) and frontend (Vite) in parallel,
opens the browser when ready, and cleans up both on Ctrl+C.

Usage:
    python run.py

Requires: Python 3.10+, Node.js, npm
Backend:  http://127.0.0.1:8000
Frontend: http://localhost:5173
"""
import os
import sys
import time
import signal
import subprocess
import webbrowser
from pathlib import Path

# Project root (where run.py lives)
ROOT = Path(__file__).resolve().parent
BACKEND_DIR = ROOT / "backend"
FRONTEND_DIR = ROOT / "frontend"

def kill_process(proc: subprocess.Popen | None) -> None:
    """Terminate process and children (Windows-safe)."""
    if proc is None:
        return
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            proc.kill()
        except ProcessLookupError:
            pass


def main() -> int:
    backend_proc = None
    frontend_proc = None

    def cleanup(_signum=None, _frame=None):
        print("\nShutting down...")
        kill_process(backend_proc)
        kill_process(frontend_proc)
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, cleanup)

    # Ensure backend dir exists
    if not BACKEND_DIR.is_dir():
        print("Error: backend/ directory not found")
        return 1

    # Start backend
    print("Starting backend (http://127.0.0.1:8000)...")
    backend_cmd = [
        sys.executable, "-m", "uvicorn", "main:app",
        "--host", "127.0.0.1", "--port", "8000", "--reload"
    ]
    backend_proc = subprocess.Popen(
        backend_cmd,
        cwd=str(BACKEND_DIR),
        stdout=None,
        stderr=None,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )

    # Start frontend
    print("Starting frontend (http://localhost:5173)...")
    npm_cmd = ["npm", "run", "dev"]
    frontend_proc = subprocess.Popen(
        npm_cmd,
        cwd=str(FRONTEND_DIR),
        shell=sys.platform == "win32",
        stdout=None,
        stderr=None,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if sys.platform == "win32" else 0,
        env={**os.environ},
    )

    # Wait for both to be ready, then open browser
    time.sleep(4)
    try:
        import urllib.request
        urllib.request.urlopen("http://127.0.0.1:8000/health", timeout=2)
    except Exception:
        pass  # Backend may still be starting
    try:
        urllib.request.urlopen("http://localhost:5173", timeout=2)
    except Exception:
        pass
    time.sleep(1)
    webbrowser.open("http://localhost:5173")
    print("Browser opened at http://localhost:5173")
    print("Press Ctrl+C to stop both servers\n")

    # Keep running until Ctrl+C or either process exits
    try:
        while backend_proc.poll() is None and frontend_proc.poll() is None:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    cleanup()
    return 0


if __name__ == "__main__":
    sys.exit(main())
